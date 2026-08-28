from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import modal

TAG = "embodiedgen-v2.0.0-affordance-py310-cu126-torch280-sm89-v1"
OUT_ROOT = Path("/build-artifacts/embodiedgen-affordance")
WHEELS = Path("/tmp/affordance-wheels")

PINS = {
    "embodiedgen": "cc3015ca5ccdacf94df3428d9e65f79375982216",
    "graspgen": "a56d518f3b76ea2a432b5b838b3c68027d29be49",
    "hunyuan3d_part": "e96be065375438962375b55326416291342958a7",
    "torch_scatter": "2.1.2",
    "torch_scatter_wheel_sha256": "3585a1ef1f4886d037a76a21ff987fbcac354805dbae42c7992b0b6d7cf8ad54",
}
TORCH_SCATTER_WHEEL_URL = (
    "https://data.pyg.org/whl/torch-2.8.0%2Bcu126/"
    "torch_scatter-2.1.2%2Bpt28cu126-cp310-cp310-linux_x86_64.whl"
)
TORCH_SCATTER_WHEEL_NAME = "torch_scatter-2.1.2+pt28cu126-cp310-cp310-linux_x86_64.whl"

app = modal.App("modal-build-embodiedgen-affordance")
artifacts = modal.Volume.from_name("modal-3d-build-artifacts", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "gcc", "g++", "ninja-build")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "MAX_JOBS": "8",
            "CC": "gcc",
            "CXX": "g++",
            "FORCE_CUDA": "1",
        }
    )
    .run_commands(
        "python -m pip install --upgrade 'pip>=25' setuptools==80.10.2 wheel packaging ninja psutil",
        "python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126",
    )
)


def sh(cmd: str, cwd: str | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


def clone_at(url: str, revision: str, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", url], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "-q", "--depth=1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "-q", "--detach", "FETCH_HEAD"],
        check=True,
    )


def gitlink_at(repo: Path, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"HEAD:{path}"], text=True
    ).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wheel_shared_objects(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return sorted(name for name in zf.namelist() if name.endswith(".so"))


@app.function(
    image=image,
    volumes={"/build-artifacts": artifacts},
    timeout=60 * 60,
    cpu=8.0,
    memory=32768,
    max_containers=1,
)
def build_affordance_artifacts() -> dict:
    """CPU-only nvcc build of affordance CUDA wheels, staged before release."""
    output = OUT_ROOT / TAG
    manifest_path = output / f"{TAG}.manifest.json"
    if manifest_path.exists():
        raise RuntimeError(
            f"staged artifacts already exist for {TAG}; refusing to overwrite. "
            "Delete the staging directory or bump TAG."
        )

    # Do not create persistent staging until every expensive build succeeds. Modal may
    # preempt CPU builders and retry the same input; /tmp is fresh on retry, while Volumes
    # persist. Deferring the directory creation keeps the build naturally retry-safe.
    if output.exists():
        shutil.rmtree(output)
    shutil.rmtree(WHEELS, ignore_errors=True)
    WHEELS.mkdir(parents=True, exist_ok=True)

    env = (
        "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 "
        "TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8"
    )

    # Tie the native dependency pins to the exact EmbodiedGen source tree. This makes
    # the manifest's source pin meaningful and fails closed if upstream gitlinks change.
    embodiedgen = Path("/tmp/EmbodiedGen")
    clone_at("https://github.com/HorizonRobotics/EmbodiedGen.git", PINS["embodiedgen"], embodiedgen)
    expected_gitlinks = {
        "thirdparty/GraspGen": PINS["graspgen"],
        "thirdparty/Hunyuan3D-Part": PINS["hunyuan3d_part"],
    }
    for submodule_path, expected_revision in expected_gitlinks.items():
        actual_revision = gitlink_at(embodiedgen, submodule_path)
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"submodule pin mismatch for {submodule_path}: "
                f"{actual_revision} != {expected_revision}"
            )

    # Sonata / P3-SAM requires torch-scatter. PyG publishes an exact Torch 2.8 /
    # CUDA 12.6 / CPython 3.10 binary, so use it instead of wasting CPU on a rebuild.
    scatter_wheel = WHEELS / TORCH_SCATTER_WHEEL_NAME
    urllib.request.urlretrieve(TORCH_SCATTER_WHEEL_URL, scatter_wheel)
    scatter_hash = sha256(scatter_wheel)
    if scatter_hash != PINS["torch_scatter_wheel_sha256"]:
        raise RuntimeError(
            f"torch-scatter wheel hash mismatch: {scatter_hash} != "
            f"{PINS['torch_scatter_wheel_sha256']}"
        )

    # GraspGen pointnet2 CUDA extension from the exact upstream gitlink.
    graspgen = Path("/tmp/GraspGen")
    clone_at("https://github.com/NVlabs/GraspGen.git", PINS["graspgen"], graspgen)
    sh(
        f"{env} python -m pip wheel --no-deps --no-build-isolation . -w '{WHEELS}'",
        str(graspgen / "pointnet2_ops"),
    )

    # P3-SAM imports chamfer_3D at module load and otherwise JIT-compiles it.
    hunyuan_part = Path("/tmp/Hunyuan3D-Part")
    clone_at(
        "https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git",
        PINS["hunyuan3d_part"],
        hunyuan_part,
    )
    sh(
        f"{env} python -m pip wheel --no-deps --no-build-isolation . -w '{WHEELS}'",
        str(hunyuan_part / "P3-SAM" / "utils" / "chamfer3D"),
    )

    wheels = []
    for path in sorted(WHEELS.glob("*.whl")):
        shared_objects = wheel_shared_objects(path)
        if not shared_objects:
            raise RuntimeError(f"wheel has no compiled shared object: {path.name}")
        wheels.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "shared_objects": shared_objects,
            }
        )
    if len(wheels) != 3:
        raise RuntimeError(f"expected 3 wheels, got {[x['file'] for x in wheels]}")

    output.mkdir(parents=True, exist_ok=False)
    for path in WHEELS.glob("*.whl"):
        shutil.copy2(path, output / path.name)

    manifest = {
        "tag": TAG,
        "source": f"HorizonRobotics/EmbodiedGen@{PINS['embodiedgen']}",
        "python": "3.10",
        "ubuntu": "22.04",
        "cuda": "12.6.3",
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "build_gpu": None,
        "build_strategy": "CPU-only nvcc compile; stage to Modal Volume before L40S validation",
        "pins": PINS,
        "wheels": wheels,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sha_path = output / f"{TAG}.sha256"
    sha_path.write_text(
        "".join(f"{item['sha256']}  {item['file']}\n" for item in wheels)
        + f"{sha256(manifest_path)}  {manifest_path.name}\n"
    )
    artifacts.commit()
    print("AFFORDANCE_ARTIFACTS_STAGED", json.dumps(manifest), flush=True)
    return manifest


@app.function(
    image=image,
    volumes={"/build-artifacts": artifacts},
    gpu="L40S",
    timeout=15 * 60,
    cpu=4.0,
    memory=16384,
    max_containers=1,
)
def validate_affordance_artifacts() -> dict:
    """Load the staged wheels on an L40S and execute one CUDA kernel from each."""
    import sys

    output = OUT_ROOT / TAG
    manifest_path = output / f"{TAG}.manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing staged manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    wheel_paths = []
    for item in manifest["wheels"]:
        wheel_path = output / item["file"]
        if not wheel_path.exists():
            raise RuntimeError(f"missing staged wheel: {wheel_path}")
        actual_sha256 = sha256(wheel_path)
        if actual_sha256 != item["sha256"]:
            raise RuntimeError(
                f"staged wheel hash mismatch for {wheel_path.name}: "
                f"{actual_sha256} != {item['sha256']}"
            )
        wheel_paths.append(str(wheel_path))

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", *wheel_paths],
        check=True,
    )

    # Keep torch imports inside the remote subprocess. Modal serializes this function
    # from the local development environment, where torch is intentionally not installed.
    smoke_script = r'''
import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on validation worker")
device_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if capability != (8, 9):
    raise RuntimeError(f"expected SM89 L40S, got {device_name} capability={capability}")

from torch_scatter import scatter
src = torch.tensor([1.0, 2.0, 3.0], device="cuda")
index = torch.tensor([0, 1, 0], device="cuda")
scatter_out = scatter(src, index, dim=0, reduce="sum")
torch.cuda.synchronize()
if scatter_out.tolist() != [4.0, 2.0]:
    raise RuntimeError(f"torch-scatter CUDA smoke mismatch: {scatter_out.tolist()}")

import pointnet2_ops._ext as pointnet2_ext
xyz = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
    device="cuda",
    dtype=torch.float32,
).contiguous()
fps = pointnet2_ext.furthest_point_sampling(xyz, 2)
torch.cuda.synchronize()
if tuple(fps.shape) != (1, 2):
    raise RuntimeError(f"pointnet2 CUDA smoke returned unexpected shape: {tuple(fps.shape)}")

import chamfer_3D
xyz1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device="cuda")
xyz2 = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device="cuda")
dist1 = torch.zeros((1, 2), device="cuda")
dist2 = torch.zeros((1, 2), device="cuda")
idx1 = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
idx2 = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
torch.cuda.synchronize()
if dist1.tolist() != [[0.0, 1.0]] or dist2.tolist() != [[0.0, 1.0]]:
    raise RuntimeError(
        f"chamfer_3D CUDA smoke mismatch: dist1={dist1.tolist()} dist2={dist2.tolist()}"
    )

report = {
    "gpu": device_name,
    "compute_capability": f"{capability[0]}.{capability[1]}",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_scatter": scatter_out.tolist(),
    "pointnet2_fps_shape": list(fps.shape),
    "chamfer_dist1": dist1.tolist(),
    "chamfer_dist2": dist2.tolist(),
    "result": "AFFORDANCE_L40S_SMOKE_OK",
}
print("AFFORDANCE_SMOKE_JSON=" + json.dumps(report, sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", smoke_script],
        check=True,
        capture_output=True,
        text=True,
    )
    print(proc.stdout, end="", flush=True)
    marker = "AFFORDANCE_SMOKE_JSON="
    report_line = next((line for line in proc.stdout.splitlines() if line.startswith(marker)), None)
    if report_line is None:
        raise RuntimeError(f"validation report marker missing from output: {proc.stdout}")
    report = json.loads(report_line[len(marker) :])
    report["tag"] = TAG
    print(json.dumps(report, indent=2), flush=True)
    return report
