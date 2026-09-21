from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import modal

TAG = "fastsam3d-native-py311-cu121-torch251-sm89-v3"
BASE_TAG = "fastsam3d-native-py311-cu121-torch251-sm89-v2"
BASE_SHA256 = "b85717ca208078f1802ec4f85113f3cedbd405d5512f25f27f1af007a8ba0715"
BASE_URL = f"https://github.com/xiaoqianran/modal-build/releases/download/{BASE_TAG}/{BASE_TAG}.wheels.zip"
NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"
WHEELS = Path("/tmp/wheels")
OUT = Path("/out")
build_artifacts = modal.Volume.from_name("modal-build-artifacts", create_if_missing=True)

app = modal.App("modal-build-fastsam3d-native-v3")
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade uv setuptools wheel packaging ninja",
        "uv pip install --system torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121",
        "uv pip install --system fvcore==0.1.5.post20221221 iopath==0.1.10",
    )
)


def sh(cmd: str, cwd: str | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


def _wheel_manifest() -> list[dict]:
    return [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(WHEELS.glob("*.whl"))
    ]


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/out": build_artifacts},
    timeout=60 * 60,
    max_containers=1,
)
def build() -> dict:
    shutil.rmtree(WHEELS, ignore_errors=True)
    shutil.rmtree("/tmp/nvdiffrast", ignore_errors=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    bundle = Path("/tmp/base.wheels.zip")
    urllib.request.urlretrieve(BASE_URL, bundle)
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if bundle_sha != BASE_SHA256:
        raise RuntimeError(f"base bundle SHA256 mismatch: {bundle_sha}")
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(WHEELS)

    sh("git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast")
    sh(f"git checkout {NVDIFFRAST_COMMIT}", "/tmp/nvdiffrast")
    env = "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 FORCE_CUDA=1"
    sh(
        f"{env} python -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        "/tmp/nvdiffrast",
    )

    wheels = _wheel_manifest()
    names = [item["file"].lower() for item in wheels]
    expected = ("pytorch3d-", "gsplat-", "nvdiffrast-")
    if len(wheels) != 3 or any(not any(name.startswith(prefix) for name in names) for prefix in expected):
        raise RuntimeError(f"expected PyTorch3D + gsplat + nvdiffrast wheels, got {[x['file'] for x in wheels]}")

    sh(f"python -m pip install --no-deps {WHEELS}/*.whl")
    sh(
        "python - <<'PY'\n"
        "import torch\n"
        "import nvdiffrast.torch as dr\n"
        "import pytorch3d, gsplat\n"
        "assert torch.cuda.get_device_capability() == (8, 9)\n"
        "ctx = dr.RasterizeCudaContext(device=torch.device('cuda'))\n"
        "pos = torch.tensor([[[-0.5,-0.5,0.0,1.0],[0.5,-0.5,0.0,1.0],[0.0,0.5,0.0,1.0]]], device='cuda', dtype=torch.float32)\n"
        "tri = torch.tensor([[0,1,2]], device='cuda', dtype=torch.int32)\n"
        "rast, _ = dr.rasterize(ctx, pos, tri, resolution=[16,16])\n"
        "assert rast.shape == (1,16,16,4)\n"
        "assert float(rast[...,3].max()) > 0\n"
        "print(torch.__version__, pytorch3d.__file__, gsplat.__file__, dr.__file__, float(rast[...,3].max()))\n"
        "PY"
    )

    archive = Path(shutil.make_archive(str(OUT / f"{TAG}.wheels"), "zip", WHEELS))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "tag": TAG,
        "base_tag": BASE_TAG,
        "base_sha256": BASE_SHA256,
        "python": "3.11",
        "cuda": "12.1.1",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "sources": {"nvdiffrast": {"repository": "NVlabs/nvdiffrast", "revision": NVDIFFRAST_COMMIT}},
        "wheels": wheels,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
    }
    (OUT / f"{TAG}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / f"{TAG}.wheels.zip.sha256").write_text(f"{archive_sha}  {archive.name}\n")
    build_artifacts.commit()
    return manifest
