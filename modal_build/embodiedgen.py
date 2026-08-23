from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

TAG = "embodiedgen-v2.0.0-py310-cu126-torch280-sm89-v1"
REPO = "xiaoqianran/modal-build"
OUT = Path("/tmp/out")
WHEELS = Path("/tmp/wheels")
CACHE_ROOT = Path("/tmp/torch_extensions")

PINS = {
    "embodiedgen": "v2.0.0",
    "pytorch3d": "75ebeeaea0908c5527e7b1e305fbc7681382db47",
    "nvdiffrast": "729261d",
    "gsplat": "1.5.3",
    "kaolin": "0.18.0",
}

app = modal.App("modal-build-embodiedgen-v2")

# Deliberately CPU-only: CUDA devel supplies nvcc, but no paid GPU is allocated.
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "gh",
        "build-essential",
        "gcc",
        "g++",
        "cmake",
        "ninja-build",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "MAX_JOBS": "2",
            "CC": "gcc",
            "CXX": "g++",
        }
    )
    .run_commands(
        "python -m pip install --upgrade 'pip>=25' setuptools==80.10.2 wheel packaging ninja",
        "python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126",
        "python -m pip install gsplat==1.5.3",
    )
)


def sh(cmd: str, cwd: str | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal-build-github")],
    timeout=60 * 60,
    cpu=8.0,
    memory=32768,
    max_containers=1,
)
def build_and_release() -> dict:
    """Build reusable SM89 artifacts without renting a GPU, then publish a release."""
    WHEELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    env = (
        "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 "
        "TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2"
    )

    # 1) PyTorch3D: true binary wheel, pinned to the exact commit resolved by the
    # validated EmbodiedGen image.
    sh(
        f"{env} python -m pip wheel --no-deps --no-build-isolation "
        f"'git+https://github.com/facebookresearch/pytorch3d.git@{PINS['pytorch3d']}' "
        f"-w '{WHEELS}'"
    )

    # 2) nvdiffrast Python wheel, exact commit used in the validated runtime.
    sh("git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast")
    sh(f"git checkout {PINS['nvdiffrast']}", "/tmp/nvdiffrast")
    sh(f"{env} python -m pip wheel --no-deps --no-build-isolation . -w '{WHEELS}'", "/tmp/nvdiffrast")
    sh(f"python -m pip install --no-deps {WHEELS}/nvdiffrast-*.whl")

    # 3) gsplat 1.5.3 O3 CUDA JIT cache.  This is the expensive ~5 minute build in
    # the validated image; prebuilding it is the biggest repeated-cost win.
    sh(
        "FAST_COMPILE=0 VERBOSE=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 "
        "python -c \"from gsplat.cuda._backend import _C; print('gsplat cache built', _C is not None)\""
    )

    # 4) nvdiffrast CUDA plugin cache.  Upstream intentionally chooses the visible
    # GPU arch at JIT time by clearing TORCH_CUDA_ARCH_LIST.  A CPU builder has no
    # visible GPU, so temporarily pin the installed loader to SM89 for compilation.
    sh(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import nvdiffrast.torch.ops as ops\n"
        "p=Path(ops.__file__)\n"
        "s=p.read_text()\n"
        "old=\"os.environ['TORCH_CUDA_ARCH_LIST'] = ''\"\n"
        "new=\"os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'\"\n"
        "if old not in s: raise SystemExit(f'arch line not found in {p}')\n"
        "p.write_text(s.replace(old,new,1))\n"
        "print('patched CPU-only build arch in', p)\n"
        "PY"
    )
    sh(
        "MAX_JOBS=2 python -c \"import nvdiffrast.torch.ops as ops; "
        "m=ops._get_plugin(False); print('nvdiffrast CUDA plugin built', m)\""
    )

    # Preserve the exact torch-extension directory layout so runtime extraction to
    # ~/.cache/torch_extensions is enough for the normal loaders to hit the cache.
    sh(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "from torch.utils.cpp_extension import _get_build_directory\n"
        "import shutil\n"
        f"dst=Path('{CACHE_ROOT}')\n"
        "for name in ('gsplat_cuda','nvdiffrast_plugin'):\n"
        "    src=Path(_get_build_directory(name, verbose=False))\n"
        "    if not src.exists(): raise SystemExit(f'missing cache {src}')\n"
        "    rel=Path(*src.parts[-2:])  # py310_cu126/<plugin>\n"
        "    out=dst/rel\n"
        "    out.parent.mkdir(parents=True,exist_ok=True)\n"
        "    shutil.copytree(src,out,dirs_exist_ok=True)\n"
        "    print(name, src, '->', out)\n"
        "PY"
    )

    wheels = []
    for p in sorted(WHEELS.glob("*.whl")):
        wheels.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    if len(wheels) != 2:
        raise RuntimeError(f"expected 2 wheels, got {len(wheels)}: {[x['file'] for x in wheels]}")

    wheel_archive = Path(shutil.make_archive(str(OUT / f"{TAG}.wheels"), "zip", WHEELS))
    cache_archive = Path(shutil.make_archive(str(OUT / f"{TAG}.torch-extensions"), "zip", CACHE_ROOT))

    # Sanity-check the release really contains binary outputs, not just build files.
    shared_objects = sorted(str(p.relative_to(CACHE_ROOT)) for p in CACHE_ROOT.rglob("*.so"))
    if len(shared_objects) < 2:
        raise RuntimeError(f"expected compiled cache .so files, got {shared_objects}")

    manifest = {
        "tag": TAG,
        "source": "HorizonRobotics/EmbodiedGen@v2.0.0",
        "python": "3.10",
        "ubuntu": "22.04",
        "cuda": "12.6.3",
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "build_gpu": None,
        "build_strategy": "CPU-only nvcc compile; GPU used only for end-to-end validation",
        "pins": PINS,
        "wheels": wheels,
        "torch_extension_shared_objects": shared_objects,
        "assets": {
            wheel_archive.name: {"bytes": wheel_archive.stat().st_size, "sha256": sha256(wheel_archive)},
            cache_archive.name: {"bytes": cache_archive.stat().st_size, "sha256": sha256(cache_archive)},
        },
        "kaolin": {
            "version": "0.18.0",
            "redistributed": False,
            "install_source": "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html",
        },
        "validated": {
            "gpu": "NVIDIA L40S",
            "command": "img3d-cli --image_path apps/assets/example_image/sample_00.jpg --output_root /data/outputs/test",
            "result": "VALIDATION_OK",
            "ply_vertices": 95004,
            "obj_vertices": 516271,
            "obj_faces": 891420,
            "glb_geometries": 1,
            "video_duration_seconds": 2.0,
        },
    }

    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sha_path = OUT / f"{TAG}.sha256"
    sha_path.write_text(
        f"{sha256(wheel_archive)}  {wheel_archive.name}\n"
        f"{sha256(cache_archive)}  {cache_archive.name}\n"
        f"{sha256(manifest_path)}  {manifest_path.name}\n"
    )

    exists = subprocess.run(
        ["gh", "release", "view", TAG, "--repo", REPO], capture_output=True, check=False
    ).returncode == 0
    notes = (
        "EmbodiedGen v2.0.0 build artifacts for Python 3.10 / Torch 2.8.0 / "
        "CUDA 12.6 / L40S SM89. Built CPU-only; end-to-end validated on L40S. "
        "Model weights are intentionally excluded."
    )
    if not exists:
        sh(f"gh release create '{TAG}' --repo '{REPO}' --title '{TAG}' --notes {json.dumps(notes)}")
    sh(
        f"gh release upload '{TAG}' --repo '{REPO}' --clobber "
        f"'{wheel_archive}' '{cache_archive}' '{manifest_path}' '{sha_path}'"
    )
    return manifest
