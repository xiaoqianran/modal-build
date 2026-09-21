from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import modal

TAG = "fastsam3d-native-py311-cu121-torch251-sm89-v2"
PYTORCH3D_REPO = "facebookresearch/pytorch3d"
PYTORCH3D_COMMIT = "75ebeeaea0908c5527e7b1e305fbc7681382db47"
PYTORCH3D_BUNDLE_TAG = "fastsam3d-pytorch3d-py311-cu121-torch251-sm89-v1"
PYTORCH3D_BUNDLE_SHA256 = "bed216cec84281e5249f37e3c492031aa19d6921b46c5993a05cf0a4f86edb6a"
PYTORCH3D_BUNDLE_URL = (
    "https://github.com/xiaoqianran/modal-build/releases/download/"
    f"{PYTORCH3D_BUNDLE_TAG}/{PYTORCH3D_BUNDLE_TAG}.wheels.zip"
)
GSPLAT_REPO = "nerfstudio-project/gsplat"
GSPLAT_COMMIT = "2323de5905d5e90e035f792fe65bad0fedd413e7"
WHEELS = Path("/tmp/wheels")
OUT = Path("/out")
build_artifacts = modal.Volume.from_name("modal-build-artifacts", create_if_missing=True)

app = modal.App("modal-build-fastsam3d-native-v2")
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
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
    shutil.rmtree("/tmp/gsplat", ignore_errors=True)
    bundle_path = Path("/tmp/pytorch3d-v1.wheels.zip")
    bundle_path.unlink(missing_ok=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    env = (
        "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda "
        "TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 FORCE_CUDA=1"
    )

    urllib.request.urlretrieve(PYTORCH3D_BUNDLE_URL, bundle_path)
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if bundle_sha != PYTORCH3D_BUNDLE_SHA256:
        raise RuntimeError(f"PyTorch3D bundle SHA256 mismatch: {bundle_sha}")
    with zipfile.ZipFile(bundle_path) as archive:
        archive.extractall(WHEELS)

    sh(f"git clone --recurse-submodules https://github.com/{GSPLAT_REPO}.git /tmp/gsplat")
    sh(f"git checkout {GSPLAT_COMMIT}", "/tmp/gsplat")
    sh("git submodule update --init --recursive", "/tmp/gsplat")
    sh(
        f"{env} python -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        "/tmp/gsplat",
    )

    wheels = _wheel_manifest()
    names = [item["file"].lower() for item in wheels]
    if len(wheels) != 2 or not any(name.startswith("pytorch3d-") for name in names) or not any(
        name.startswith("gsplat-") for name in names
    ):
        raise RuntimeError(f"expected PyTorch3D + gsplat wheels, got {[x['file'] for x in wheels]}")

    sh(f"uv pip install --system --no-deps {WHEELS}/*.whl")
    sh(
        "python -c \"import torch, pytorch3d, gsplat; "
        "from pytorch3d.renderer import MeshRasterizer; "
        "from gsplat import rasterization; "
        "assert torch.cuda.get_device_capability() == (8, 9); "
        "assert callable(rasterization); "
        "print(torch.__version__, pytorch3d.__file__, gsplat.__file__, MeshRasterizer)\""
    )

    archive = Path(shutil.make_archive(str(OUT / f"{TAG}.wheels"), "zip", WHEELS))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "tag": TAG,
        "python": "3.11",
        "cuda": "12.1.1",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "sources": {
            "pytorch3d": {
                "repository": PYTORCH3D_REPO,
                "revision": PYTORCH3D_COMMIT,
                "bundle_tag": PYTORCH3D_BUNDLE_TAG,
                "bundle_sha256": PYTORCH3D_BUNDLE_SHA256,
            },
            "gsplat": {"repository": GSPLAT_REPO, "revision": GSPLAT_COMMIT},
        },
        "wheels": wheels,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
    }
    (OUT / f"{TAG}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / f"{TAG}.wheels.zip.sha256").write_text(f"{archive_sha}  {archive.name}\n")
    build_artifacts.commit()
    return manifest
