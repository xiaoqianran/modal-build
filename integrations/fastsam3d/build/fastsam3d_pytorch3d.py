from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import modal

TAG = "fastsam3d-pytorch3d-py311-cu121-torch251-sm89-v1"
PYTORCH3D_REPO = "facebookresearch/pytorch3d"
PYTORCH3D_COMMIT = "75ebeeaea0908c5527e7b1e305fbc7681382db47"
WHEELS = Path("/tmp/wheels")
OUT = Path("/out")
build_artifacts = modal.Volume.from_name("modal-build-artifacts", create_if_missing=True)

app = modal.App("modal-build-fastsam3d-pytorch3d")
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


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/out": build_artifacts},
    timeout=60 * 60,
    max_containers=1,
)
def build() -> dict:
    WHEELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = (
        "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda "
        "TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 FORCE_CUDA=1"
    )
    sh(f"git clone https://github.com/{PYTORCH3D_REPO}.git /tmp/pytorch3d")
    sh(f"git checkout {PYTORCH3D_COMMIT}", "/tmp/pytorch3d")
    sh(
        f"{env} python -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        "/tmp/pytorch3d",
    )

    wheels = []
    for path in sorted(WHEELS.glob("*.whl")):
        wheels.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if len(wheels) != 1:
        raise RuntimeError(f"expected one pytorch3d wheel, got {[x['file'] for x in wheels]}")

    sh(f"uv pip install --system --no-deps {WHEELS}/*.whl")
    sh(
        "python -c \"import torch, pytorch3d; "
        "from pytorch3d.renderer import MeshRasterizer; "
        "assert torch.cuda.get_device_capability() == (8, 9); "
        "print(torch.__version__, pytorch3d.__file__, MeshRasterizer)\""
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
        "source": PYTORCH3D_REPO,
        "source_revision": PYTORCH3D_COMMIT,
        "wheels": wheels,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
    }
    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sha_path = OUT / f"{TAG}.wheels.zip.sha256"
    sha_path.write_text(f"{archive_sha}  {archive.name}\n")
    build_artifacts.commit()
    return manifest
