from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import modal

TAG = "pixal3d-py310-cu124-torch260-sm89-v1"
REPO = "xiaoqianran/modal-build"
OUT = Path("/tmp/out")
WHEELS = Path("/tmp/wheels")

COMMITS = {
    "pixal3d": "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af",
    "trellis2": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
    "flex_gemm": "6dd94a859c26ee8246888502eada3dd8ad85532e",
    "cumesh": "12289e1062f0603f2f0d0771b02e1395d247f26f",
    "nvdiffrast": "253ac4fcea7de5f396371124af597e6cc957bfae",
    "nvdiffrec": "b296927cc7fd01c2ac1087c8065c4d7248f72da4",
}

app = modal.App("modal-build-pixal3d")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "gh", "build-essential", "libeigen3-dev", "clang", "cmake", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade uv",
        "uv pip install --system wheel setuptools packaging ninja",
        "uv pip install --system torch==2.6.0 torchvision==0.21.0 triton==3.2.0 --index-url https://download.pytorch.org/whl/cu124",
    )
)


def sh(cmd: str, cwd: str | None = None):
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


def clone(repo: str, dst: str, commit: str, recursive: bool = False):
    flag = "--recursive " if recursive else ""
    sh(f"git clone {flag}{repo} {dst} && git -C {dst} checkout {commit}")
    if recursive:
        sh(f"git -C {dst} submodule update --init --recursive")


@app.function(
    image=image,
    gpu="L40S",
    secrets=[modal.Secret.from_name("modal-build-github")],
    timeout=60 * 60,
    max_containers=1,
)
def build_and_release() -> dict:
    WHEELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 NATTEN_CUDA_ARCH=8.9 NATTEN_N_WORKERS=4 MAX_JOBS=4 FORCE_CUDA=1 CPLUS_INCLUDE_PATH=/usr/include/eigen3"

    clone("https://github.com/NVlabs/nvdiffrast.git", "/tmp/nvdiffrast", COMMITS["nvdiffrast"])
    clone("https://github.com/JeffreyXiang/nvdiffrec.git", "/tmp/nvdiffrec", COMMITS["nvdiffrec"])
    clone("https://github.com/JeffreyXiang/FlexGEMM.git", "/tmp/FlexGEMM", COMMITS["flex_gemm"], True)
    clone("https://github.com/JeffreyXiang/CuMesh.git", "/tmp/CuMesh", COMMITS["cumesh"], True)
    clone("https://github.com/microsoft/TRELLIS.2.git", "/tmp/TRELLIS.2", COMMITS["trellis2"], True)

    for src in ("/tmp/nvdiffrast", "/tmp/nvdiffrec", "/tmp/FlexGEMM", "/tmp/CuMesh", "/tmp/TRELLIS.2/o-voxel"):
        sh(f"{env} python -m pip wheel '{src}' --no-build-isolation --no-deps -w '{WHEELS}'")
    sh(f"{env} python -m pip wheel natten==0.21.0 --no-build-isolation --no-deps -w '{WHEELS}'")

    wheels = []
    for p in sorted(WHEELS.glob("*.whl")):
        wheels.append(
            {"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        )
    if len(wheels) != 6:
        raise RuntimeError(f"expected 6 wheels, got {len(wheels)}: {[x['file'] for x in wheels]}")

    archive = Path(shutil.make_archive(str(OUT / f"{TAG}.wheels"), "zip", WHEELS))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "tag": TAG,
        "python": "3.10",
        "cuda": "12.4.1",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "triton": "3.2.0",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "commits": COMMITS,
        "wheels": wheels,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
    }
    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sha_path = OUT / f"{TAG}.wheels.zip.sha256"
    sha_path.write_text(f"{archive_sha}  {TAG}.wheels.zip\n")

    exists = subprocess.run(
        ["gh", "release", "view", TAG, "--repo", REPO], capture_output=True, check=False
    ).returncode == 0
    if not exists:
        sh(f"gh release create '{TAG}' --repo '{REPO}' --title '{TAG}' --notes 'Pixal3D CUDA wheels for L40S / SM89.'")
    sh(f"gh release upload '{TAG}' --repo '{REPO}' --clobber '{archive}' '{manifest_path}' '{sha_path}'")
    return manifest
