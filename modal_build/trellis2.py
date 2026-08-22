from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import modal

TAG = "trellis2-py311-cu124-torch260-sm89-v1"
REPO = "xiaoqianran/modal-build"
WHEELHOUSE = Path("/tmp/wheelhouse")
OUT = Path("/tmp/out")

app = modal.App("modal-build-trellis2")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "gh", "build-essential", "libjpeg-dev", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja uv",
        "uv pip install --system torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
    )
)


def sh(cmd: str, cwd: str | None = None):
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


@app.function(
    image=image,
    gpu="L40S",
    secrets=[modal.Secret.from_name("modal-build-github")],
    timeout=60 * 60,
    max_containers=1,
)
def build_and_release() -> dict:
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4"

    # Build distributable wheels once. Runtime images install these instead of recompiling CUDA.
    sh(f"{env} python -m pip wheel flash-attn==2.7.3 --no-build-isolation --no-deps -w {WHEELHOUSE}")

    sh("git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh")
    sh(f"{env} python -m pip wheel . --no-build-isolation --no-deps -w {WHEELHOUSE}", "/tmp/CuMesh")

    sh("git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM")
    sh(f"{env} python -m pip wheel . --no-build-isolation --no-deps -w {WHEELHOUSE}", "/tmp/FlexGEMM")

    sh("git clone https://github.com/Archerkattri/hermit-trellis2-plus-plus.git /tmp/hermit")
    sh("git checkout 2c8402a92ea97c510c09e278fae557771aad774d", "/tmp/hermit")
    sh("git submodule update --init --recursive", "/tmp/hermit")
    sh(f"{env} python -m pip wheel ./o-voxel --no-build-isolation --no-deps -w {WHEELHOUSE}", "/tmp/hermit")

    manifest = {
        "tag": TAG,
        "python": "3.11",
        "cuda": "12.4.1",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "wheels": [],
    }
    for p in sorted(WHEELHOUSE.glob("*.whl")):
        manifest["wheels"].append(
            {"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        )

    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive_base = OUT / f"{TAG}.wheels"
    archive = Path(shutil.make_archive(str(archive_base), "zip", WHEELHOUSE))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    sha_path = OUT / f"{archive.name}.sha256"
    sha_path.write_text(f"{sha}  {archive.name}\n")

    # Idempotent release: create once, otherwise replace assets.
    exists = subprocess.run(["gh", "release", "view", TAG, "--repo", REPO], capture_output=True, check=False).returncode == 0
    if not exists:
        sh(f"gh release create {TAG} --repo {REPO} --title {TAG} --notes 'Prebuilt TRELLIS2 CUDA wheels for L40S / SM89.'")
    sh(
        f"gh release upload {TAG} --repo {REPO} --clobber "
        f"{archive} {manifest_path} {sha_path}"
    )
    return {"tag": TAG, "archive": archive.name, "sha256": sha, "wheels": manifest["wheels"]}
