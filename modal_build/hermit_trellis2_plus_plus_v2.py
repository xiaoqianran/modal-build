from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import modal

TAG = "hermit-trellis2-plus-plus-py311-cu124-torch260-sm89-v2"
FORK = "Archerkattri/hermit-trellis2-plus-plus"
FORK_COMMIT = "2c8402a92ea97c510c09e278fae557771aad774d"
WHEELS = Path("/tmp/wheels")
OUT = Path("/out")
build_artifacts = modal.Volume.from_name("modal-build-artifacts", create_if_missing=True)

COMMITS = {
    "cumesh": "12289e1062f0603f2f0d0771b02e1395d247f26f",
    "flex_gemm": "6dd94a859c26ee8246888502eada3dd8ad85532e",
    "nvdiffrast": "253ac4fcea7de5f396371124af597e6cc957bfae",
    "nvdiffrec": "b296927cc7fd01c2ac1087c8065c4d7248f72da4",
    "eigen": "e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b",
}

app = modal.App("modal-build-hermit-trellis2-plus-plus-v2")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "libjpeg-dev", "libgl1", "libglib2.0-0", "libeigen3-dev")
    .run_commands(
        "python -m pip install --upgrade uv setuptools wheel packaging ninja",
        "uv pip install --system torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
    )
)


def sh(cmd: str, cwd: str | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


def clone(repo: str, dst: str, commit: str, *, recursive: bool = False) -> None:
    flag = "--recursive " if recursive else ""
    sh(f"git clone {flag}{repo} {dst} && git -C {dst} checkout {commit}")
    if recursive:
        sh(f"git -C {dst} submodule update --init --recursive")


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
        "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 "
        "MAX_JOBS=4 FORCE_CUDA=1 CPLUS_INCLUDE_PATH=/usr/include/eigen3"
    )

    sh(f"{env} python -m pip wheel flash-attn==2.7.3 --no-build-isolation --no-deps -w {WHEELS}")

    clone("https://github.com/NVlabs/nvdiffrast.git", "/tmp/nvdiffrast", COMMITS["nvdiffrast"])
    clone("https://github.com/JeffreyXiang/nvdiffrec.git", "/tmp/nvdiffrec", COMMITS["nvdiffrec"])
    clone("https://github.com/JeffreyXiang/CuMesh.git", "/tmp/CuMesh", COMMITS["cumesh"], recursive=True)
    clone("https://github.com/JeffreyXiang/FlexGEMM.git", "/tmp/FlexGEMM", COMMITS["flex_gemm"], recursive=True)

    for src in ("/tmp/nvdiffrast", "/tmp/nvdiffrec", "/tmp/CuMesh", "/tmp/FlexGEMM"):
        sh(f"{env} python -m pip wheel '{src}' --no-build-isolation --no-deps -w '{WHEELS}'")

    clone(f"https://github.com/{FORK}.git", "/tmp/hermit", FORK_COMMIT)
    sh("git submodule update --init --recursive", "/tmp/hermit")
    sh("mkdir -p o-voxel/third_party && git clone https://gitlab.com/libeigen/eigen.git o-voxel/third_party/eigen", "/tmp/hermit")
    sh(f"git -C o-voxel/third_party/eigen checkout {COMMITS['eigen']}", "/tmp/hermit")
    sh(f"{env} python -m pip wheel ./o-voxel --no-build-isolation --no-deps -w {WHEELS}", "/tmp/hermit")

    wheels = []
    for path in sorted(WHEELS.glob("*.whl")):
        wheels.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if len(wheels) != 6:
        raise RuntimeError(f"expected 6 wheels, got {[x['file'] for x in wheels]}")

    archive = Path(shutil.make_archive(str(OUT / f"{TAG}.wheels"), "zip", WHEELS))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "tag": TAG,
        "python": "3.11",
        "cuda": "12.4.1",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "source": FORK,
        "source_revision": FORK_COMMIT,
        "commits": COMMITS,
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
