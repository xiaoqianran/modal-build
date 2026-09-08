from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import modal

TAG = "pixal3d-py310-cu124-torch260-sm90-fa3-v1"
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

# This is the FA3 wheel pinned by Pixal3D's own requirements-hfdemo.txt for
# the same torch==2.6.0 / CUDA 12.4 environment. The digest comes from the
# GitHub Release asset metadata for JeffreyXiang/Storages:Space_Wheels_251210.
FLASH_ATTN_3_URL = (
    "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
    "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
)
FLASH_ATTN_3_SHA256 = "fa9b08af0b264bb297a8a9a1f3c84417e21c2b9c81c1002a2bf980f511ff524a"

app = modal.App("modal-build-pixal3d-h100-sm90")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "gh", "build-essential", "libeigen3-dev", "clang", "cmake", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade uv",
        "uv pip install --system wheel setuptools packaging ninja",
        "uv pip install --system torch==2.6.0 torchvision==0.21.0 triton==3.2.0 --index-url https://download.pytorch.org/whl/cu124",
    )
)


def sh(cmd: str, cwd: str | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


def clone(repo: str, dst: str, commit: str, recursive: bool = False) -> None:
    flag = "--recursive " if recursive else ""
    sh(f"git clone {flag}{repo} {dst} && git -C {dst} checkout {commit}")
    if recursive:
        sh(f"git -C {dst} submodule update --init --recursive")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("modal-build-github")],
    timeout=60 * 60,
    max_containers=1,
)
def build_and_release() -> dict:
    WHEELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = (
        "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda "
        "TORCH_CUDA_ARCH_LIST=9.0 NATTEN_CUDA_ARCH=9.0 "
        "NATTEN_N_WORKERS=4 MAX_JOBS=4 FORCE_CUDA=1 "
        "CPLUS_INCLUDE_PATH=/usr/include/eigen3"
    )

    clone("https://github.com/NVlabs/nvdiffrast.git", "/tmp/nvdiffrast", COMMITS["nvdiffrast"])
    clone("https://github.com/JeffreyXiang/nvdiffrec.git", "/tmp/nvdiffrec", COMMITS["nvdiffrec"])
    clone("https://github.com/JeffreyXiang/FlexGEMM.git", "/tmp/FlexGEMM", COMMITS["flex_gemm"], True)
    clone("https://github.com/JeffreyXiang/CuMesh.git", "/tmp/CuMesh", COMMITS["cumesh"], True)
    clone("https://github.com/microsoft/TRELLIS.2.git", "/tmp/TRELLIS.2", COMMITS["trellis2"], True)

    for src in (
        "/tmp/nvdiffrast",
        "/tmp/nvdiffrec",
        "/tmp/FlexGEMM",
        "/tmp/CuMesh",
        "/tmp/TRELLIS.2/o-voxel",
    ):
        sh(f"{env} python -m pip wheel '{src}' --no-build-isolation --no-deps -w '{WHEELS}'")
    sh(f"{env} python -m pip wheel natten==0.21.0 --no-build-isolation --no-deps -w '{WHEELS}'")

    fa3 = WHEELS / "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
    urllib.request.urlretrieve(FLASH_ATTN_3_URL, fa3)
    actual_fa3_sha = sha256(fa3)
    if actual_fa3_sha != FLASH_ATTN_3_SHA256:
        raise RuntimeError(
            f"FlashAttention-3 wheel sha256 mismatch: {actual_fa3_sha} != "
            f"{FLASH_ATTN_3_SHA256}"
        )

    wheels = []
    for path in sorted(WHEELS.glob("*.whl")):
        wheels.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if len(wheels) != 7:
        raise RuntimeError(f"expected 7 wheels, got {len(wheels)}: {[x['file'] for x in wheels]}")

    # Import smoke tests execute on the same H100 used for compilation.
    sh(f"uv pip install --system --no-deps '{WHEELS}'/*.whl")
    sh(
        "python - <<'PY'\n"
        "import torch\n"
        "import flash_attn_interface as fa3\n"
        "import flex_gemm, natten, o_voxel\n"
        "assert torch.cuda.get_device_capability() == (9, 0), torch.cuda.get_device_capability()\n"
        "q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)\n"
        "k = torch.randn_like(q)\n"
        "v = torch.randn_like(q)\n"
        "out = fa3.flash_attn_func(q, k, v)\n"
        "torch.cuda.synchronize()\n"
        "assert out.shape == q.shape\n"
        "print('FA3_SMOKE_OK', out.shape)\n"
        "PY"
    )

    archive = Path(shutil.make_archive(str(OUT / f"{TAG}.wheels"), "zip", WHEELS))
    archive_sha = sha256(archive)
    manifest = {
        "tag": TAG,
        "python": "3.10",
        "cuda": "12.4.1",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "triton": "3.2.0",
        "cuda_arch": "9.0",
        "target_gpu": "H100",
        "attention_backend": "flash_attn_3",
        "flash_attn_3": {
            "source": FLASH_ATTN_3_URL,
            "sha256": FLASH_ATTN_3_SHA256,
        },
        "commits": COMMITS,
        "wheels": wheels,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "notes": "Compatibility-first H100 artifact. Benchmark CUDA 12.8 separately after correctness is established.",
    }
    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sha_path = OUT / f"{TAG}.wheels.zip.sha256"
    sha_path.write_text(f"{archive_sha}  {TAG}.wheels.zip\n")

    exists = (
        subprocess.run(
            ["gh", "release", "view", TAG, "--repo", REPO],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if not exists:
        sh(
            f"gh release create '{TAG}' --repo '{REPO}' --title '{TAG}' "
            "--notes 'Pixal3D native H100/SM90 CUDA wheels plus the upstream-pinned FlashAttention-3 wheel.'"
        )
    sh(
        f"gh release upload '{TAG}' --repo '{REPO}' --clobber "
        f"'{archive}' '{manifest_path}' '{sha_path}'"
    )
    return manifest
