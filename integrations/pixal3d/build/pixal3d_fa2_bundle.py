from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import modal

BASE_TAG = "pixal3d-py310-cu124-torch260-sm89-v1"
TAG = "pixal3d-py310-cu124-torch260-sm89-fa2-v1"
REPO = "xiaoqianran/modal-build"
OUT = Path("/tmp/out")
WHEELS = Path("/tmp/wheels")

FLASH_ATTN_VERSION = "2.8.3"
FLASH_ATTN_FILE = "flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_SHA256 = "1c8d2fb083edea3647688fcd4bb21d4641eb01f50359d8fa6200ecd093ba1c54"
FLASH_ATTN_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    + FLASH_ATTN_FILE
)
BASE_URL = f"https://github.com/{REPO}/releases/download/{BASE_TAG}/{BASE_TAG}.wheels.zip"

app = modal.App("modal-build-pixal3d-fa2-bundle")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("gh", "unzip")
    .run_commands("python -m pip install --upgrade uv")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def sh(cmd: str) -> None:
    subprocess.run(["bash", "-lc", cmd], check=True)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal-build-github")],
    timeout=15 * 60,
    max_containers=1,
)
def build_and_release() -> dict:
    """Augment the already-built SM89 bundle with the official FA2 wheel.

    This intentionally does not rebuild nvdiffrast/FlexGEMM/CuMesh/O-Voxel/
    NATTEN. The expensive CUDA compilation remains owned by BASE_TAG; this
    function only verifies and repackages immutable artifacts.
    """

    OUT.mkdir(parents=True, exist_ok=True)
    WHEELS.mkdir(parents=True, exist_ok=True)

    base_archive = Path("/tmp/base.wheels.zip")
    download(BASE_URL, base_archive)
    sh(f"unzip -q '{base_archive}' -d '{WHEELS}'")

    flash_wheel = WHEELS / FLASH_ATTN_FILE
    download(FLASH_ATTN_URL, flash_wheel)
    actual_flash_sha = sha256(flash_wheel)
    if actual_flash_sha != FLASH_ATTN_SHA256:
        raise RuntimeError(
            f"FlashAttention wheel sha256 mismatch: {actual_flash_sha} != {FLASH_ATTN_SHA256}"
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
        raise RuntimeError(f"expected base 6 wheels + FlashAttention, got {len(wheels)}")

    archive = Path(shutil.make_archive(str(OUT / f"{TAG}.wheels"), "zip", WHEELS))
    archive_sha = sha256(archive)
    manifest = {
        "tag": TAG,
        "derived_from": BASE_TAG,
        "python": "3.10",
        "cuda": "12.4.1 runtime / cu12 FlashAttention wheel",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "triton": "3.2.0",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "attention_backend": "flash_attn",
        "flash_attn": {
            "version": FLASH_ATTN_VERSION,
            "source": FLASH_ATTN_URL,
            "sha256": FLASH_ATTN_SHA256,
            "cxx11abi": False,
        },
        "wheels": wheels,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
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
            "--notes 'Pixal3D L40S/SM89 bundle derived from the existing CUDA wheels plus official FlashAttention 2.8.3.'"
        )
    sh(
        f"gh release upload '{TAG}' --repo '{REPO}' --clobber "
        f"'{archive}' '{manifest_path}' '{sha_path}'"
    )
    return manifest
