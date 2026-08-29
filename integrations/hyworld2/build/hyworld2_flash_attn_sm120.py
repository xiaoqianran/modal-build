from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

ARTIFACT_VOLUME = "modal-build-artifacts"


def sh(cmd: str, *, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=env, check=True)


def clone(repo: str, dst: str | Path, revision: str, *, recursive: bool = False) -> None:
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    recurse = "--recursive " if recursive else ""
    sh(f"git clone {recurse}--filter=blob:none {repo} {dst}")
    sh(f"git checkout --detach {revision}", cwd=dst)
    if recursive:
        sh("git submodule update --init --recursive", cwd=dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wheel_records(wheel_dir: Path, owners: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(wheel_dir.glob("*.whl")):
        owner = next(
            (value for prefix, value in owners.items() if path.name.startswith(prefix)), None
        )
        records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "artifact": owner,
            }
        )
    return records


def package_bundle(
    *, tag: str, wheel_dir: Path, out_dir: Path, manifest: dict[str, Any], license_dir: Path
) -> dict[str, Any]:
    staging = Path("/tmp") / f"{tag}-bundle"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "wheels").mkdir(parents=True)
    for wheel in wheel_dir.glob("*.whl"):
        shutil.copy2(wheel, staging / "wheels" / wheel.name)
    if license_dir.exists():
        shutil.copytree(license_dir, staging / "LICENSES")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(shutil.make_archive(str(out_dir / f"{tag}.wheels"), "zip", staging))
    archive_sha = sha256(archive)
    manifest.update(
        {
            "archive": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
        }
    )
    (out_dir / f"{tag}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / f"{tag}.wheels.zip.sha256").write_text(f"{archive_sha}  {archive.name}\n")
    return manifest


TAG = "hyworld2-flash-attn-py311-cu128-torch271-sm120-v1"
FLASH_ATTN_REVISION = "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820"
PYTHON, CUDA, TORCH = "3.11", "12.8.1", "2.7.1"
CUDA_ARCH, GPU = "12.0", "RTX-PRO-6000"
WHEELS, LICENSES, OUT = Path("/tmp/wheels"), Path("/tmp/licenses"), Path("/out")

app = modal.App("modal-build-hyworld2-flash-attn-sm120")
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu22.04", add_python=PYTHON)
    .apt_install("git", "build-essential", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja",
        f"python -m pip install torch=={TORCH} --index-url https://download.pytorch.org/whl/cu128",
    )
)


@app.function(
    image=image, gpu=GPU, volumes={"/out": artifacts}, timeout=2 * 60 * 60, max_containers=1
)
def build() -> dict:
    import torch

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("FlashAttention Blackwell build must run on sm_120")
    shutil.rmtree(WHEELS, ignore_errors=True)
    WHEELS.mkdir(parents=True)
    shutil.rmtree(LICENSES, ignore_errors=True)
    LICENSES.mkdir(parents=True)

    src = Path("/tmp/flash-attention")
    clone(
        "https://github.com/Dao-AILab/flash-attention.git", src, FLASH_ATTN_REVISION, recursive=True
    )
    shutil.copy2(src / "LICENSE", LICENSES / "flash-attention-LICENSE.txt")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
            "FLASH_ATTN_CUDA_ARCHS": "120",
            "MAX_JOBS": "4",
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
        }
    )
    sh(
        f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        cwd=src,
        env=env,
    )
    wheels = sorted(WHEELS.glob("flash_attn-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one flash_attn wheel, got {[p.name for p in wheels]}")
    sh(f"{sys.executable} -m pip install --force-reinstall --no-deps {wheels[0]}")

    from flash_attn import flash_attn_func

    q = torch.randn((1, 32, 4, 64), device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, q, q, causal=False)
    if out.shape != q.shape or not torch.isfinite(out).all():
        raise RuntimeError("flash-attn sm_120 CUDA smoke failed")

    manifest = {
        "tag": TAG,
        "bundle_kind": "hyworld2-optional-flash-attention",
        "public_release": True,
        "experimental": True,
        "python": PYTHON,
        "cuda": CUDA,
        "torch": TORCH,
        "cuda_arch": CUDA_ARCH,
        "target_gpu": GPU,
        "source": "Dao-AILab/flash-attention",
        "source_revision": FLASH_ATTN_REVISION,
        "license": "BSD-3-Clause",
        "wheels": wheel_records(WHEELS, {"flash_attn-": "flash-attention"}),
        "smoke": ["gpu-sm120", "flash_attn_func-bf16"],
    }
    result = package_bundle(
        tag=TAG, wheel_dir=WHEELS, out_dir=OUT, manifest=manifest, license_dir=LICENSES
    )
    artifacts.commit()
    return result
