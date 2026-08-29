from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import modal
from _common import ARTIFACT_VOLUME, clone, package_bundle, sh, wheel_records

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
