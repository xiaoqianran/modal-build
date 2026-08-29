from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import modal
from _common import ARTIFACT_VOLUME, clone, package_bundle, sh, wheel_records

TAG = "hyworld2-oss-native-py311-cu128-torch271-sm120-v1"
PYTORCH3D_REVISION = "75ebeeaea0908c5527e7b1e305fbc7681382db47"
FUSED_SSIM_REVISION = "328dc9836f513d00c4b5bc38fe30478b4435cbb5"
SPZ_REVISION = "5bf2945de1a003cee07133b1e495fe9c6ffdc7e7"
PYTHON, CUDA, TORCH, TORCHVISION = "3.11", "12.8.1", "2.7.1", "0.22.1"
CUDA_ARCH, GPU = "12.0", "RTX-PRO-6000"
WHEELS, LICENSES, OUT = Path("/tmp/wheels"), Path("/tmp/licenses"), Path("/out")

app = modal.App("modal-build-hyworld2-oss-native-sm120")
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu22.04", add_python=PYTHON)
    .apt_install("git", "build-essential", "cmake", "ninja-build", "pkg-config")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja cmake scikit-build-core pybind11",
        f"python -m pip install torch=={TORCH} torchvision=={TORCHVISION} --index-url https://download.pytorch.org/whl/cu128",
    )
)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
            "MAX_JOBS": "4",
        }
    )
    return env


def copy_license(src: Path, target: str) -> None:
    for name in ("LICENSE", "LICENSE.txt", "COPYING"):
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, LICENSES / target)
            return
    raise RuntimeError(f"license missing in {src}")


def smoke() -> None:
    import torch
    from fused_ssim import fused_ssim
    from pytorch3d.ops import knn_points

    x = torch.rand((1, 16, 3), device="cuda")
    result = knn_points(x, x, K=1)
    if result.dists.numel() != 16:
        raise RuntimeError("PyTorch3D CUDA KNN smoke failed")
    a = torch.rand((1, 3, 32, 32), device="cuda")
    score = fused_ssim(a, a)
    if float(score) < 0.99:
        raise RuntimeError(f"fused-ssim CUDA smoke failed: {float(score)}")
    import spz  # noqa: F401


@app.function(
    image=image, gpu=GPU, volumes={"/out": artifacts}, timeout=2 * 60 * 60, max_containers=1
)
def build() -> dict:
    import torch

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            f"expected sm_120, got {torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}"
        )
    shutil.rmtree(WHEELS, ignore_errors=True)
    WHEELS.mkdir(parents=True)
    shutil.rmtree(LICENSES, ignore_errors=True)
    LICENSES.mkdir(parents=True)
    env = build_env()

    p3d = Path("/tmp/pytorch3d")
    clone("https://github.com/facebookresearch/pytorch3d.git", p3d, PYTORCH3D_REVISION)
    copy_license(p3d, "PyTorch3D-LICENSE.txt")
    sh(
        f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        cwd=p3d,
        env=env,
    )

    fused = Path("/tmp/fused-ssim")
    clone("https://github.com/rahul-goel/fused-ssim.git", fused, FUSED_SSIM_REVISION)
    copy_license(fused, "fused-ssim-LICENSE.txt")
    sh(
        f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        cwd=fused,
        env=env,
    )

    spz = Path("/tmp/spz")
    clone("https://github.com/nianticlabs/spz.git", spz, SPZ_REVISION, recursive=True)
    copy_license(spz, "SPZ-LICENSE.txt")
    sh(
        f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        cwd=spz,
        env=env,
    )

    wheels = sorted(WHEELS.glob("*.whl"))
    if len(wheels) != 3:
        raise RuntimeError(f"expected 3 OSS native wheels, got {[p.name for p in wheels]}")
    sh(f"{sys.executable} -m pip install --force-reinstall --no-deps " + " ".join(map(str, wheels)))
    smoke()

    manifest = {
        "tag": TAG,
        "bundle_kind": "hyworld2-oss-native",
        "public_release": True,
        "python": PYTHON,
        "cuda": CUDA,
        "torch": TORCH,
        "torchvision": TORCHVISION,
        "cuda_arch": CUDA_ARCH,
        "target_gpu": GPU,
        "artifacts": [
            {"name": "pytorch3d", "revision": PYTORCH3D_REVISION, "license": "BSD"},
            {"name": "fused-ssim", "revision": FUSED_SSIM_REVISION, "license": "MIT"},
            {"name": "spz", "revision": SPZ_REVISION, "license": "MIT"},
        ],
        "wheels": wheel_records(
            WHEELS, {"pytorch3d-": "pytorch3d", "fused_ssim-": "fused-ssim", "spz-": "spz"}
        ),
        "smoke": ["gpu-sm120", "pytorch3d-cuda-knn", "fused-ssim-cuda", "spz-import"],
    }
    result = package_bundle(
        tag=TAG, wheel_dir=WHEELS, out_dir=OUT, manifest=manifest, license_dir=LICENSES
    )
    artifacts.commit()
    return result
