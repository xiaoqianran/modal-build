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
    .apt_install(
        "git", "build-essential", "cmake", "ninja-build", "pkg-config", "zlib1g-dev", "libzstd-dev"
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja cmake scikit-build-core pybind11 nanobind python_devtools",
        f"python -m pip install torch=={TORCH} torchvision=={TORCHVISION} --index-url https://download.pytorch.org/whl/cu128",
        "python -m pip install --force-reinstall numpy==1.26.4",
    )
)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CC": "gcc",
            "CXX": "g++",
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
