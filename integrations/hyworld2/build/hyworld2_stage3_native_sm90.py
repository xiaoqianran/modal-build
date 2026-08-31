from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import modal

ARTIFACT_VOLUME = "modal-build-artifacts"
TAG = "hyworld2-stage3-native-py311-cu128-torch271-sm90-v1"
PYTORCH3D_REVISION = "75ebeeaea0908c5527e7b1e305fbc7681382db47"
GSPLAT_REVISION = "937e29912570c372bed6747a5c9bf85fed877bae"  # v1.5.3
PYTHON, CUDA, TORCH, TORCHVISION = "3.11", "12.8.1", "2.7.1", "0.22.1"
CUDA_ARCH, GPU = "9.0", "H100"
WHEELS, LICENSES, OUT = Path("/tmp/wheels"), Path("/tmp/licenses"), Path("/out")

app = modal.App("modal-build-hyworld2-stage3-native-sm90")
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu22.04", add_python=PYTHON)
    .apt_install("git", "build-essential", "cmake", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja cmake",
        f"python -m pip install torch=={TORCH} torchvision=={TORCHVISION} --index-url https://download.pytorch.org/whl/cu128",
        "python -m pip install numpy==1.26.4 'rich>=12,<14' 'jaxtyping>=0.2,<0.3'",
    )
)


def sh(command: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, shell=True, check=True)


def clone(url: str, dest: Path, revision: str) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    sh(f"git clone --filter=blob:none {url} {dest}")
    sh(f"git checkout --detach {revision}", cwd=dest)


def copy_license(src: Path, target: str) -> None:
    for name in ("LICENSE", "LICENSE.txt", "COPYING"):
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, LICENSES / target)
            return
    raise RuntimeError(f"license missing in {src}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_records() -> list[dict[str, str | int]]:
    records = []
    for wheel in sorted(WHEELS.glob("*.whl")):
        name = wheel.name.lower()
        component = "pytorch3d" if name.startswith("pytorch3d-") else "gsplat"
        records.append({"component": component, "file": wheel.name, "bytes": wheel.stat().st_size, "sha256": sha256(wheel)})
    return records


def package_bundle(manifest: dict) -> dict:
    archive = OUT / f"{TAG}.wheels.zip"
    manifest_path = OUT / f"{TAG}.manifest.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for wheel in sorted(WHEELS.glob("*.whl")):
            zf.write(wheel, f"wheels/{wheel.name}")
        for license_file in sorted(LICENSES.iterdir()):
            zf.write(license_file, f"licenses/{license_file.name}")
    archive_sha = sha256(archive)
    manifest = {**manifest, "archive": archive.name, "archive_sha256": archive_sha}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / f"{TAG}.wheels.zip.sha256").write_text(f"{archive_sha}  {archive.name}\n")
    return manifest


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_HOME": "/usr/local/cuda",
        "CC": "gcc",
        "CXX": "g++",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
        "MAX_JOBS": "4",
    })
    return env


def smoke() -> None:
    import torch
    from gsplat.rendering import rasterization
    from pytorch3d.ops import knn_points

    points = torch.rand((1, 16, 3), device="cuda")
    if knn_points(points, points, K=1).dists.numel() != 16:
        raise RuntimeError("PyTorch3D CUDA KNN smoke failed")

    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    scales = torch.full((1, 3), 0.1, device="cuda")
    opacities = torch.ones(1, device="cuda")
    colors = torch.ones((1, 3), device="cuda")
    viewmats = torch.eye(4, device="cuda")[None]
    Ks = torch.tensor([[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]], device="cuda")
    rendered, alpha, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks,
        width=32, height=32, render_mode="RGB+ED", with_eval3d=True,
    )
    if rendered.shape[-1] != 4 or alpha.max().item() <= 0:
        raise RuntimeError("gsplat eval3d rasterization smoke failed")


@app.function(image=image, gpu=GPU, volumes={"/out": artifacts}, timeout=2 * 60 * 60, max_containers=1)
def build() -> dict:
    import torch

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError(f"expected sm_90, got {torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}")

    for directory in (WHEELS, LICENSES):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
    env = build_env()

    p3d = Path("/tmp/pytorch3d")
    clone("https://github.com/facebookresearch/pytorch3d.git", p3d, PYTORCH3D_REVISION)
    copy_license(p3d, "PyTorch3D-LICENSE.txt")
    sh(f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}", cwd=p3d, env=env)

    gsplat = Path("/tmp/gsplat")
    clone("https://github.com/nerfstudio-project/gsplat.git", gsplat, GSPLAT_REVISION)
    copy_license(gsplat, "gsplat-LICENSE.txt")
    sh(f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}", cwd=gsplat, env=env)

    wheels = sorted(WHEELS.glob("*.whl"))
    if len(wheels) != 2:
        raise RuntimeError(f"expected 2 Stage3 native wheels, got {[path.name for path in wheels]}")
    sh(f"{sys.executable} -m pip install --force-reinstall --no-deps " + " ".join(map(str, wheels)))
    smoke()

    manifest = package_bundle({
        "tag": TAG,
        "bundle_kind": "hyworld2-stage3-native",
        "public_release": True,
        "python": PYTHON,
        "cuda": CUDA,
        "torch": TORCH,
        "torchvision": TORCHVISION,
        "cuda_arch": CUDA_ARCH,
        "target_gpu": GPU,
        "artifacts": [
            {"name": "pytorch3d", "revision": PYTORCH3D_REVISION, "license": "BSD-3-Clause"},
            {"name": "gsplat", "revision": GSPLAT_REVISION, "version": "1.5.3", "license": "Apache-2.0"},
        ],
        "wheels": wheel_records(),
        "smoke": ["gpu-sm90", "pytorch3d-cuda-knn", "gsplat-eval3d-cuda"],
    })
    artifacts.commit()
    return manifest
