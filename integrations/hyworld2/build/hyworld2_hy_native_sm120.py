from __future__ import annotations

import hashlib
import inspect
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


TAG = "hyworld2-hy-native-py311-cu128-torch271-sm120-v1"
HY_REPO = "https://github.com/Tencent-Hunyuan/HY-World-2.0.git"
HY_REVISION = "df9988efb87bfc0f4947eb3889411cf957478b06"
RECAST_REVISION = "9f4ce64458dfae86e1239c525ddc219c4e9e06f1"
GLM_REVISION = "33b4a621a697a305bc3a7610d290677b96beb181"
PYTHON, CUDA, TORCH, TORCHVISION = "3.11", "12.8.1", "2.7.1", "0.22.1"
CUDA_ARCH, GPU = "12.0", "RTX-PRO-6000"
WHEELS, LICENSES, OUT = Path("/tmp/wheels"), Path("/tmp/licenses"), Path("/out")

app = modal.App("modal-build-hyworld2-hy-native-sm120")
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu22.04", add_python=PYTHON)
    .apt_install("git", "build-essential", "cmake", "ninja-build", "pkg-config")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja pybind11",
        "python -m pip install numpy==1.26.4 'rich>=12,<14' 'jaxtyping>=0.2,<0.3'",
        f"python -m pip install torch=={TORCH} torchvision=={TORCHVISION} --index-url https://download.pytorch.org/whl/cu128",
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


def verify_gsplat() -> None:
    import gsplat
    import torch
    from gsplat.rendering import rasterization

    missing = {"distloss", "gauss_masks"} - set(inspect.signature(rasterization).parameters)
    if missing:
        raise RuntimeError(f"wrong gsplat installed; missing HY-World kwargs: {sorted(missing)}")
    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    scales = torch.tensor([[0.1, 0.1, 0.1]], device="cuda")
    opacities = torch.tensor([0.9], device="cuda")
    colors = torch.tensor([[[0.8, 0.2, 0.1]]], device="cuda")
    viewmats = torch.eye(4, device="cuda")[None]
    ks = torch.tensor([[[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]], device="cuda")
    _, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=ks,
        width=64,
        height=64,
        sh_degree=0,
        packed=False,
        distloss=True,
        gauss_masks=torch.ones((1,), device="cuda"),
    )
    if alpha.max().item() <= 0:
        raise RuntimeError("HY-World gsplat CUDA smoke rendered empty alpha")
    print("HY-World gsplat smoke passed", gsplat.__version__)


def verify_recast() -> None:
    import recast

    if not hasattr(recast, "RecastNavMesh"):
        raise RuntimeError("recast wheel imported but RecastNavMesh is missing")


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

    src = Path("/tmp/HY-World-2.0")
    clone(HY_REPO, src, HY_REVISION)
    shutil.copy2(src / "License.txt", LICENSES / "Tencent-HY-WORLD-2.0-License.txt")
    (LICENSES / "NOTICE.txt").write_text(
        "Tencent HY-WORLD 2.0 is licensed under the Tencent HY-WORLD 2.0 Community License Agreement, "
        "Copyright © 2026 Tencent. All Rights Reserved. The trademark rights of Tencent HY are owned by Tencent or its affiliate.\n"
        "Packaging metadata was modified only to give the cached gsplat wheel a collision-resistant local version.\n"
    )

    gsplat_dir = src / "hyworld2/worldgen/third_party/gsplat_maskgaussian"
    glm_dir = gsplat_dir / "gsplat/cuda/csrc/third_party/glm"
    clone("https://github.com/g-truc/glm.git", glm_dir, GLM_REVISION)
    shutil.copy2(glm_dir / "copying.txt", LICENSES / "GLM-copying.txt")
    (gsplat_dir / "gsplat/version.py").write_text(
        f'__version__ = "1.5.3+hyworld2.{HY_REVISION[:8]}.sm120"\n'
    )
    sh(
        f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        cwd=gsplat_dir,
        env=env,
    )

    recast_dir = src / "hyworld2/worldgen/third_party/recastnavigation"
    clone("https://github.com/recastnavigation/recastnavigation.git", recast_dir, RECAST_REVISION)
    navmesh_dir = src / "hyworld2/worldgen/third_party/navmesh"
    nav_env = env.copy()
    nav_env["RECAST_PATH"] = str(recast_dir)
    sh(
        f"{sys.executable} -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        cwd=navmesh_dir,
        env=nav_env,
    )

    wheels = sorted(WHEELS.glob("*.whl"))
    if len(wheels) != 2:
        raise RuntimeError(f"expected gsplat + recast wheels, got {[p.name for p in wheels]}")
    sh(f"{sys.executable} -m pip install --force-reinstall --no-deps " + " ".join(map(str, wheels)))
    verify_gsplat()
    verify_recast()

    manifest = {
        "tag": TAG,
        "bundle_kind": "hyworld2-restricted-native",
        "public_release": False,
        "public_release_reason": "HY-WORLD-derived binaries are Territory-restricted; keep in Modal Volume.",
        "python": PYTHON,
        "cuda": CUDA,
        "torch": TORCH,
        "torchvision": TORCHVISION,
        "cuda_arch": CUDA_ARCH,
        "target_gpu": GPU,
        "source": "Tencent-Hunyuan/HY-World-2.0",
        "source_revision": HY_REVISION,
        "recast_revision": RECAST_REVISION,
        "glm_revision": GLM_REVISION,
        "wheels": wheel_records(
            WHEELS, {"gsplat-": "hyworld2-gsplat-maskgaussian", "recast-": "hyworld2-navmesh"}
        ),
        "smoke": ["gpu-sm120", "gsplat-distloss-gauss_masks-cuda", "recast-import"],
    }
    result = package_bundle(
        tag=TAG, wheel_dir=WHEELS, out_dir=OUT, manifest=manifest, license_dir=LICENSES
    )
    artifacts.commit()
    return result
