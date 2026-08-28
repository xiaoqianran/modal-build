from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import modal

TAG = "hunyuan3d-2.1-paint-py311-cu124-torch251-sm89-v2"
FORK = "Archerkattri/hunyuan2.1-plus-plus"
FORK_COMMIT = "9efd760fbec8ab490e68b330225ea1fab10de7fd"
BUNDLE = Path("/tmp/bundle")
WHEELS = BUNDLE / "wheels"
NATIVE = BUNDLE / "native"
OUT = Path("/out")
build_artifacts = modal.Volume.from_name("modal-build-artifacts", create_if_missing=True)

app = modal.App("modal-build-hunyuan3d21-paint-v2")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade uv setuptools wheel packaging ninja pybind11==2.13.4",
        "uv pip install --system torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124",
    )
)


def sh(cmd: str, cwd: str | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


def file_info(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/out": build_artifacts},
    timeout=60 * 60,
    max_containers=1,
)
def build() -> dict:
    WHEELS.mkdir(parents=True, exist_ok=True)
    NATIVE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = (
        "CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda "
        "TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 FORCE_CUDA=1"
    )

    sh(f"git clone https://github.com/{FORK}.git /tmp/hunyuan")
    sh(f"git checkout {FORK_COMMIT}", "/tmp/hunyuan")

    sh(
        f"{env} python -m pip wheel . --no-build-isolation --no-deps -w {WHEELS}",
        "/tmp/hunyuan/hy3dpaint/custom_rasterizer",
    )

    renderer = Path("/tmp/hunyuan/hy3dpaint/DifferentiableRenderer")
    sh(
        "c++ -O3 -Wall -shared -std=c++11 -fPIC "
        "$(python -m pybind11 --includes) mesh_inpaint_processor.cpp "
        "-o mesh_inpaint_processor$(python3-config --extension-suffix)",
        str(renderer),
    )
    compiled = list(renderer.glob("mesh_inpaint_processor*.so"))
    if len(compiled) != 1:
        raise RuntimeError(f"expected one mesh inpaint extension, got {compiled}")
    shutil.copy2(compiled[0], NATIVE / compiled[0].name)

    wheels = [file_info(path) for path in sorted(WHEELS.glob("*.whl"))]
    native = [file_info(path) for path in sorted(NATIVE.glob("*.so"))]
    if len(wheels) != 1 or len(native) != 1:
        raise RuntimeError(f"unexpected bundle contents wheels={wheels}, native={native}")

    # Sanity-check both compiled artifacts in the exact build ABI before publishing.
    sh(f"uv pip install --system --no-deps {WHEELS}/*.whl")
    sh(
        f"PYTHONPATH={renderer.parent} python -c \"import torch; import custom_rasterizer; "
        f"import sys; sys.path.insert(0, '{renderer}'); import mesh_inpaint_processor; "
        "assert callable(mesh_inpaint_processor.meshVerticeInpaint)\""
    )

    archive = Path(shutil.make_archive(str(OUT / f"{TAG}.bundle"), "zip", BUNDLE))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "tag": TAG,
        "python": "3.11",
        "cuda": "12.4.1",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "pybind11": "2.13.4",
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "source": FORK,
        "source_revision": FORK_COMMIT,
        "wheels": wheels,
        "native_extensions": native,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
    }
    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sha_path = OUT / f"{TAG}.bundle.zip.sha256"
    sha_path.write_text(f"{archive_sha}  {archive.name}\n")
    build_artifacts.commit()
    return manifest
