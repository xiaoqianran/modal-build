from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import modal

TAG = "trellis.cpp-pynone-cu129-torchnone-sm89-v2"
REPO = "xiaoqianran/modal-build"
COMMIT = "16f3109e82f3922033bfa62b83c42899678b7b6f"
OUT = Path("/tmp/out")
DIST = Path("/tmp/dist")
SRC = Path("/tmp/trellis.cpp")

app = modal.App("modal-build-trellis.cpp")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "gh", "patchelf", "build-essential", "curl", "ca-certificates", "ccache")
    .run_commands("python -m pip install --upgrade uv", "uv pip install --system cmake ninja")
)


def sh(cmd: str, cwd: str | None = None):
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=True)


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    secrets=[modal.Secret.from_name("modal-build-github")],
    timeout=60 * 60,
    max_containers=1,
)
def build_and_release() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    DIST.mkdir(parents=True, exist_ok=True)

    sh(f"git clone --recursive https://github.com/pwilkin/trellis.cpp.git '{SRC}'")
    sh(f"git checkout {COMMIT}", str(SRC))
    sh("git submodule update --init --recursive", str(SRC))
    sh(
        "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "
        "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89",
        str(SRC),
    )
    sh("cmake --build build --target trellis-server trellis-cli -j8", str(SRC))

    for name in ("trellis-server", "trellis-cli"):
        shutil.copy2(SRC / "build" / name, DIST / name)
    for p in (SRC / "build").glob("libggml*.so*"):
        shutil.copy2(p, DIST / p.name)

    # CUDA runtime is supplied by the pinned NVIDIA runtime image.
    # Bundle only our compiled artifacts.


    for p in DIST.iterdir():
        if p.is_file() and not p.is_symlink():
            subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(p)], check=False)

    ldd = {}
    for name in ("trellis-server", "trellis-cli"):
        result = subprocess.run(
            ["ldd", str(DIST / name)], capture_output=True, text=True, check=True
        )
        missing = [line.strip() for line in result.stdout.splitlines() if "not found" in line]
        unsupported = [line for line in missing if not line.startswith("libcuda.so.1 ")]
        if unsupported:
            raise RuntimeError(f"unresolved runtime dependency for {name}: {unsupported}")
        ldd[name] = result.stdout

    archive = OUT / f"{TAG}.tar.gz"
    sh(f"tar -C '{DIST}' -czf '{archive}' .")
    manifest = {
        "tag": TAG,
        "source": "pwilkin/trellis.cpp",
        "commit": COMMIT,
        "python": None,
        "cuda": "12.9.1",
        "torch": None,
        "cuda_arch": "8.9",
        "target_gpu": "L40S",
        "backend": "GGML_CUDA",
        "runtime_base": "nvidia/cuda:12.9.1-runtime-ubuntu22.04 + libgomp1",
        "ldd": ldd,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": [],
    }
    for p in sorted(DIST.iterdir()):
        if p.is_file() and not p.is_symlink():
            manifest["files"].append(
                {"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            )
    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sha_path = OUT / f"{archive.name}.sha256"
    sha_path.write_text(f"{manifest['archive_sha256']}  {archive.name}\n")

    exists = subprocess.run(
        ["gh", "release", "view", TAG, "--repo", REPO], capture_output=True, check=False
    ).returncode == 0
    if not exists:
        sh(f"gh release create '{TAG}' --repo '{REPO}' --title '{TAG}' --notes 'trellis.cpp native CUDA/SM89 bundle for Modal L40S.'")
    sh(f"gh release upload '{TAG}' --repo '{REPO}' --clobber '{archive}' '{manifest_path}' '{sha_path}'")
    return manifest
