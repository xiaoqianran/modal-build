from __future__ import annotations

import shutil
import sys
from pathlib import Path

import modal
from _common import ARTIFACT_VOLUME, clone, package_bundle, sh, wheel_records

TAG = "hyworld2-oss-source-py311-v1"
NERFVIEW_REVISION = "4538024fe0d15fd1a0e4d760f3695fc44ca72787"
MOGE_REVISION = "0286b495230a074aadf1c76cc5c679e943e5d1c6"
PYTHON = "3.11"
WHEELS, LICENSES, OUT = Path("/tmp/wheels"), Path("/tmp/licenses"), Path("/out")

app = modal.App("modal-build-hyworld2-oss-source")
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version=PYTHON)
    .apt_install("git", "build-essential")
    .run_commands("python -m pip install --upgrade pip setuptools wheel build")
)


def copy_license(src: Path, target: str) -> None:
    for name in ("LICENSE", "LICENSE.txt", "COPYING"):
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, LICENSES / target)
            return
    raise RuntimeError(f"license missing in {src}")


@app.function(image=image, volumes={"/out": artifacts}, timeout=30 * 60, max_containers=1)
def build() -> dict:
    shutil.rmtree(WHEELS, ignore_errors=True)
    WHEELS.mkdir(parents=True)
    shutil.rmtree(LICENSES, ignore_errors=True)
    LICENSES.mkdir(parents=True)

    nerfview = Path("/tmp/nerfview")
    clone("https://github.com/nerfstudio-project/nerfview.git", nerfview, NERFVIEW_REVISION)
    copy_license(nerfview, "nerfview-LICENSE.txt")
    sh(f"{sys.executable} -m pip wheel . --no-deps -w {WHEELS}", cwd=nerfview)

    moge = Path("/tmp/MoGe")
    clone("https://github.com/microsoft/MoGe.git", moge, MOGE_REVISION)
    copy_license(moge, "MoGe-LICENSE.txt")
    sh(f"{sys.executable} -m pip wheel . --no-deps -w {WHEELS}", cwd=moge)

    wheels = sorted(WHEELS.glob("*.whl"))
    if len(wheels) != 2:
        raise RuntimeError(f"expected nerfview + MoGe wheels, got {[p.name for p in wheels]}")
    manifest = {
        "tag": TAG,
        "bundle_kind": "hyworld2-oss-source-wheels",
        "public_release": True,
        "python": PYTHON,
        "artifacts": [
            {"name": "nerfview", "revision": NERFVIEW_REVISION, "license": "Apache-2.0"},
            {"name": "MoGe", "revision": MOGE_REVISION, "license": "MIT/Apache-2.0"},
        ],
        "wheels": wheel_records(WHEELS, {"nerfview-": "nerfview", "moge-": "MoGe"}),
    }
    result = package_bundle(
        tag=TAG, wheel_dir=WHEELS, out_dir=OUT, manifest=manifest, license_dir=LICENSES
    )
    artifacts.commit()
    return result
