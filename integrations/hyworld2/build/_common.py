from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
