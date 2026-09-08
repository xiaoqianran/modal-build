from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import modal

ARTIFACT_VOLUME = "modal-build-artifacts"
REPOSITORY = "xiaoqianran/modal-build"

app = modal.App("modal-build-hyworld2-restore")
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "modal-build-artifact-restore/1"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, volumes={"/out": artifacts}, timeout=15 * 60, max_containers=1)
def restore_public_bundle(tag: str, expected_sha256: str) -> dict[str, object]:
    """Restore one publicly released HYWorld2 bundle into the private build-artifact Volume."""
    tag = str(tag).strip()
    expected_sha256 = str(expected_sha256).strip().lower()
    if not tag or len(expected_sha256) != 64:
        raise ValueError("tag and expected_sha256 are required")

    root = Path("/out")
    archive = root / f"{tag}.wheels.zip"
    manifest_path = root / f"{tag}.manifest.json"
    sidecar = root / f"{tag}.wheels.zip.sha256"

    if archive.is_file() and manifest_path.is_file() and _sha256(archive) == expected_sha256:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("tag") == tag and manifest.get("archive_sha256") == expected_sha256:
            return {"status": "ready", "tag": tag, "sha256": expected_sha256}

    base = f"https://github.com/{REPOSITORY}/releases/download/{tag}"
    tmp = Path("/tmp") / f"restore-{tag}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tmp_manifest = tmp / manifest_path.name
    tmp_archive = tmp / archive.name
    _download(f"{base}/{manifest_path.name}", tmp_manifest)
    manifest = json.loads(tmp_manifest.read_text(encoding="utf-8"))
    if manifest.get("tag") != tag or manifest.get("public_release") is not True:
        raise RuntimeError(f"release manifest is not public or has wrong tag: {tag}")
    if manifest.get("archive_sha256") != expected_sha256:
        raise RuntimeError(f"release manifest checksum mismatch: {tag}")
    _download(f"{base}/{archive.name}", tmp_archive)
    actual = _sha256(tmp_archive)
    if actual != expected_sha256:
        raise RuntimeError(f"release archive checksum mismatch for {tag}: {actual}")

    shutil.copy2(tmp_archive, archive)
    shutil.copy2(tmp_manifest, manifest_path)
    sidecar.write_text(f"{actual}  {archive.name}\n", encoding="utf-8")
    artifacts.commit()
    return {"status": "restored", "tag": tag, "sha256": actual}
