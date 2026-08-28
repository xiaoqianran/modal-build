"""Pinned BiRefNet weight preloader for modal-3D input conditioning."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import modal

MODEL_NAME = "birefnet-general-lite"
MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
    "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
)
MODEL_BYTES = 224_005_088
MODEL_MD5 = "4fab47adc4ff364be1713e97b7e66334"
MODEL_SHA256 = "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333"
WEIGHT_ROOT = Path("/weights/rembg")
MODEL_PATH = WEIGHT_ROOT / "models" / MODEL_NAME / f"{MODEL_NAME}.onnx"
WEIGHT_MANIFEST = WEIGHT_ROOT / "manifest.json"
VOLUME_NAME = "modal-3d-birefnet-weights"

app = modal.App("modal-build-birefnet-weights")
weights = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").env({"PYTHONUNBUFFERED": "1"})


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weight(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    md5 = digest(path, "md5")
    sha256 = digest(path, "sha256")
    if size != MODEL_BYTES:
        raise RuntimeError(f"BiRefNet weight size mismatch: {size}")
    if md5 != MODEL_MD5:
        raise RuntimeError(f"BiRefNet weight MD5 mismatch: {md5}")
    if sha256 != MODEL_SHA256:
        raise RuntimeError(f"BiRefNet weight SHA-256 mismatch: {sha256}")
    return {"bytes": size, "md5": md5, "sha256": sha256}


@app.function(
    image=image,
    volumes={"/weights": weights},
    timeout=20 * 60,
    cpu=2.0,
    memory=2048,
    max_containers=1,
)
def preload_birefnet_weights() -> dict[str, object]:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        verified = verify_weight(MODEL_PATH)
    except (FileNotFoundError, RuntimeError):
        temporary = MODEL_PATH.with_suffix(".download")
        temporary.unlink(missing_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, temporary)
            verified = verify_weight(temporary)
            temporary.replace(MODEL_PATH)
        finally:
            temporary.unlink(missing_ok=True)

    manifest = {
        "model": MODEL_NAME,
        "source": MODEL_URL,
        "path": str(MODEL_PATH),
        **verified,
    }
    WEIGHT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    WEIGHT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    weights.commit()
    print("BIREFNET_WEIGHTS_READY", json.dumps(manifest), flush=True)
    return manifest
