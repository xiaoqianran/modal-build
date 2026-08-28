"""Pinned model-weight preloader for the EmbodiedGen P3-SAM affordance runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

HUNYUAN3D_PART_MODEL_REVISION = "677174466c53571e8bacd5050dff5948734a1a4d"
SONATA_MODEL_REVISION = "df99897472c09f91ba9288da0a034aacffc0b010"
P3SAM_WEIGHT_SHA256 = "eb76550cfbe06f154c6e9b17167ccfc28222bb4a216ec7b12ac2bf7d762de38c"
SONATA_WEIGHT_SHA256 = "c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50"
P3SAM_WEIGHT = Path("/weights/affordance/p3sam/p3sam.safetensors")
SONATA_WEIGHT = Path("/weights/affordance/sonata/sonata.pth")
WEIGHT_MANIFEST = Path("/weights/affordance/manifest.json")

app = modal.App("modal-build-embodiedgen-affordance-weights")
weights = modal.Volume.from_name("modal-3d-embodiedgen-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/weights/hf"})
    .pip_install("huggingface_hub==0.34.4")
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=image,
    volumes={"/weights": weights},
    timeout=30 * 60,
    cpu=2.0,
    memory=4096,
    max_containers=1,
)
def preload_affordance_weights() -> dict:
    from huggingface_hub import hf_hub_download

    P3SAM_WEIGHT.parent.mkdir(parents=True, exist_ok=True)
    SONATA_WEIGHT.parent.mkdir(parents=True, exist_ok=True)

    p3sam = Path(
        hf_hub_download(
            repo_id="tencent/Hunyuan3D-Part",
            filename="p3sam/p3sam.safetensors",
            revision=HUNYUAN3D_PART_MODEL_REVISION,
            local_dir=str(P3SAM_WEIGHT.parent.parent),
        )
    )
    if p3sam != P3SAM_WEIGHT:
        raise RuntimeError(f"unexpected P3-SAM weight path: {p3sam}")

    sonata = Path(
        hf_hub_download(
            repo_id="facebook/sonata",
            filename="sonata.pth",
            revision=SONATA_MODEL_REVISION,
            local_dir=str(SONATA_WEIGHT.parent),
        )
    )
    if sonata != SONATA_WEIGHT:
        raise RuntimeError(f"unexpected Sonata weight path: {sonata}")

    actual_p3sam_sha256 = sha256(p3sam)
    actual_sonata_sha256 = sha256(sonata)
    if actual_p3sam_sha256 != P3SAM_WEIGHT_SHA256:
        raise RuntimeError(f"P3-SAM weight hash mismatch: {actual_p3sam_sha256}")
    if actual_sonata_sha256 != SONATA_WEIGHT_SHA256:
        raise RuntimeError(f"Sonata weight hash mismatch: {actual_sonata_sha256}")

    manifest = {
        "hunyuan3d_part_model_revision": HUNYUAN3D_PART_MODEL_REVISION,
        "sonata_model_revision": SONATA_MODEL_REVISION,
        "p3sam": {"path": str(P3SAM_WEIGHT), "bytes": p3sam.stat().st_size, "sha256": actual_p3sam_sha256},
        "sonata": {"path": str(SONATA_WEIGHT), "bytes": sonata.stat().st_size, "sha256": actual_sonata_sha256},
    }
    WEIGHT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    WEIGHT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    weights.commit()
    print("AFFORDANCE_WEIGHTS_READY", json.dumps(manifest), flush=True)
    return manifest
