"""Pinned GraspGen Franka checkpoint preloader for EmbodiedGen affordance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

GRASPGEN_MODELS_REVISION = "ec1ccbb5eec0680db669246ac312a3636f16ee43"
GRASPGEN_CONFIG_SHA256 = "3b666d28ffb91001ddb6ba24a2e0c11458478a986b808b493cf6fa9a987c2abd"
GRASPGEN_GEN_SHA256 = "0597583b89b322d42ceb4e596967d6ed68d1b56cba4039895909ccd5bdc66eff"
GRASPGEN_DIS_SHA256 = "e47d703c63b54c2d11fbc1effd43898f251b4147250888541e3b16e9c0d19e1c"
GRASPGEN_ROOT = Path("/weights/affordance/graspgen/franka_panda")
GRASPGEN_CONFIG = GRASPGEN_ROOT / "graspgen_franka_panda.yml"
GRASPGEN_GEN = GRASPGEN_ROOT / "graspgen_franka_panda_gen.pth"
GRASPGEN_DIS = GRASPGEN_ROOT / "graspgen_franka_panda_dis.pth"
GRASPGEN_MANIFEST = GRASPGEN_ROOT / "manifest.json"

app = modal.App("modal-build-embodiedgen-graspgen-weights")
weights = modal.Volume.from_name("modal-3d-embodiedgen-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/weights/hf"})
    .pip_install("huggingface_hub==0.34.4", "hf-xet==1.1.8")
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=image,
    volumes={"/weights": weights},
    timeout=45 * 60,
    cpu=2.0,
    memory=4096,
    max_containers=1,
)
def preload_graspgen_weights() -> dict:
    from huggingface_hub import hf_hub_download

    GRASPGEN_ROOT.mkdir(parents=True, exist_ok=True)
    files = {
        "config": "checkpoints/graspgen_franka_panda.yml",
        "generator": "checkpoints/graspgen_franka_panda_gen.pth",
        "discriminator": "checkpoints/graspgen_franka_panda_dis.pth",
    }
    paths: dict[str, Path] = {}
    for key, filename in files.items():
        path = Path(
            hf_hub_download(
                repo_id="adithyamurali/GraspGenModels",
                filename=filename,
                revision=GRASPGEN_MODELS_REVISION,
                local_dir=str(GRASPGEN_ROOT),
            )
        )
        expected_path = GRASPGEN_ROOT / Path(filename).name
        # huggingface_hub keeps the relative checkpoints/ prefix in local_dir.
        if path != expected_path:
            nested = GRASPGEN_ROOT / filename
            if path != nested:
                raise RuntimeError(f"unexpected GraspGen weight path: {path}")
            nested.replace(expected_path)
            path = expected_path
        paths[key] = path

    actual_hashes = {key: sha256(path) for key, path in paths.items()}
    expected_hashes = {
        "config": GRASPGEN_CONFIG_SHA256,
        "generator": GRASPGEN_GEN_SHA256,
        "discriminator": GRASPGEN_DIS_SHA256,
    }
    for key, expected in expected_hashes.items():
        if actual_hashes[key] != expected:
            raise RuntimeError(
                f"GraspGen {key} hash mismatch: {actual_hashes[key]} != {expected}"
            )

    manifest = {
        "revision": GRASPGEN_MODELS_REVISION,
        "files": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual_hashes[key],
            }
            for key, path in paths.items()
        },
    }
    GRASPGEN_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    weights.commit()
    print("GRASPGEN_WEIGHTS_READY", json.dumps(manifest), flush=True)
    return manifest
