from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import modal

TAG = "glm53-flash-dflash2-vllm0281rc1-fi0618-sm103a-b300-cache-v2"

ARTIFACT_VOLUME = "modal-build-artifacts"
CACHE_VOLUME = "glm53-flash-compile-cache-v1"
LEGACY_VLLM_VOLUME = "glm53-flash-vllm-cache"
LEGACY_FLASHINFER_VOLUME = "glm53-flash-flashinfer-cache"

TARGET_REVISION = "175ae8ce3b5af842b0d0140dbeb43e9cfc557c49"
DRAFTER_REVISION = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
VLLM_IMAGE = "vllm/vllm-openai:glm53-flash"
VLLM_VERSION = "0.28.1rc1"
FLASHINFER_VERSION = "0.6.18"
TARGET_GPU = "B300"
CUDA_ARCH = "sm_103a"

CACHE = Path("/cache")
LEGACY_VLLM = Path("/legacy/vllm")
LEGACY_FLASHINFER = Path("/legacy/flashinfer")
OUT = Path("/out")

app = modal.App("modal-build-glm53-b300-cache")

cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
legacy_vllm = modal.Volume.from_name(LEGACY_VLLM_VOLUME, create_if_missing=True)
legacy_flashinfer = modal.Volume.from_name(
    LEGACY_FLASHINFER_VOLUME, create_if_missing=True
)
artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            # seed-legacy is a migration/backfill step. Never overwrite files
            # produced later by the B300 precompile or production runtime.
            if target.exists():
                continue
            shutil.copy2(item, target)
            copied += 1
    return copied


def _file_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        # Ignore the obsolete first migration target; the real FlashInfer JIT
        # tree lives under flashinfer-workspace/.cache/flashinfer.
        if relative.startswith("flashinfer/"):
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


@app.function(
    image=image,
    volumes={
        str(CACHE): cache_volume,
        str(LEGACY_VLLM): legacy_vllm.with_mount_options(read_only=True),
        str(LEGACY_FLASHINFER): legacy_flashinfer.with_mount_options(read_only=True),
    },
    cpu=4,
    memory=8192,
    timeout=30 * 60,
)
def seed_legacy() -> dict[str, object]:
    """One-time CPU-only migration from the two original cache Volumes."""
    copied_vllm = _copy_tree(LEGACY_VLLM, CACHE / "vllm")
    copied_flashinfer = _copy_tree(
        LEGACY_FLASHINFER,
        CACHE / "flashinfer-workspace/.cache/flashinfer",
    )

    marker = {
        "tag": TAG,
        "seeded_at": datetime.now(UTC).isoformat(),
        "legacy_vllm_files": copied_vllm,
        "legacy_flashinfer_files": copied_flashinfer,
    }
    (CACHE / ".glm53-cache-seed.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache_volume.commit()
    return marker


@app.function(
    image=image,
    volumes={
        str(CACHE): cache_volume.with_mount_options(read_only=True),
        str(OUT): artifacts,
    },
    cpu=4,
    memory=8192,
    timeout=60 * 60,
)
def package() -> dict[str, object]:
    """Package the warmed compile cache into a public GitHub Release bundle."""
    if not CACHE.exists():
        raise RuntimeError(f"cache volume is not mounted at {CACHE}")

    files = _file_records(CACHE)
    if not files:
        raise RuntimeError("refusing to publish an empty GLM53 compile cache")

    required_prefixes = (
        "vllm/",
        "flashinfer-workspace/.cache/flashinfer/",
        "tilelang/",
    )
    present = {prefix: any(str(item["path"]).startswith(prefix) for item in files)
               for prefix in required_prefixes}
    if not present["vllm/"] or not present["flashinfer-workspace/.cache/flashinfer/"]:
        raise RuntimeError(
            f"compile cache is incomplete; required vLLM/FlashInfer trees missing: {present}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"{TAG}.cache.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
        for path in sorted(CACHE.rglob("*")):
            if path.is_file():
                relative = path.relative_to(CACHE).as_posix()
                if relative.startswith("flashinfer/"):
                    continue
                bundle.add(path, arcname=relative, recursive=False)

    archive_sha = _sha256(archive)
    manifest = {
        "tag": TAG,
        "bundle_kind": "glm53-flash-dflash2-b300-compile-cache",
        "public_release": True,
        "created_at": datetime.now(UTC).isoformat(),
        "target_gpu": TARGET_GPU,
        "cuda_arch": CUDA_ARCH,
        "vllm_image": VLLM_IMAGE,
        "vllm_version": VLLM_VERSION,
        "flashinfer_version": FLASHINFER_VERSION,
        "target_model_revision": TARGET_REVISION,
        "drafter_model_revision": DRAFTER_REVISION,
        "contains_model_weights": False,
        "cache_roots": [
            "vllm/",
            "flashinfer-workspace/.cache/flashinfer/",
            "tilelang/",
            "triton/",
            "torchinductor/",
            "nv/",
        ],
        "required_cache_roots": ["vllm/", "flashinfer-workspace/.cache/flashinfer/"],
        "optional_cache_roots": ["tilelang/", "triton/", "torchinductor/", "nv/"],
        "files": files,
        "file_count": len(files),
        "uncompressed_bytes": sum(int(item["bytes"]) for item in files),
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "licenses": {
            "vllm": "Apache-2.0",
            "flashinfer": "Apache-2.0",
            "note": (
                "Bundle contains generated runtime/JIT/autotune caches only. "
                "No GLM or DFlash2 model weights are included."
            ),
        },
    }

    manifest_path = OUT / f"{TAG}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sha_path = OUT / f"{TAG}.cache.tar.gz.sha256"
    sha_path.write_text(f"{archive_sha}  {archive.name}\n", encoding="utf-8")

    artifacts.commit()
    return manifest


@app.function(
    image=image,
    volumes={str(CACHE): cache_volume.with_mount_options(read_only=True)},
    cpu=2,
    memory=2048,
    timeout=10 * 60,
)
def inspect_cache() -> dict[str, object]:
    files = _file_records(CACHE)
    roots: dict[str, int] = {}
    for item in files:
        root = str(item["path"]).split("/", 1)[0]
        roots[root] = roots.get(root, 0) + 1
    return {
        "tag": TAG,
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "roots": roots,
    }


@app.local_entrypoint()
def main(action: str = "inspect"):
    actions = {
        "inspect": inspect_cache,
        "seed-legacy": seed_legacy,
        "package": package,
    }
    if action not in actions:
        raise ValueError(f"unknown action {action!r}; choose from {sorted(actions)}")
    print(json.dumps(actions[action].remote(), indent=2, sort_keys=True))



