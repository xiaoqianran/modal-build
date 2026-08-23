"""EmbodiedGen v2.0.0 L40S release-consumer runtime.

This file intentionally uses nvidia/cuda:*runtime* (not devel): nvcc is absent.
All expensive CUDA artifacts are consumed from the modal-build GitHub Release.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

TAG = "embodiedgen-v2.0.0-py310-cu126-torch280-sm89-v1"
RELEASE = f"https://github.com/xiaoqianran/modal-build/releases/download/{TAG}"
APP_NAME = "modal-3d-embodiedgen"
EMBODIEDGEN_COMMIT = "cc3015ca5ccdacf94df3428d9e65f79375982216"
CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
KOLORS_COMMIT = "c59c0aa67587e472de657bc9f4f9c18272c94165"
RELEASE_WHEELS_SHA256 = "4168abccbc9a0033825e3ad8b9a9e992795f6449107adf357a4dd4acafec398c"
RELEASE_EXTENSIONS_SHA256 = "e5e1991ec465b399d46bca271af46394b054afd9eefdbcdcd8b5329f4c8e5bb3"
SAM3D_STAGE1_STEPS = 16
SAM3D_STAGE2_STEPS = 16
TARGET_MESH_FACES = 50_000
JOB_ROOT = Path("/artifacts/embodiedgen/jobs")
API_JOB_PREFIX = "job-"
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_INPUT_PIXELS = 40_000_000
API_RESULT_TTL_SECONDS = 7 * 24 * 60 * 60
API_FAILED_TTL_SECONDS = 24 * 60 * 60
API_ACTIVE_STALE_SECONDS = 6 * 60 * 60
MAX_PROMPT_CHARS = 1000
TEXT2IMG_MODEL_ID = "Kwai-Kolors/Kolors-diffusers"
TEXT2IMG_MODEL_REVISION = "7e091c75199e910a26cd1b51ed52c28de5db3711"
TEXT2IMG_MODEL_DIR = "/weights/text2img/kolors-diffusers"
TEXT2IMG_REVISION_MARKER = ".modal-build-revision"
RETEXTURE_MODEL_ID = "xinjjj/RoboAssetGen"
RETEXTURE_MODEL_REVISION = "a64fcdebeea17287d830736cd0853df1093b97ab"
RETEXTURE_MODEL_DIR = "/weights/retexture/roboassetgen"
RETEXTURE_REVISION_MARKER = ".modal-build-revision"
RETEXTURE_CONTROLNET_DIR = f"{RETEXTURE_MODEL_DIR}/texture_gen_mv_v1"
RETEXTURE_SR_PATH = f"{RETEXTURE_MODEL_DIR}/super_resolution/RealESRGAN_x4plus.pth"
TEXT2IMG_SCALEDOWN_WINDOWS = {
    "min_cost": 2,
    "cost_first": 30,
    "balanced": 90,
    "burst": 180,
}
RETEXTURE_SCALEDOWN_WINDOWS = dict(TEXT2IMG_SCALEDOWN_WINDOWS)

STATIC_AUTOSCALE_PROFILE = "cost_first"
DEFAULT_REQUEST_PROFILE = "auto"
AUTOSCALE_PROFILES = {
    "min_cost": {"rembg": 2, "sam3d": 2, "mesh": 2, "lite": 2, "finalize": 2},
    "cost_first": {"rembg": 60, "sam3d": 30, "mesh": 30, "lite": 10, "finalize": 2},
    "balanced": {"rembg": 120, "sam3d": 90, "mesh": 90, "lite": 30, "finalize": 10},
    "burst": {"rembg": 300, "sam3d": 180, "mesh": 120, "lite": 60, "finalize": 30},
}
AUTO_TRAFFIC_WINDOW_SECONDS = 60.0
AUTO_COST_FIRST_REQUESTS = 2
TRAFFIC_EVENT_PREFIX = "request:"
_active_autoscale_profile: str | None = None

# Current workspace rates from `modal billing rates` on 2026-08-23.
RATE_CPU_CORE_HOUR = 0.04730
RATE_MEMORY_GIB_HOUR = 0.00800
RATE_L40S_HOUR = 1.95
RESOURCE_HOURLY_COST = {
    "rembg": 1 * RATE_CPU_CORE_HOUR + 4 * RATE_MEMORY_GIB_HOUR,
    "sam3d": RATE_L40S_HOUR + 6 * RATE_CPU_CORE_HOUR + 32 * RATE_MEMORY_GIB_HOUR,
    "mesh": 4 * RATE_CPU_CORE_HOUR + 8 * RATE_MEMORY_GIB_HOUR,
    "lite": RATE_L40S_HOUR + 4 * RATE_CPU_CORE_HOUR + 16 * RATE_MEMORY_GIB_HOUR,
    "finalize": 4 * RATE_CPU_CORE_HOUR + 16 * RATE_MEMORY_GIB_HOUR,
    "text2image": RATE_L40S_HOUR + 4 * RATE_CPU_CORE_HOUR + 32 * RATE_MEMORY_GIB_HOUR,
    "retexture": RATE_L40S_HOUR + 4 * RATE_CPU_CORE_HOUR + 32 * RATE_MEMORY_GIB_HOUR,
}


def autoscale_profile_summary(name: str) -> dict:
    if name not in AUTOSCALE_PROFILES:
        raise ValueError(f"unknown autoscale profile {name!r}; choose {sorted(AUTOSCALE_PROFILES)}")
    windows = AUTOSCALE_PROFILES[name]
    idle_tail = {
        stage: RESOURCE_HOURLY_COST[stage] / 3600.0 * seconds
        for stage, seconds in windows.items()
    }
    return {
        "profile": name,
        "scaledown_window_seconds": dict(windows),
        "idle_tail_cost_usd": {k: round(v, 8) for k, v in idle_tail.items()},
        "idle_tail_total_usd": round(sum(idle_tail.values()), 8),
    }


def text_autoscale_profile_summary(name: str) -> dict:
    """Idle-tail ceiling for Text→Image plus the existing Image→3D pipeline."""
    image_summary=autoscale_profile_summary(name)
    seconds=TEXT2IMG_SCALEDOWN_WINDOWS[name]
    text_tail=RESOURCE_HOURLY_COST["text2image"] / 3600.0 * seconds
    return {
        "profile":name,
        "image_to_3d_idle_tail_usd":image_summary["idle_tail_total_usd"],
        "text2image_scaledown_window_seconds":seconds,
        "text2image_idle_tail_usd":round(text_tail,8),
        "text_to_3d_idle_tail_total_usd":round(
            image_summary["idle_tail_total_usd"]+text_tail,
            8,
        ),
    }


def retexture_autoscale_profile_summary(name: str) -> dict:
    if name not in RETEXTURE_SCALEDOWN_WINDOWS:
        raise ValueError(f"unknown autoscale profile {name!r}; choose {sorted(RETEXTURE_SCALEDOWN_WINDOWS)}")
    seconds=RETEXTURE_SCALEDOWN_WINDOWS[name]
    tail=RESOURCE_HOURLY_COST["retexture"] / 3600.0 * seconds
    return {
        "profile":name,
        "scaledown_window_seconds":seconds,
        "idle_tail_cost_usd":round(tail,8),
    }


def auto_profile_for_timestamps(timestamps, now: float) -> tuple[str, int]:
    """Pure cost-first classifier; BALANCED/BURST are intentionally manual only."""
    recent = sum(
        1
        for timestamp in timestamps
        if 0.0 <= now - float(timestamp) <= AUTO_TRAFFIC_WINDOW_SECONDS
    )
    profile = "cost_first" if recent >= AUTO_COST_FIRST_REQUESTS else "min_cost"
    return profile, recent


def select_request_profile(requested: str = DEFAULT_REQUEST_PROFILE, now: float | None = None) -> dict:
    """Resolve an explicit profile or record one request and select AUTO cheaply.

    AUTO stores one independent event per request, so concurrent writers cannot overwrite a shared
    counter/list. Old events are opportunistically deleted on the next request.
    """
    if requested != "auto":
        summary = autoscale_profile_summary(requested)
        return {"requested_profile": requested, "selected_profile": requested, **summary}

    import uuid

    now = time.time() if now is None else float(now)
    traffic_events.put(f"{TRAFFIC_EVENT_PREFIX}{now:.6f}:{uuid.uuid4().hex}", now)
    recent_timestamps = []
    stale_keys = []
    for key, timestamp in traffic_events.items():
        if not str(key).startswith(TRAFFIC_EVENT_PREFIX):
            continue
        age = now - float(timestamp)
        if 0.0 <= age <= AUTO_TRAFFIC_WINDOW_SECONDS:
            recent_timestamps.append(float(timestamp))
        elif age > AUTO_TRAFFIC_WINDOW_SECONDS:
            stale_keys.append(key)
    for key in stale_keys:
        traffic_events.pop(key, None)

    selected, recent = auto_profile_for_timestamps(recent_timestamps, now)
    summary = autoscale_profile_summary(selected)
    return {
        "requested_profile": "auto",
        "selected_profile": selected,
        "recent_requests_60s": recent,
        "pruned_events": len(stale_keys),
        **summary,
    }


def normalize_text_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prompt must be a string")
    prompt = value.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
    return prompt


def new_job_id() -> str:
    import uuid

    return f"{API_JOB_PREFIX}{uuid.uuid4().hex}"


def is_api_job_id(job_id: str) -> bool:
    import re

    return bool(re.fullmatch(r"job-[0-9a-f]{32}", job_id))


def api_job_root(job_id: str) -> Path:
    if not is_api_job_id(job_id):
        raise ValueError(f"invalid API job id: {job_id!r}")
    return JOB_ROOT / job_id


def resolve_job_input(job_id: str, root: Path, fallback: Path) -> tuple[Path, str]:
    """API jobs require an uploaded image; legacy/debug jobs may use the sample fallback."""
    uploaded = root / "input_image"
    if is_api_job_id(job_id):
        if not uploaded.is_file():
            raise FileNotFoundError(f"missing uploaded/generated image for API job {job_id}")
        source = "generated-text" if (root / "prompt.txt").is_file() else "uploaded"
        return uploaded, source
    if uploaded.is_file():
        return uploaded, "uploaded"
    return fallback, "benchmark-sample"


def prune_job_intermediates(root: Path) -> list[str]:
    """Keep only final deliverables and the validation report."""
    import shutil

    keep = {"result", "validation_report.json"}
    removed = []
    if not root.exists():
        return removed
    for path in root.iterdir():
        if path.name in keep:
            continue
        removed.append(path.name)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    return sorted(removed)


def api_job_expired(status: dict | None, *, now: float, fallback_mtime: float) -> bool:
    updated = float(status.get("updated_epoch", fallback_mtime)) if status else fallback_mtime
    state = status.get("status") if status else None
    if state in {"queued", "running"}:
        return now - updated >= API_ACTIVE_STALE_SECONDS
    ttl = API_FAILED_TTL_SECONDS if state == "failed" else API_RESULT_TTL_SECONDS
    return now - updated >= ttl


RESULT_FILES = {
    "glb": "result/mesh/sample_00.glb",
    "obj": "result/mesh/sample_00.obj",
    "mtl": "result/mesh/material.mtl",
    "obj_texture": "result/mesh/material_0.png",
    "urdf": "result/sample_00.urdf",
    "video": "result/video.mp4",
    "gs_ply": "result/mesh/sample_00_gs.ply",
    "gs_aligned_ply": "result/mesh/sample_00_gs_aligned.ply",
    "validation": "validation_report.json",
}


def simplify_mesh_if_needed(vertices, faces, simplify_fn, target_faces: int = TARGET_MESH_FACES):
    """Simplify only oversized meshes; small meshes pass through unchanged."""
    if len(faces) <= target_faces:
        return vertices, faces, False
    vertices, faces = simplify_fn(
        vertices,
        faces,
        target_count=target_faces,
        agg=7.0,
        preserve_border=False,
    )
    return vertices, faces, True


def obj_material_dependencies(obj_path: Path) -> list[Path]:
    """Return generated MTL/texture files referenced by an OBJ, without leaving its directory."""
    import shlex

    root = obj_path.parent.resolve()
    dependencies = []
    material_files = []
    for raw_line in obj_path.read_text(errors="ignore").splitlines():
        tokens = shlex.split(raw_line, comments=True)
        if tokens and tokens[0].lower() == "mtllib":
            material_files.extend(tokens[1:])

    def add_relative(reference: str, base: Path = root):
        candidate = (base / reference).resolve()
        if candidate != root and root not in candidate.parents:
            raise RuntimeError(f"OBJ material reference escapes job directory: {reference}")
        if candidate not in dependencies:
            dependencies.append(candidate)

    for material in material_files:
        add_relative(material)
        mtl_path = (root / material).resolve()
        if not mtl_path.exists():
            continue
        for raw_line in mtl_path.read_text(errors="ignore").splitlines():
            tokens = shlex.split(raw_line, comments=True)
            if not tokens:
                continue
            directive = tokens[0].lower()
            if (
                directive.startswith("map_") or directive in {"bump", "disp", "decal", "norm"}
            ) and len(tokens) > 1:
                # Trimesh emits simple references; the final token also handles common MTL options.
                add_relative(tokens[-1], mtl_path.parent)
    return dependencies


def copy_obj_bundle(obj_path: Path, destination: Path) -> None:
    """Copy OBJ plus every referenced MTL/texture asset as a self-contained bundle."""
    import shutil

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(obj_path, destination / obj_path.name)
    source_root = obj_path.parent.resolve()
    for dependency in obj_material_dependencies(obj_path):
        if not dependency.exists():
            raise FileNotFoundError(f"missing OBJ dependency: {dependency}")
        relative = dependency.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dependency, target)


def missing_obj_material_dependencies(obj_path: Path) -> list[str]:
    root = obj_path.parent.resolve()
    return [str(path.relative_to(root)) for path in obj_material_dependencies(obj_path) if not path.exists()]


def validation_passes(checks: dict) -> bool:
    required_positive = ("ply_vertices", "obj_vertices", "obj_faces", "glb_geometries")
    return all(checks[name] > 0 for name in required_positive) and all(
        checks[name]
        for name in ("urdf_mesh_exists", "video_exists", "obj_material_refs_ok")
    )


async def select_request_profile_aio(
    requested: str = DEFAULT_REQUEST_PROFILE, now: float | None = None
) -> dict:
    """Async ASGI variant of select_request_profile; never blocks the event loop."""
    if requested != "auto":
        summary = autoscale_profile_summary(requested)
        return {"requested_profile": requested, "selected_profile": requested, **summary}

    import uuid

    now = time.time() if now is None else float(now)
    await traffic_events.put.aio(f"{TRAFFIC_EVENT_PREFIX}{now:.6f}:{uuid.uuid4().hex}", now)
    recent_timestamps=[]
    stale_keys=[]
    async for key,timestamp in traffic_events.items.aio():
        if not str(key).startswith(TRAFFIC_EVENT_PREFIX):
            continue
        age=now-float(timestamp)
        if 0.0 <= age <= AUTO_TRAFFIC_WINDOW_SECONDS:
            recent_timestamps.append(float(timestamp))
        elif age > AUTO_TRAFFIC_WINDOW_SECONDS:
            stale_keys.append(key)
    for key in stale_keys:
        await traffic_events.pop.aio(key,None)

    selected,recent=auto_profile_for_timestamps(recent_timestamps,now)
    summary=autoscale_profile_summary(selected)
    return {
        "requested_profile":"auto",
        "selected_profile":selected,
        "recent_requests_60s":recent,
        "pruned_events":len(stale_keys),
        **summary,
    }


def runtime_handles():
    """Return the single shared pool handles used by every autoscale profile."""
    return RembgWorker(), Sam3DWorker(), MeshWorker(), lite_gpu_bake, cpu_finalize


def apply_autoscale_profile(profile: str, handles=None) -> tuple:
    """Update the shared pools only when this control process changes profile."""
    global _active_autoscale_profile

    cfg = AUTOSCALE_PROFILES.get(profile)
    if cfg is None:
        raise ValueError(f"unknown autoscale profile {profile!r}; choose {sorted(AUTOSCALE_PROFILES)}")
    resolved = handles or runtime_handles()
    if _active_autoscale_profile == profile:
        return resolved

    rembg_worker, sam_worker, mesh_worker, lite_worker, finalizer = resolved
    common = {"min_containers": 0, "max_containers": 1, "buffer_containers": 0}
    autoscalers = {
        "rembg": rembg_worker,
        "sam3d": sam_worker,
        "mesh": mesh_worker,
        "lite": lite_worker,
        "finalize": finalizer,
    }
    for stage, target in autoscalers.items():
        target.update_autoscaler(scaledown_window=cfg[stage], **common)
    _active_autoscale_profile = profile
    return resolved


async def apply_autoscale_profile_aio(profile: str, handles=None) -> tuple:
    """Async ASGI variant; updates the same pool only when the profile changes."""
    global _active_autoscale_profile

    import asyncio

    cfg=AUTOSCALE_PROFILES.get(profile)
    if cfg is None:
        raise ValueError(f"unknown autoscale profile {profile!r}; choose {sorted(AUTOSCALE_PROFILES)}")
    resolved=handles or runtime_handles()
    if profile == _active_autoscale_profile:
        return resolved

    rembg_worker,sam_worker,mesh_worker,lite_worker,finalizer=resolved
    common={"min_containers":0,"max_containers":1,"buffer_containers":0}
    autoscalers={
        "rembg":rembg_worker,
        "sam3d":sam_worker,
        "mesh":mesh_worker,
        "lite":lite_worker,
        "finalize":finalizer,
    }
    await asyncio.gather(*(
        target.update_autoscaler.aio(scaledown_window=cfg[stage],**common)
        for stage,target in autoscalers.items()
    ))
    _active_autoscale_profile=profile
    return resolved


app = modal.App(APP_NAME)
weights = modal.Volume.from_name("modal-3d-embodiedgen-weights", create_if_missing=True)
artifacts = modal.Volume.from_name("modal-3d-artifacts", create_if_missing=True)
state_handoff = modal.Dict.from_name("modal-3d-embodiedgen-state", create_if_missing=True)
traffic_events = modal.Dict.from_name("modal-3d-embodiedgen-traffic", create_if_missing=True)
job_states = modal.Dict.from_name("modal-3d-embodiedgen-jobs", create_if_missing=True)
control_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi==0.116.1", "pillow==11.3.0"
)
text_weight_image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"HF_HOME": "/weights/hf", "PYTHONUNBUFFERED": "1"})
    .pip_install("huggingface_hub==0.34.4")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"PYTHONUNBUFFERED": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    .apt_install("libgomp1")
    .pip_install(
        "numpy==1.26.4",
        "pillow==11.3.0",
        "rembg==2.0.61",
        "onnxruntime==1.20.1",
        "xatlas==0.0.11",
        "fast-simplification==0.2.0",
        "trimesh==5.0.0",
    )
)

ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "HF_HOME": "/weights/hf",
    "MODELSCOPE_CACHE": "/weights/modelscope",
    "TORCH_HOME": "/weights/torch",
    "U2NET_HOME": "/weights/u2net",
    "PYOPENGL_PLATFORM": "egl",
    "TORCH_CUDA_ARCH_LIST": "8.9",
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-runtime-ubuntu22.04", add_python="3.10")
    .env(ENV)
    .apt_install(
        "git", "curl", "unzip", "ffmpeg",
        "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libegl1", "libegl1-mesa", "libgomp1", "libx11-6", "libxrandr2", "libxi6",
    )
    .run_commands(
        "! command -v nvcc",  # hard invariant: consumer cannot compile CUDA
        "git init /workspace/EmbodiedGen && cd /workspace/EmbodiedGen && git remote add origin https://github.com/HorizonRobotics/EmbodiedGen.git",
        f"cd /workspace/EmbodiedGen && git fetch --depth 1 origin {EMBODIEDGEN_COMMIT} && git checkout --detach FETCH_HEAD",
        "cd /workspace/EmbodiedGen && git submodule update --init --recursive --progress thirdparty/sam3d",
        "cd /workspace/EmbodiedGen && git submodule update --init --recursive --depth 1 thirdparty/TRELLIS",
    )
    .run_commands(
        "python -m pip install --upgrade 'pip>=25' setuptools==80.10.2 wheel packaging 'Cython>=0.29.37'",
        "python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126",
        "python -m pip install xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu126",
        "printf 'numpy==1.26.4\\nopencv-python==4.11.0.86\\nopencv-python-headless==4.11.0.86\\npillow<12\\n' >/tmp/eg-constraints.txt",
        "cd /workspace/EmbodiedGen && PIP_CONSTRAINT=/tmp/eg-constraints.txt python -m pip install -r requirements.txt --use-deprecated=legacy-resolver",
    )
    .run_commands(
        "python -m pip install --no-deps 'utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15'",
        f"python -m pip install --no-deps 'clip@git+https://github.com/openai/CLIP.git@{CLIP_COMMIT}'",
        "python -m pip install --no-deps 'segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f'",
        f"python -m pip install --no-deps 'kolors@git+https://github.com/HochCC/Kolors.git@{KOLORS_COMMIT}'",
        "python -m pip install --no-deps 'MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734'",
        "PIP_CONSTRAINT=/tmp/eg-constraints.txt python -m pip install plyfile moderngl glcontext ftfy fvcore iopath",
        "python -m pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 'pillow<12'",
        "python -m pip install --no-deps kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html",
        "python -m pip install pygltflib warp-lang usd-core ipycanvas ipyevents 'jupyter_client<8' tornado",
        "python -m pip install --no-deps gsplat==1.5.3",
        "python -m pip install --no-deps fast-simplification==0.2.0",
    )
    # Consume release artifacts: no source builds.
    .run_commands(
        f"mkdir -p /opt/embodiedgen-release/wheels /root/.cache/torch_extensions && curl -fL '{RELEASE}/{TAG}.wheels.zip' -o /tmp/wheels.zip",
        f"curl -fL '{RELEASE}/{TAG}.torch-extensions.zip' -o /tmp/ext.zip",
        f"echo '{RELEASE_WHEELS_SHA256}  /tmp/wheels.zip' | sha256sum -c -",
        f"echo '{RELEASE_EXTENSIONS_SHA256}  /tmp/ext.zip' | sha256sum -c -",
        "unzip -q /tmp/wheels.zip -d /opt/embodiedgen-release/wheels",
        "unzip -q /tmp/ext.zip -d /root/.cache/torch_extensions",
        "python -m pip install --no-deps /opt/embodiedgen-release/wheels/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl /opt/embodiedgen-release/wheels/nvdiffrast-0.3.3-py3-none-any.whl",
        "rm -f /tmp/wheels.zip /tmp/ext.zip",
    )
    .workdir("/workspace/EmbodiedGen")
)

# Apply only the validated headless/source patches after all packages are installed.
image = (
    image
    .add_local_file("patches/embodiedgen-v2.0.0/production/headless-l40s.patch", "/tmp/headless-l40s.patch", copy=True)
    .add_local_file(
        "patches/embodiedgen-v2.0.0/production/retexture-lazy-delight.patch",
        "/tmp/retexture-lazy-delight.patch",
        copy=True,
    )
    .run_commands(
        "cd /workspace/EmbodiedGen && git apply /tmp/headless-l40s.patch && git apply /tmp/retexture-lazy-delight.patch",
        "cd /workspace/EmbodiedGen && grep -RIl '@spaces.GPU' embodied_gen --include='*.py' | xargs -r sed -i '/^[[:space:]]*@spaces.GPU[[:space:]]*$/d'",
        "cd /workspace/EmbodiedGen && python -m pip install --no-deps -e .",
        "cd /workspace/EmbodiedGen && python -m py_compile embodied_gen/scripts/imageto3d.py embodied_gen/data/backproject_v3.py embodied_gen/models/gs_model.py",
        "! command -v nvcc",
    )
)

# Single release-loader implementation: direct-load the validated .so files and
# leave no torch cpp_extension/JIT fallback in the production consumer.
image = (
    image
    .add_local_file(
        "patches/embodiedgen-v2.0.0/production/patch_nvdiffrast_init_release.py",
        "/tmp/patch_nvdiffrast_init_release.py",
        copy=True,
    )
    .add_local_file(
        "patches/embodiedgen-v2.0.0/production/gsplat_backend_release.py",
        "/usr/local/lib/python3.10/site-packages/gsplat/cuda/_backend.py",
        copy=True,
    )
    .run_commands(
        "python /tmp/patch_nvdiffrast_init_release.py",
        "rm -rf /usr/local/lib/python3.10/site-packages/nvdiffrast/torch/__pycache__ /usr/local/lib/python3.10/site-packages/gsplat/cuda/__pycache__",
        "grep -q 'modal-build release-only loader' /usr/local/lib/python3.10/site-packages/nvdiffrast/torch/__init__.py",
        "grep -q 'Release-only gsplat CUDA backend' /usr/local/lib/python3.10/site-packages/gsplat/cuda/_backend.py",
        "! command -v nvcc",
    )
)



def _weights_info() -> dict:
    target = Path("/weights/sam-3d-objects")
    marker = target / "checkpoints/pipeline.yaml"
    if not marker.exists():
        raise RuntimeError("SAM3D weights missing; run preload_weights first")
    return {
        "path": str(target),
        "size": subprocess.check_output(["du", "-sh", str(target)], text=True).split()[0],
    }


@app.function(
    image=image,
    volumes={"/weights": weights},
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
)
def preload_weights():
    """CPU-only model/cache pull for a fresh Modal workspace."""
    os.environ.update({"TORCH_HOME": "/weights/torch", "U2NET_HOME": "/weights/u2net"})
    t0 = time.perf_counter()
    u2net = Path("/weights/u2net/u2net.onnx")
    if not u2net.exists():
        import urllib.request
        u2net.parent.mkdir(parents=True, exist_ok=True)
        print("CPU ONLY: downloading U2Net", flush=True)
        urllib.request.urlretrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            str(u2net),
        )

    example = Path("/weights/examples/sample_00.jpg")
    if not example.exists():
        import shutil

        example.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2("/workspace/EmbodiedGen/apps/assets/example_image/sample_00.jpg", example)

    dino = Path("/weights/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
    dino_repo = Path("/weights/torch/hub/facebookresearch_dinov2_main")
    if not dino.exists() or not dino_repo.exists():
        print("CPU ONLY: downloading DINOv2 repo + ViT-L/14 reg4", flush=True)
        import torch
        m = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14_reg",
            pretrained=True,
            trust_repo=True,
        )
        del m

    target = Path("/weights/sam-3d-objects")
    marker = target / "checkpoints/pipeline.yaml"
    if not marker.exists():
        print("CPU ONLY: downloading SAM3D weights", flush=True)
        from modelscope import snapshot_download
        snapshot_download("facebook/sam-3d-objects", local_dir=str(target))
    weights.commit()
    info = _weights_info()
    info["seconds"] = round(time.perf_counter() - t0, 3)
    print("WEIGHTS_READY", json.dumps(info), flush=True)
    return info


@app.function(
    image=text_weight_image,
    volumes={"/weights": weights},
    timeout=60 * 60,
    cpu=2.0,
    memory=4096,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
)
def preload_text2img_weights() -> dict:
    """CPU-only pull of the exact public Kolors snapshot used by Text→3D."""
    from huggingface_hub import snapshot_download

    t0=time.perf_counter()
    target=Path(TEXT2IMG_MODEL_DIR)
    model_index=target/"model_index.json"
    revision_marker=target/TEXT2IMG_REVISION_MARKER
    current_revision=revision_marker.read_text().strip() if revision_marker.exists() else None
    if not model_index.exists() or current_revision != TEXT2IMG_MODEL_REVISION:
        target.mkdir(parents=True,exist_ok=True)
        print(
            f"CPU ONLY: syncing {TEXT2IMG_MODEL_ID}@{TEXT2IMG_MODEL_REVISION}",
            flush=True,
        )
        snapshot_download(
            repo_id=TEXT2IMG_MODEL_ID,
            revision=TEXT2IMG_MODEL_REVISION,
            local_dir=str(target),
        )
        revision_marker.write_text(TEXT2IMG_MODEL_REVISION+"\n")
    weights.commit()
    if not model_index.exists() or revision_marker.read_text().strip() != TEXT2IMG_MODEL_REVISION:
        raise RuntimeError("text2img weights/revision marker incomplete")
    info={
        "model":TEXT2IMG_MODEL_ID,
        "revision":TEXT2IMG_MODEL_REVISION,
        "path":str(target),
        "size_gib":round(sum(p.stat().st_size for p in target.rglob("*") if p.is_file())/1024**3,3),
        "seconds":round(time.perf_counter()-t0,3),
    }
    print("TEXT2IMG_WEIGHTS_READY",json.dumps(info),flush=True)
    return info


@app.function(
    image=text_weight_image,
    volumes={"/weights": weights},
    timeout=30 * 60,
    cpu=2.0,
    memory=4096,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
)
def preload_retexture_weights() -> dict:
    """CPU-only pull of pinned ControlNet/SR weights for texture generation."""
    import shutil

    from huggingface_hub import snapshot_download

    t0=time.perf_counter()
    target=Path(RETEXTURE_MODEL_DIR)
    marker=target/RETEXTURE_REVISION_MARKER
    current=marker.read_text().strip() if marker.exists() else None
    controlnet=Path(RETEXTURE_CONTROLNET_DIR)/"diffusion_pytorch_model.safetensors"
    sr=Path(RETEXTURE_SR_PATH)
    if current != RETEXTURE_MODEL_REVISION or not controlnet.exists() or not sr.exists():
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True,exist_ok=True)
        print(
            f"CPU ONLY: syncing {RETEXTURE_MODEL_ID}@{RETEXTURE_MODEL_REVISION}",
            flush=True,
        )
        snapshot_download(
            repo_id=RETEXTURE_MODEL_ID,
            revision=RETEXTURE_MODEL_REVISION,
            allow_patterns=["texture_gen_mv_v1/*","super_resolution/*"],
            local_dir=str(target),
        )
        marker.write_text(RETEXTURE_MODEL_REVISION+"\n")
    weights.commit()
    if marker.read_text().strip() != RETEXTURE_MODEL_REVISION:
        raise RuntimeError("retexture revision marker mismatch")
    if not controlnet.exists() or not sr.exists():
        raise RuntimeError("retexture weights incomplete")
    info={
        "model":RETEXTURE_MODEL_ID,
        "revision":RETEXTURE_MODEL_REVISION,
        "path":str(target),
        "size":subprocess.check_output(["du","-sh",str(target)],text=True).split()[0],
        "seconds":round(time.perf_counter()-t0,3),
    }
    print("RETEXTURE_WEIGHTS_READY",json.dumps(info),flush=True)
    return info


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/weights":weights,"/artifacts":artifacts},
    cpu=4.0,
    memory=32768,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=TEXT2IMG_SCALEDOWN_WINDOWS[STATIC_AUTOSCALE_PROFILE],
    timeout=30*60,
)
class Text2ImageWorker:
    """Pinned public Kolors Text→Image stage feeding the existing Image→3D pipeline."""

    @modal.enter()
    def load(self):
        import torch
        from diffusers import DPMSolverMultistepScheduler, KolorsPipeline

        t0=time.perf_counter()
        model_dir=Path(TEXT2IMG_MODEL_DIR)
        model_index=model_dir/"model_index.json"
        revision_marker=model_dir/TEXT2IMG_REVISION_MARKER
        current_revision=revision_marker.read_text().strip() if revision_marker.exists() else None
        if not model_index.exists() or current_revision != TEXT2IMG_MODEL_REVISION:
            raise RuntimeError("Kolors weights/revision mismatch; run preload_text2img_weights first")
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        pipe=KolorsPipeline.from_pretrained(
            TEXT2IMG_MODEL_DIR,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        ).to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler=DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        self.pipe=pipe
        self.load_seconds=time.perf_counter()-t0
        print(
            "TEXT2IMG_RESIDENT_READY "+json.dumps({
                "model":TEXT2IMG_MODEL_ID,
                "revision":TEXT2IMG_MODEL_REVISION,
                "load_seconds":round(self.load_seconds,3),
            }),
            flush=True,
        )

    @modal.method()
    def generate(self,job_id: str,prompt: str,seed: int=0) -> dict:
        import torch
        from embodied_gen.models.text_model import PROMPT_APPEND

        prompt=normalize_text_prompt(prompt)
        if not is_api_job_id(job_id):
            raise ValueError(f"invalid API job id: {job_id!r}")
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed <= 100000:
            raise ValueError("seed must be an integer in 0..100000")
        artifacts.reload()
        root=api_job_root(job_id)
        root.mkdir(parents=True,exist_ok=True)
        full_prompt=PROMPT_APPEND.format(object=prompt)
        generator=torch.Generator().manual_seed(seed)
        t0=time.perf_counter()
        image_out=self.pipe(
            prompt=full_prompt,
            height=1024,
            width=1024,
            num_inference_steps=25,
            guidance_scale=7.0,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]
        output=root/"input_image"
        image_out.save(output,format="PNG")
        (root/"prompt.txt").write_text(prompt+"\n")
        artifacts.commit()
        out={
            "job_id":job_id,
            "model":"kolors",
            "model_revision":TEXT2IMG_MODEL_REVISION,
            "seed":seed,
            "width":image_out.width,
            "height":image_out.height,
            "load_seconds":round(self.load_seconds,3),
            "generate_seconds":round(time.perf_counter()-t0,3),
        }
        print("TEXT2IMG_OK",json.dumps(out),flush=True)
        return out


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/weights":weights,"/artifacts":artifacts},
    cpu=4.0,
    memory=32768,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=RETEXTURE_SCALEDOWN_WINDOWS[STATIC_AUTOSCALE_PROFILE],
    timeout=30*60,
)
class RetextureWorker:
    """Prompt-driven texture edit for an existing validated asset; geometry stays fixed."""

    @modal.enter()
    def load(self):
        import torch
        from diffusers import KolorsPipeline
        from kolors.models.controlnet import ControlNetModel
        from kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import (
            StableDiffusionXLControlNetImg2ImgPipeline,
        )

        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        base=Path(TEXT2IMG_MODEL_DIR)
        ret=Path(RETEXTURE_MODEL_DIR)
        if (base/TEXT2IMG_REVISION_MARKER).read_text().strip() != TEXT2IMG_MODEL_REVISION:
            raise RuntimeError("Kolors weights/revision mismatch; run preload_text2img_weights first")
        if (ret/RETEXTURE_REVISION_MARKER).read_text().strip() != RETEXTURE_MODEL_REVISION:
            raise RuntimeError("retexture weights/revision mismatch; run preload_retexture_weights first")
        t0=time.perf_counter()
        base_pipe=KolorsPipeline.from_pretrained(
            TEXT2IMG_MODEL_DIR,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        )
        controlnet=ControlNetModel.from_pretrained(
            RETEXTURE_CONTROLNET_DIR,
            use_safetensors=True,
            local_files_only=True,
        ).half()
        pipe=StableDiffusionXLControlNetImg2ImgPipeline(
            vae=base_pipe.vae,
            controlnet=controlnet,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            image_encoder=None,
            feature_extractor=None,
            force_zeros_for_empty_prompt=False,
        ).to("cuda")
        del base_pipe
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        self.pipe=pipe
        self.load_seconds=time.perf_counter()-t0
        print(
            "RETEXTURE_RESIDENT_READY "+json.dumps({
                "gpu":torch.cuda.get_device_name(0),
                "load_seconds":round(self.load_seconds,3),
                "base_revision":TEXT2IMG_MODEL_REVISION,
                "controlnet_revision":RETEXTURE_MODEL_REVISION,
            }),
            flush=True,
        )

    @modal.method()
    def generate(self,job_id: str,source_job_id: str,prompt: str,seed: int=0) -> dict:
        import shutil
        import xml.etree.ElementTree as ET

        import numpy as np
        import trimesh
        from embodied_gen.data.backproject_v2 import entrypoint as backproject_api
        from embodied_gen.data.differentiable_render import entrypoint as drender_api
        from embodied_gen.models.sr_model import ImageRealESRGAN
        from embodied_gen.scripts.render_mv import infer_pipe as render_mv_api

        prompt=normalize_text_prompt(prompt)
        if not is_api_job_id(job_id) or not is_api_job_id(source_job_id):
            raise ValueError("invalid API job id")
        if job_id == source_job_id:
            raise ValueError("retexture job must differ from source job")
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed <= 100000:
            raise ValueError("seed must be an integer in 0..100000")

        artifacts.reload()
        source_root=api_job_root(source_job_id)
        source_result=source_root/"result"
        source_obj=source_result/"mesh"/"sample_00.obj"
        source_urdf=source_result/"sample_00.urdf"
        source_gs=source_result/"mesh"/"sample_00_gs.ply"
        source_gs_aligned=source_result/"mesh"/"sample_00_gs_aligned.ply"
        for required in (source_obj,source_urdf,source_gs,source_gs_aligned):
            if not required.is_file():
                raise FileNotFoundError(f"source asset incomplete: {required}")

        root=api_job_root(job_id)
        work=root/"retexture"
        if work.exists():
            shutil.rmtree(work)
        condition=work/"condition"
        multi_view=work/"multi_view"
        texture_mesh=work/"texture_mesh"
        preview=work/"preview"
        texture_mesh.mkdir(parents=True,exist_ok=True)
        out_obj=texture_mesh/"sample_00.obj"
        out_glb=texture_mesh/"sample_00.glb"

        total0=time.perf_counter()
        t0=time.perf_counter()
        drender_api(
            mesh_path=str(source_obj),
            output_root=str(condition),
            uuid="sample_00",
            with_mtl=True,
        )
        condition_seconds=time.perf_counter()-t0

        t0=time.perf_counter()
        render_mv_api(
            index_file=str(condition/"index.json"),
            controlnet_cond_scale=0.7,
            guidance_scale=9.0,
            strength=0.9,
            num_inference_steps=40,
            ip_adapt_scale=0.0,
            ip_img_path=None,
            prompt=prompt,
            save_dir=str(multi_view),
            sub_idxs=[[0,1,2],[3,4,5]],
            pipeline=self.pipe,
            seed=seed,
        )
        diffusion_seconds=time.perf_counter()-t0

        t0=time.perf_counter()
        imagesr=ImageRealESRGAN(outscale=4,model_path=RETEXTURE_SR_PATH)
        backproject_api(
            delight_model=None,
            imagesr_model=imagesr,
            mesh_path=str(source_obj),
            color_path=str(multi_view/"color_sample0.png"),
            output_path=str(out_obj),
            save_glb_path=str(out_glb),
            skip_fix_mesh=True,
            delight=False,
            no_save_delight_img=True,
            texture_wh=[2048,2048],
            no_mesh_post_process=True,
        )
        backproject_seconds=time.perf_counter()-t0

        t0=time.perf_counter()
        drender_api(
            mesh_path=str(out_obj),
            output_root=str(preview),
            uuid="sample_00",
            num_images=90,
            elevation=[20],
            with_mtl=True,
            gen_color_mp4=True,
            pbr_light_factor=1.2,
        )
        preview_seconds=time.perf_counter()-t0
        preview_mp4=preview/"sample_00"/"color.mp4"
        if not preview_mp4.is_file():
            raise FileNotFoundError(f"missing retexture preview: {preview_mp4}")

        result=root/"result"
        meshdir=result/"mesh"
        if result.exists():
            shutil.rmtree(result)
        meshdir.mkdir(parents=True,exist_ok=True)
        copy_obj_bundle(out_obj,meshdir)
        shutil.copy2(out_glb,meshdir/"sample_00.glb")
        shutil.copy2(source_gs,meshdir/"sample_00_gs.ply")
        shutil.copy2(source_gs_aligned,meshdir/"sample_00_gs_aligned.ply")
        shutil.copy2(source_urdf,result/"sample_00.urdf")
        shutil.copy2(preview_mp4,result/"video.mp4")
        texture_candidates=[
            path for path in obj_material_dependencies(out_obj)
            if path.suffix.lower() in {".png",".jpg",".jpeg",".webp"}
        ]
        if texture_candidates:
            shutil.copy2(texture_candidates[0],meshdir/"texture.png")

        result_obj=meshdir/"sample_00.obj"
        result_glb=meshdir/"sample_00.glb"
        src_mesh=trimesh.load(source_obj,force="mesh")
        objm=trimesh.load(result_obj,force="mesh")
        glbs=trimesh.load(result_glb,force="scene")
        ET.parse(result/"sample_00.urdf")
        with (meshdir/"sample_00_gs.ply").open("rb") as f:
            header=f.read(8192).decode("ascii","ignore")
        ply_vertices=next(int(x.split()[-1]) for x in header.splitlines() if x.startswith("element vertex "))
        geometry_preserved=(
            len(src_mesh.faces)==len(objm.faces)
            and np.allclose(src_mesh.bounds,objm.bounds,rtol=1e-5,atol=1e-6)
        )
        checks={
            "ply_vertices":ply_vertices,
            "obj_vertices":len(objm.vertices),
            "obj_faces":len(objm.faces),
            "glb_geometries":len(glbs.geometry),
            "urdf_mesh_exists":result_obj.exists(),
            "video_exists":(result/"video.mp4").exists(),
            "obj_material_missing":missing_obj_material_dependencies(result_obj),
            "source_geometry_preserved":bool(geometry_preserved),
            "source_obj_faces":len(src_mesh.faces),
        }
        checks["obj_material_refs_ok"]=not checks["obj_material_missing"]
        if not validation_passes(checks) or not checks["source_geometry_preserved"]:
            raise RuntimeError(checks)
        report={
            "job_id":job_id,
            "source_job_id":source_job_id,
            "kind":"retexture",
            "checks":checks,
            "timings":{
                "resident_model_load_seconds":round(self.load_seconds,3),
                "condition_render_seconds":round(condition_seconds,3),
                "diffusion_seconds":round(diffusion_seconds,3),
                "backproject_seconds":round(backproject_seconds,3),
                "preview_seconds":round(preview_seconds,3),
                "method_seconds":round(time.perf_counter()-total0,3),
            },
        }
        report["cleaned_intermediates"]=prune_job_intermediates(root)
        (root/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
        artifacts.commit()
        print("RETEXTURE_VALIDATION_OK",json.dumps(report),flush=True)
        return report


def _rembg_load(worker, cpu_label: str) -> None:
    import uuid

    import rembg

    t0=time.perf_counter()
    os.environ.update({"U2NET_HOME":"/weights/u2net", "TORCH_HOME":"/weights/torch"})
    worker.session=rembg.new_session("u2net", providers=["CPUExecutionProvider"])
    worker.session_load_seconds=time.perf_counter()-t0
    worker.instance_id=uuid.uuid4().hex
    worker.cpu_label=cpu_label
    print(
        f"REMBG_RESIDENT_READY cpu={cpu_label} instance_id={worker.instance_id} "
        f"load_seconds={worker.session_load_seconds:.3f}",
        flush=True,
    )


def _rembg_prepare(worker, job_id: str) -> dict:
    import rembg
    from PIL import Image

    t0=time.perf_counter()
    artifacts.reload()
    root=Path("/artifacts/embodiedgen/jobs")/job_id
    root.mkdir(parents=True,exist_ok=True)
    src,source=resolve_job_input(
        job_id,
        root,
        Path("/weights/examples/sample_00.jpg"),
    )
    raw=root/"sample_00_raw.png"
    cond=root/"sample_00_cond.png"
    image_in=Image.open(src)
    image_in.load()
    image_in.save(raw)
    current_max=max(image_in.size)
    scale=min(1.0,1024.0/current_max)
    if scale < 1.0:
        image_in=image_in.resize(
            (int(image_in.width*scale),int(image_in.height*scale)),
            Image.Resampling.LANCZOS,
        )
    r0=time.perf_counter()
    rembg.remove(image_in,session=worker.session).save(cond)
    r1=time.perf_counter()
    artifacts.commit()
    out={
        "job_id":job_id,
        "raw":str(raw),
        "cond":str(cond),
        "source":source,
        "cpu":worker.cpu_label,
        "instance_id":worker.instance_id,
        "session_load_seconds":round(worker.session_load_seconds,3),
        "remove_seconds":round(r1-r0,3),
        "method_seconds":round(time.perf_counter()-t0,3),
    }
    print("REMBG_PREPARE_OK",json.dumps(out),flush=True)
    return out


@app.cls(
    image=cpu_image,
    volumes={"/weights": weights, "/artifacts": artifacts},
    cpu=1.0,
    memory=4096,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=AUTOSCALE_PROFILES[STATIC_AUTOSCALE_PROFILE]["rembg"],
    timeout=10 * 60,
)
class RembgWorker:
    """Production rembg worker: 1 CPU + 4 GiB; static default is COST_FIRST."""

    @modal.enter()
    def load(self):
        _rembg_load(self,"1cpu-4g")

    @modal.method()
    def prepare(self, job_id: str) -> dict:
        return _rembg_prepare(self,job_id)


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/weights": weights, "/artifacts": artifacts},
    cpu=6.0,
    memory=32768,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=AUTOSCALE_PROFILES[STATIC_AUTOSCALE_PROFILE]["sam3d"],
    timeout=30 * 60,
)
class Sam3DWorker:
    """Heavy L40S worker: resident SAM3D; static default is COST_FIRST."""

    @modal.enter()
    def load(self):
        t0=time.perf_counter()
        os.chdir("/workspace/EmbodiedGen")
        os.environ.update({"TORCH_HOME":"/weights/torch", "U2NET_HOME":"/weights/u2net"})
        if subprocess.run(
            ["bash", "-lc", "command -v nvcc"], capture_output=True, check=False
        ).returncode == 0:
            raise RuntimeError("release-consumer invariant violated: nvcc is present")
        import uuid

        import torch
        from embodied_gen.models.sam3d import Sam3dInference
        self.torch=torch
        self.pipeline=Sam3dInference(local_dir="/weights/sam-3d-objects")
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-t0
        self.instance_id=uuid.uuid4().hex
        print(f"SAM3D_RESIDENT_READY instance_id={self.instance_id} load_seconds={self.load_seconds:.3f}",flush=True)

    @modal.method()
    def generate(self, job_id: str, seed: int = 0) -> dict:
        import pickle

        import numpy as np
        from embodied_gen.utils.trender import pack_state
        from PIL import Image

        t0=time.perf_counter()
        artifacts.reload()
        root=Path("/artifacts/embodiedgen/jobs")/job_id
        cond=root/"sample_00_cond.png"
        if not cond.exists(): raise FileNotFoundError(cond)
        image=Image.open(cond).convert("RGBA")
        i0=time.perf_counter()
        outputs=self.pipeline.run(
            image,
            seed=seed,
            stage1_inference_steps=SAM3D_STAGE1_STEPS,
            stage2_inference_steps=SAM3D_STAGE2_STEPS,
        )
        self.torch.cuda.synchronize()
        i1=time.perf_counter()
        gs=outputs["gaussian"][0]; mesh=outputs["mesh"][0]
        p0=time.perf_counter()
        state=pack_state(gs,mesh)
        # Mesh indices are far below int32 limits (~450k vertices). This is lossless
        # and removes ~10 MiB from every cross-stage state transfer.
        state["mesh"]["faces"]=state["mesh"]["faces"].astype(np.int32,copy=False)
        p1=time.perf_counter()

        # Serialize once in-memory, then hand off through Modal Dict. Values above
        # 2 MiB use Modal's blob transport automatically. This prevents the heavy
        # L40S from waiting on a Volume commit (observed 1.4-6.3s variance).
        s0=time.perf_counter()
        state_payload=pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL)
        s1=time.perf_counter()
        h0=time.perf_counter()
        state_handoff.put(job_id,state_payload)
        h1=time.perf_counter()
        state_bytes=len(state_payload)

        del outputs,gs,mesh,state,state_payload
        self.torch.cuda.empty_cache()
        result={
            "job_id":job_id,
            "instance_id":self.instance_id,
            "resident_model_load_seconds":round(self.load_seconds,3),
            "stage1_steps":SAM3D_STAGE1_STEPS,
            "stage2_steps":SAM3D_STAGE2_STEPS,
            "inference_seconds":round(i1-i0,3),
            "pack_state_seconds":round(p1-p0,3),
            "serialize_seconds":round(s1-s0,3),
            "state_handoff_put_seconds":round(h1-h0,3),
            "state_mib":round(state_bytes/1024/1024,3),
            "method_seconds":round(time.perf_counter()-t0,3),
            "gpu":self.torch.cuda.get_device_name(0),
        }
        print("SAM3D_GENERATE_OK",json.dumps(result),flush=True)
        return result


@app.cls(
    image=cpu_image,
    volumes={"/artifacts": artifacts},
    cpu=4.0,
    memory=8192,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=AUTOSCALE_PROFILES[STATIC_AUTOSCALE_PROFILE]["mesh"],
    timeout=15 * 60,
)
class MeshWorker:
    """Persistent CPU mesh worker: fast-simplification -> 50k -> xatlas."""

    @modal.enter()
    def load(self):
        import uuid
        t0=time.perf_counter()
        # Warm imports once per container. The C++ modules stay resident.
        import fast_simplification  # noqa: F401
        import numpy  # noqa: F401
        import xatlas  # noqa: F401
        self.instance_id=uuid.uuid4().hex
        self.load_seconds=time.perf_counter()-t0
        print(
            f"MESH_RESIDENT_READY instance_id={self.instance_id} "
            f"load_seconds={self.load_seconds:.3f}",
            flush=True,
        )

    @modal.method()
    def process(self, job_id: str) -> dict:
        import pickle

        import fast_simplification
        import numpy as np
        import xatlas

        t0=time.perf_counter()
        artifacts.reload()
        root=Path("/artifacts/embodiedgen/jobs")/job_id
        root.mkdir(parents=True,exist_ok=True)
        # The heavy GPU only puts the serialized state into transient Dict storage.
        # CPU owns durable Volume persistence and all following mesh/PLY work.
        g0=time.perf_counter(); payload=state_handoff.get(job_id); g1=time.perf_counter()
        handoff_source="dict"
        persist0=time.perf_counter()
        if payload is None:
            # Retry/debug safety: if CPU already persisted the state in a previous attempt,
            # reuse it instead of forcing another expensive SAM3D generation.
            state_path=root/"sample_00_state.pkl"
            if not state_path.exists():
                raise RuntimeError(f"missing transient state handoff for {job_id}")
            payload=state_path.read_bytes()
            handoff_source="volume-fallback"
        else:
            with (root/"sample_00_state.pkl").open("wb") as f:
                f.write(payload)
        state=pickle.loads(payload)
        persist1=time.perf_counter()
        # Rebuild the raw and aligned Gaussian PLYs entirely on CPU from state.
        # This reproduces the fields consumed by GaussianOperator.load_from_ply()
        # without importing torch/gsplat or allocating any GPU.
        g=state["gaussian"]
        pg0=time.perf_counter()
        aabb=np.asarray(g["aabb"],dtype=np.float32)
        means=np.asarray(g["_xyz"],dtype=np.float32)*aabb[3:]+aabb[:3]
        fdc=np.asarray(g["_features_dc"],dtype=np.float32).transpose(0,2,1).reshape(len(means),-1)
        opacity_bias=np.float32(g["opacity_bias"])
        logit_bias=np.log(opacity_bias/(np.float32(1.0)-opacity_bias)).astype(np.float32)
        opacities=np.asarray(g["_opacity"],dtype=np.float32).reshape(-1)+logit_bias
        hidden_scale=np.asarray(g["_scaling"],dtype=np.float32)
        scale_bias=np.float32(g["scaling_bias"])
        activation=g["scaling_activation"]
        if activation == "softplus":
            inv_bias=scale_bias+np.log(-np.expm1(-scale_bias))
            active_scale=np.logaddexp(np.float32(0.0),hidden_scale+inv_bias)
        elif activation == "exp":
            active_scale=np.exp(hidden_scale+np.log(scale_bias))
        else:
            raise RuntimeError(f"unsupported Gaussian scaling activation: {activation}")
        active_scale=np.sqrt(active_scale*active_scale+np.float32(g["mininum_kernel_size"])**2)
        log_scales=np.log(active_scale).astype(np.float32)
        raw_quats=np.asarray(g["_rotation"],dtype=np.float32)+np.asarray([1,0,0,0],dtype=np.float32)

        def write_ply(path, xyz, quats):
            fields=["x","y","z"]+[f"f_dc_{i}" for i in range(fdc.shape[1])]+["opacity"]+[f"scale_{i}" for i in range(3)]+[f"rot_{i}" for i in range(4)]
            dtype=np.dtype([(name,"<f4") for name in fields])
            out=np.empty(len(xyz),dtype=dtype)
            out["x"],out["y"],out["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
            for i in range(fdc.shape[1]): out[f"f_dc_{i}"]=fdc[:,i]
            out["opacity"]=opacities
            for i in range(3): out[f"scale_{i}"]=log_scales[:,i]
            for i in range(4): out[f"rot_{i}"]=quats[:,i]
            with open(path,"wb") as f:
                f.write(b"ply\nformat binary_little_endian 1.0\n")
                f.write(f"element vertex {len(xyz)}\n".encode())
                f.writelines(f"property float {name}\n".encode() for name in fields)
                f.write(b"end_header\n")
                out.tofile(f)

        write_ply(root/"sample_00_gs.ply",means,raw_quats)
        # Same fixed alignment formerly done by GaussianOperator.resave_ply().
        align_rot=np.asarray([[0,0,-1],[0,-1,0],[-1,0,0]],dtype=np.float32)
        aligned_means=means@align_rot.T
        q=raw_quats/np.linalg.norm(raw_quats,axis=1,keepdims=True)
        qi=np.asarray([0.0,0.7071067811865476,0.0,-0.7071067811865476],dtype=np.float32)  # wxyz
        v1=qi[1:]; w1=qi[0]; w2=q[:,0]; v2=q[:,1:]
        aligned_q=np.empty_like(q)
        aligned_q[:,0]=w1*w2-np.sum(v2*v1,axis=1)
        aligned_q[:,1:]=w1*v2+w2[:,None]*v1+np.cross(np.broadcast_to(v1,v2.shape),v2)
        write_ply(root/"sample_00_gs_aligned.ply",aligned_means,aligned_q)
        pg1=time.perf_counter()

        vertices=np.asarray(state["mesh"]["vertices"],dtype=np.float32)
        faces=np.asarray(state["mesh"]["faces"],dtype=np.int32)
        input_vertices,input_faces=len(vertices),len(faces)

        mesh_add_rot=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=np.float32)
        rot_matrix=np.array([[0,0,-1],[0,1,0],[1,0,0]],dtype=np.float32)
        vertices=vertices @ mesh_add_rot @ rot_matrix

        # Critical path deliberately does not export the high-poly raw OBJ.
        # Small meshes must bypass simplification: fast-simplification rejects
        # target_count >= current face count.
        simplify0=time.perf_counter()
        vertices,faces,was_simplified=simplify_mesh_if_needed(
            vertices,
            faces,
            fast_simplification.simplify,
        )
        simplify1=time.perf_counter()
        vertices=np.asarray(vertices,dtype=np.float32)
        faces=np.asarray(faces,dtype=np.int32)

        bbmin=vertices.min(0); bbmax=vertices.max(0)
        extent=float((bbmax-bbmin).max())
        if not np.isfinite(extent) or extent <= 0.0:
            raise RuntimeError(f"invalid mesh extent for {job_id}: {extent}")
        center=(bbmin+bbmax)*0.5
        scale=np.float32(2.0/extent)
        norm=(vertices-center)*scale
        x_rot=np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=np.float32)
        z_rot=np.array([[0,1,0],[-1,0,0],[0,0,1]],dtype=np.float32)
        norm=norm @ x_rot @ z_rot

        x0=time.perf_counter()
        vmapping,indices,uvs=xatlas.parametrize(norm,faces)
        x1=time.perf_counter()
        baked_vertices=norm[vmapping]

        # These arrays are only a few MB; compression burns CPU and adds latency.
        np.savez(
            root/"bake_mesh.npz",
            vertices=baked_vertices.astype(np.float32),
            faces=np.asarray(indices,dtype=np.int32),
            uvs=np.asarray(uvs,dtype=np.float32),
            scale=np.asarray(scale,dtype=np.float32),
            center=center.astype(np.float32),
            x_rot=x_rot,
            z_rot=z_rot,
        )
        artifacts.commit()
        delete0=time.perf_counter(); state_handoff.pop(job_id,None); delete1=time.perf_counter()
        result={
            "job_id":job_id,
            "instance_id":self.instance_id,
            "worker_load_seconds":round(self.load_seconds,3),
            "state_handoff_source":handoff_source,
            "state_handoff_get_seconds":round(float(g1-g0),3),
            "state_persist_and_load_seconds":round(float(persist1-persist0),3),
            "state_handoff_delete_seconds":round(float(delete1-delete0),3),
            "ply_rebuild_seconds":round(pg1-pg0,3),
            "input_vertices":int(input_vertices),
            "input_faces":int(input_faces),
            "dec_vertices":len(vertices),
            "dec_faces":len(faces),
            "was_simplified":was_simplified,
            "simplify_seconds":round(simplify1-simplify0,3),
            "uv_vertices":len(baked_vertices),
            "uv_faces":len(indices),
            "xatlas_seconds":round(x1-x0,3),
            "method_seconds":round(time.perf_counter()-t0,3),
        }
        print("MESH_PROCESS_OK",json.dumps(result),flush=True)
        return result


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/artifacts": artifacts},
    timeout=15 * 60,
    cpu=4.0,
    memory=16384,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=AUTOSCALE_PROFILES[STATIC_AUTOSCALE_PROFILE]["lite"],
)
def lite_gpu_bake(job_id: str) -> dict:
    """Light L40S: gsplat multiview render + texture bake; no SAM3D model."""
    import math

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from embodied_gen.data.backproject_v3 import TextureBaker
    from embodied_gen.data.utils import CameraSetting, init_kal_camera, post_process_texture
    from embodied_gen.models.gs_model import load_gs_model
    from gsplat import rasterization
    from PIL import Image

    t0=time.perf_counter()
    artifacts.reload()
    root=Path("/artifacts/embodiedgen/jobs")/job_id
    d=np.load(root/"bake_mesh.npz")
    vertices=d["vertices"]; faces=d["faces"]; uvs=d["uvs"]
    cp=CameraSetting(num_images=24,elevation=[0],distance=5.0,resolution_hw=(512,512),fov=math.radians(30),device="cuda")
    cam=init_kal_camera(cp,flip_az=True)
    mv=cam.view_matrix(); mv[:,:3,3]=-mv[:,:3,3]
    K=torch.tensor(cp.Ks,device="cuda")
    model=load_gs_model(str(root/"sample_00_gs_aligned.ply"),pre_quat=[0.,0.,1.,0.])
    views=[]
    r0=time.perf_counter()
    for m in mv:
        c2w=torch.linalg.inv(m.to("cuda")); gs=model.get_gaussians(c2w,apply_activate=True)
        renders,_,_=rasterization(
            means=gs._means,quats=gs._quats,scales=gs._scales,
            opacities=gs._opacities.squeeze(),colors=gs._rgbs,
            viewmats=torch.linalg.inv(c2w)[None,...],Ks=K[None,...],width=512,height=512,
            packed=False,absgrad=True,sparse_grad=False,rasterize_mode="antialiased",
            near_plane=0.01,far_plane=1_000_000_000,radius_clip=0.0,render_mode="RGB")
        torch.cuda.synchronize()
        views.append((renders[0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy())
    r1=time.perf_counter()

    b0=time.perf_counter()
    baker=TextureBaker(vertices,faces,uvs,cp,device="cuda")
    texture=baker.bake_texture([v[...,:3] for v in views],texture_size=1024,mode="fast")
    texture=post_process_texture(texture)
    b1=time.perf_counter()
    Image.fromarray(texture).save(root/"texture.png")

    # Preview is nearly free compared with model inference; reuse horizontal gsplat orbit.
    preview=[]
    cpv=CameraSetting(num_images=60,elevation=[0],distance=5.0,resolution_hw=(512,512),fov=math.radians(30),device="cuda")
    camv=init_kal_camera(cpv,flip_az=True); mvv=camv.view_matrix(); mvv[:,:3,3]=-mvv[:,:3,3]
    Kv=torch.tensor(cpv.Ks,device="cuda")
    for m in mvv:
        c2w=torch.linalg.inv(m.to("cuda")); gs=model.get_gaussians(c2w,apply_activate=True)
        rr,_,_=rasterization(means=gs._means,quats=gs._quats,scales=gs._scales,
            opacities=gs._opacities.squeeze(),colors=gs._rgbs,
            viewmats=torch.linalg.inv(c2w)[None,...],Ks=Kv[None,...],width=512,height=512,
            packed=False,absgrad=True,sparse_grad=False,rasterize_mode="antialiased",
            near_plane=0.01,far_plane=1_000_000_000,radius_clip=0.0,render_mode="RGB")
        torch.cuda.synchronize()
        preview.append((rr[0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy())
    imageio.mimsave(str(root/"preview.mp4"),preview,fps=30,codec="libx264")
    artifacts.commit()
    result={
        "job_id":job_id,
        "gpu":torch.cuda.get_device_name(0),
        "render24_seconds":round(r1-r0,3),
        "bake_seconds":round(b1-b0,3),
        "total_seconds":round(time.perf_counter()-t0,3),
    }
    print("LITE_GPU_BAKE_OK",json.dumps(result),flush=True)
    return result


@app.function(
    image=cpu_image,
    volumes={"/artifacts": artifacts},
    timeout=15 * 60,
    cpu=4.0,
    memory=16384,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=AUTOSCALE_PROFILES[STATIC_AUTOSCALE_PROFILE]["finalize"],
)
def cpu_finalize(job_id: str, cleanup_intermediates: bool = False) -> dict:
    """Pure CPU: export/validate final assets; optionally prune API intermediates."""
    import json as _json
    import shutil
    import xml.etree.ElementTree as ET

    import numpy as np
    import trimesh
    from PIL import Image

    t0=time.perf_counter()
    artifacts.reload()
    root=Path("/artifacts/embodiedgen/jobs")/job_id
    d=np.load(root/"bake_mesh.npz")
    vertices=d["vertices"]; faces=d["faces"]; uvs=d["uvs"]
    scale=float(d["scale"]); center=d["center"]; x_rot=d["x_rot"]; z_rot=d["z_rot"]
    vertices=vertices @ np.linalg.inv(z_rot)
    vertices=vertices @ np.linalg.inv(x_rot)
    vertices=vertices/scale + center
    texture=Image.open(root/"texture.png").convert("RGB")
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs,image=texture),process=True)
    obj=root/"sample_00.obj"; glb=root/"sample_00.glb"
    mesh.export(obj); mesh.export(glb)

    result=root/"result"; meshdir=result/"mesh"
    if result.exists(): shutil.rmtree(result)
    meshdir.mkdir(parents=True,exist_ok=True)
    copy_obj_bundle(obj,meshdir)
    for pth in root.glob("sample_00*.*"):
        if pth.suffix.lower() in {".glb",".ply"}: shutil.copy2(pth,meshdir/pth.name)
    if (root/"texture.png").exists(): shutil.copy2(root/"texture.png",meshdir/"texture.png")
    if (root/"preview.mp4").exists(): shutil.copy2(root/"preview.mp4",result/"video.mp4")

    # GPT-free fallback attributes match the upstream fallback semantics.
    robot=ET.Element("robot",{"name":"sample_00"})
    link=ET.SubElement(robot,"link",{"name":"sample_00"})
    visual=ET.SubElement(link,"visual"); ET.SubElement(visual,"origin",{"xyz":"0 0 0","rpy":"1.5708 0 1.5708"})
    geom=ET.SubElement(visual,"geometry"); ET.SubElement(geom,"mesh",{"filename":"mesh/sample_00.obj","scale":"1 1 1"})
    collision=ET.SubElement(link,"collision"); ET.SubElement(collision,"origin",{"xyz":"0 0 0","rpy":"1.5708 0 1.5708"})
    cgeom=ET.SubElement(collision,"geometry"); ET.SubElement(cgeom,"mesh",{"filename":"mesh/sample_00.obj","scale":"1 1 1"})
    inertial=ET.SubElement(link,"inertial"); ET.SubElement(inertial,"mass",{"value":"1.0"})
    extra=ET.SubElement(link,"extra_info")
    for k,v in {"category":"unknown","description":"unknown","real_height":"1.0","version":"2.0.0","gs_model":"mesh/sample_00_gs.ply"}.items(): ET.SubElement(extra,k).text=v
    urdf=result/"sample_00.urdf"; ET.ElementTree(robot).write(urdf,encoding="utf-8",xml_declaration=True)

    # Structural validation.
    result_obj=meshdir/"sample_00.obj"; result_glb=meshdir/"sample_00.glb"
    objm=trimesh.load(result_obj,force="mesh"); glbs=trimesh.load(result_glb,force="scene")
    ET.parse(urdf)
    with (root/"sample_00_gs.ply").open("rb") as f: header=f.read(8192).decode("ascii","ignore")
    ply_vertices=next(int(x.split()[-1]) for x in header.splitlines() if x.startswith("element vertex "))
    checks={
        "ply_vertices":ply_vertices,
        "obj_vertices":len(objm.vertices),
        "obj_faces":len(objm.faces),
        "glb_geometries":len(glbs.geometry),
        "urdf_mesh_exists":result_obj.exists(),
        "video_exists":(result/"video.mp4").exists(),
        "obj_material_missing":missing_obj_material_dependencies(result_obj),
    }
    checks["obj_material_refs_ok"]=not checks["obj_material_missing"]
    if not validation_passes(checks):
        raise RuntimeError(checks)
    report={"job_id":job_id,"checks":checks,"seconds":round(time.perf_counter()-t0,3)}
    if cleanup_intermediates:
        report["cleaned_intermediates"]=prune_job_intermediates(root)
    (root/"validation_report.json").write_text(_json.dumps(report,indent=2)+"\n")
    artifacts.commit()
    print("VALIDATION_OK",_json.dumps(report),flush=True)
    return report



def _job_update(job_id: str, **changes) -> dict:
    state = dict(job_states.get(job_id) or {})
    now = time.time()
    state.update(changes)
    state.update(
        job_id=job_id,
        updated_epoch=now,
        updated_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
    )
    job_states.put(job_id, state)
    return state


@app.function(
    image=control_image,
    cpu=0.25,
    memory=512,
    timeout=60 * 60,
    min_containers=0,
    max_containers=8,
    buffer_containers=0,
    scaledown_window=2,
)
def run_retexture_job(job_id: str,source_job_id: str,profile: str,prompt: str,seed: int=0) -> dict:
    if not is_api_job_id(job_id) or not is_api_job_id(source_job_id):
        raise ValueError("invalid API job id")
    if profile not in RETEXTURE_SCALEDOWN_WINDOWS:
        raise ValueError(f"unknown autoscale profile: {profile!r}")
    prompt=normalize_text_prompt(prompt)
    worker=RetextureWorker()
    worker.update_autoscaler(
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=RETEXTURE_SCALEDOWN_WINDOWS[profile],
    )
    started=time.perf_counter()
    try:
        _job_update(job_id,status="running",stage="retexture")
        result=worker.generate.remote(job_id,source_job_id,prompt,seed)
        return _job_update(
            job_id,
            status="succeeded",
            stage="done",
            profile=profile,
            stage_seconds={"retexture":round(time.perf_counter()-started,3)},
            files=sorted(RESULT_FILES),
            validation=result,
        )
    except Exception as exc:
        _job_update(
            job_id,
            status="failed",
            stage="retexture",
            profile=profile,
            stage_seconds={"retexture":round(time.perf_counter()-started,3)},
            error_type=type(exc).__name__,
            error=str(exc)[:2000],
        )
        raise


@app.function(
    image=control_image,
    cpu=0.25,
    memory=512,
    timeout=60 * 60,
    min_containers=0,
    max_containers=8,
    buffer_containers=0,
    scaledown_window=2,
)
def run_job(job_id: str, profile: str, prompt: str | None=None, text_seed: int=0) -> dict:
    """Cheap orchestrator; optional Text→Image feeds the same validated Image→3D stages."""
    if not is_api_job_id(job_id):
        raise ValueError(f"invalid API job id: {job_id!r}")
    if profile not in AUTOSCALE_PROFILES:
        raise ValueError(f"unknown autoscale profile: {profile!r}")
    rembg_worker, sam_worker, mesh_worker, lite_worker, finalizer = runtime_handles()
    stages=[]
    if prompt is not None:
        prompt=normalize_text_prompt(prompt)
        text_worker=Text2ImageWorker()
        text_worker.update_autoscaler(
            min_containers=0,
            max_containers=1,
            buffer_containers=0,
            scaledown_window=TEXT2IMG_SCALEDOWN_WINDOWS[profile],
        )
        stages.append(("text2image",lambda: text_worker.generate.remote(job_id,prompt,text_seed)))
    stages.extend((
        ("rembg", lambda: rembg_worker.prepare.remote(job_id)),
        ("sam3d", lambda: sam_worker.generate.remote(job_id)),
        ("mesh", lambda: mesh_worker.process.remote(job_id)),
        ("texture", lambda: lite_worker.remote(job_id)),
        ("finalize", lambda: finalizer.remote(job_id, True)),
    ))
    stage = "dispatch"
    timings = {}
    try:
        for stage, invoke in stages:
            _job_update(job_id, status="running", stage=stage)
            started = time.perf_counter()
            invoke()
            timings[stage] = round(time.perf_counter() - started, 3)
        return _job_update(
            job_id,
            status="succeeded",
            stage="done",
            profile=profile,
            stage_seconds=timings,
            files=sorted(RESULT_FILES),
        )
    except Exception as exc:
        _job_update(
            job_id,
            status="failed",
            stage=stage,
            profile=profile,
            stage_seconds=timings,
            error_type=type(exc).__name__,
            error=str(exc)[:2000],
        )
        raise


@app.function(
    image=control_image,
    volumes={"/artifacts": artifacts},
    cpu=0.25,
    memory=512,
    timeout=10 * 60,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
    schedule=modal.Period(hours=6),
)
def cleanup_stale_api_jobs() -> dict:
    """Delete stale UUID API jobs only; benchmark/debug directories are untouched."""
    import shutil

    artifacts.reload()
    now = time.time()
    deleted = []
    if JOB_ROOT.exists():
        for path in JOB_ROOT.iterdir():
            if not path.is_dir() or not is_api_job_id(path.name):
                continue
            status = job_states.get(path.name)
            if api_job_expired(status, now=now, fallback_mtime=path.stat().st_mtime):
                shutil.rmtree(path)
                job_states.pop(path.name, None)
                state_handoff.pop(path.name, None)
                deleted.append(path.name)
    if deleted:
        artifacts.commit()
    result = {"deleted": deleted, "count": len(deleted)}
    print("API_JOB_CLEANUP", json.dumps(result), flush=True)
    return result


@app.function(
    image=control_image,
    volumes={"/artifacts": artifacts},
    cpu=0.25,
    memory=512,
    timeout=5 * 60,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
)
@modal.asgi_app(requires_proxy_auth=True)
def job_api():
    from io import BytesIO

    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from PIL import Image

    web = FastAPI(title="EmbodiedGen Job API", version="1.2")

    @web.get("/health")
    def health():
        return {"ok": True, "app": APP_NAME, "default_profile": DEFAULT_REQUEST_PROFILE}

    @web.post("/jobs", status_code=202)
    async def submit_job(
        body: bytes = Body(..., media_type="application/octet-stream"),
        profile: str = DEFAULT_REQUEST_PROFILE,
    ):
        if profile != "auto" and profile not in AUTOSCALE_PROFILES:
            raise HTTPException(status_code=400, detail="unknown autoscale profile")
        if not body or len(body) > MAX_INPUT_BYTES:
            raise HTTPException(status_code=413, detail="image body must be 1..20 MiB")
        try:
            with Image.open(BytesIO(body)) as probe:
                width, height = probe.size
                fmt = probe.format
                probe.verify()
        except Exception as exc:
            raise HTTPException(status_code=415, detail="invalid image") from exc
        if width * height > MAX_INPUT_PIXELS:
            raise HTTPException(status_code=413, detail="image exceeds 40 megapixels")

        job_id = new_job_id()
        root = api_job_root(job_id)
        root.mkdir(parents=True, exist_ok=False)
        (root / "input_image").write_bytes(body)
        await artifacts.commit.aio()

        try:
            profile_info = await select_request_profile_aio(profile)
            await apply_autoscale_profile_aio(profile_info["selected_profile"])
        except Exception as exc:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
            await artifacts.commit.aio()
            raise HTTPException(status_code=503, detail="autoscale setup failed") from exc

        now = time.time()
        state = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "profile": profile_info["selected_profile"],
            "requested_profile": profile,
            "created_epoch": now,
            "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "updated_epoch": now,
            "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "input": {"bytes": len(body), "width": width, "height": height, "format": fmt},
        }
        await job_states.put.aio(job_id,state)
        try:
            await run_job.spawn.aio(job_id,profile_info["selected_profile"])
        except Exception as exc:
            failed=dict(state)
            failed.update(
                status="failed",
                stage="dispatch",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
            stamp=time.time()
            failed["updated_epoch"]=stamp
            failed["updated_at"]=datetime.fromtimestamp(stamp,timezone.utc).isoformat()
            await job_states.put.aio(job_id,failed)
            raise HTTPException(status_code=503, detail="job dispatch failed") from exc
        return state

    @web.post("/text-jobs", status_code=202)
    async def submit_text_job(
        payload: dict = Body(..., media_type="application/json"),  # noqa: B008
        profile: str = DEFAULT_REQUEST_PROFILE,
    ):
        if profile != "auto" and profile not in AUTOSCALE_PROFILES:
            raise HTTPException(status_code=400, detail="unknown autoscale profile")
        try:
            prompt=normalize_text_prompt(payload.get("prompt"))
        except (TypeError,ValueError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc
        seed=payload.get("seed",0)
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed <= 100000:
            raise HTTPException(status_code=422,detail="seed must be an integer in 0..100000")

        job_id=new_job_id()
        root=api_job_root(job_id)
        root.mkdir(parents=True,exist_ok=False)
        (root/"prompt.txt").write_text(prompt+"\n")
        await artifacts.commit.aio()

        try:
            profile_info=await select_request_profile_aio(profile)
            await apply_autoscale_profile_aio(profile_info["selected_profile"])
        except Exception as exc:
            import shutil

            shutil.rmtree(root,ignore_errors=True)
            await artifacts.commit.aio()
            raise HTTPException(status_code=503,detail="autoscale setup failed") from exc

        now=time.time()
        state={
            "job_id":job_id,
            "status":"queued",
            "stage":"queued",
            "profile":profile_info["selected_profile"],
            "requested_profile":profile,
            "created_epoch":now,
            "created_at":datetime.fromtimestamp(now,timezone.utc).isoformat(),
            "updated_epoch":now,
            "updated_at":datetime.fromtimestamp(now,timezone.utc).isoformat(),
            "input":{
                "type":"text",
                "prompt_chars":len(prompt),
                "backend":"kolors",
                "seed":seed,
            },
        }
        await job_states.put.aio(job_id,state)
        try:
            await run_job.spawn.aio(job_id,profile_info["selected_profile"],prompt,seed)
        except Exception as exc:
            failed=dict(state)
            failed.update(
                status="failed",
                stage="dispatch",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
            stamp=time.time()
            failed["updated_epoch"]=stamp
            failed["updated_at"]=datetime.fromtimestamp(stamp,timezone.utc).isoformat()
            await job_states.put.aio(job_id,failed)
            raise HTTPException(status_code=503,detail="job dispatch failed") from exc
        return state

    @web.post("/jobs/{source_job_id}/retexture", status_code=202)
    async def submit_retexture_job(
        source_job_id: str,
        payload: dict = Body(..., media_type="application/json"),  # noqa: B008
        profile: str = DEFAULT_REQUEST_PROFILE,
    ):
        if not is_api_job_id(source_job_id):
            raise HTTPException(status_code=404,detail="source job not found")
        source_state=await job_states.get.aio(source_job_id)
        if not source_state or source_state.get("status") != "succeeded":
            raise HTTPException(status_code=409,detail="source job must be succeeded")
        if profile != "auto" and profile not in RETEXTURE_SCALEDOWN_WINDOWS:
            raise HTTPException(status_code=400,detail="unknown autoscale profile")
        try:
            prompt=normalize_text_prompt(payload.get("prompt"))
        except (TypeError,ValueError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc
        seed=payload.get("seed",0)
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed <= 100000:
            raise HTTPException(status_code=422,detail="seed must be an integer in 0..100000")

        profile_info=await select_request_profile_aio(profile)
        selected=profile_info["selected_profile"]
        job_id=new_job_id()
        root=api_job_root(job_id)
        root.mkdir(parents=True,exist_ok=False)
        (root/"source_job.txt").write_text(source_job_id+"\n")
        (root/"retexture_prompt.txt").write_text(prompt+"\n")
        await artifacts.commit.aio()
        now=time.time()
        state={
            "job_id":job_id,
            "status":"queued",
            "stage":"queued",
            "profile":selected,
            "requested_profile":profile,
            "created_epoch":now,
            "created_at":datetime.fromtimestamp(now,timezone.utc).isoformat(),
            "updated_epoch":now,
            "updated_at":datetime.fromtimestamp(now,timezone.utc).isoformat(),
            "input":{
                "type":"retexture",
                "source_job_id":source_job_id,
                "prompt_chars":len(prompt),
                "backend":"kolors-texture-controlnet",
                "seed":seed,
                "delight":False,
                "ip_adapter":False,
            },
        }
        await job_states.put.aio(job_id,state)
        try:
            await run_retexture_job.spawn.aio(job_id,source_job_id,selected,prompt,seed)
        except Exception as exc:
            failed=dict(state)
            failed.update(
                status="failed",
                stage="dispatch",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
            stamp=time.time()
            failed["updated_epoch"]=stamp
            failed["updated_at"]=datetime.fromtimestamp(stamp,timezone.utc).isoformat()
            await job_states.put.aio(job_id,failed)
            raise HTTPException(status_code=503,detail="job dispatch failed") from exc
        return state

    @web.get("/jobs/{job_id}")
    def get_job(job_id: str):
        if not is_api_job_id(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        state = job_states.get(job_id)
        if not state:
            raise HTTPException(status_code=404, detail="job not found")
        state = dict(state)
        if state.get("status") == "succeeded":
            state["file_urls"] = {
                name: f"/jobs/{job_id}/files/{name}" for name in RESULT_FILES
            }
        return state

    @web.get("/jobs/{job_id}/files/{name}")
    def get_file(job_id: str, name: str):
        if not is_api_job_id(job_id) or name not in RESULT_FILES:
            raise HTTPException(status_code=404, detail="file not found")
        artifacts.reload()
        path = api_job_root(job_id) / RESULT_FILES[name]
        if not path.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(path, filename=path.name)

    return web


@app.local_entrypoint()
async def autoscale_policy_check(profile: str = DEFAULT_REQUEST_PROFILE):
    """Validate the ASGI-safe async control path without invoking model workers."""
    profile_info=await select_request_profile_aio(profile)
    await apply_autoscale_profile_aio(profile_info["selected_profile"])
    print("AUTOSCALE_PROFILE",json.dumps(profile_info,ensure_ascii=False,indent=2),flush=True)


@app.local_entrypoint()
def benchmark_split(profile: str = DEFAULT_REQUEST_PROFILE):
    """Run cold→warm benchmark under AUTO or an explicit same-pool autoscale profile."""
    profile_info=select_request_profile(profile)
    print("AUTOSCALE_PROFILE",json.dumps(profile_info,ensure_ascii=False,indent=2),flush=True)
    print("WEIGHTS", preload_weights.remote(), flush=True)
    handles=runtime_handles()
    rembg_worker, worker, mesh_worker, lite_worker, finalizer=apply_autoscale_profile(
        profile_info["selected_profile"], handles
    )

    jobs=[]
    # Prepare both inputs before allocating the heavy GPU. Calls are back-to-back,
    # so even COST_FIRST normally reuses the same U2Net/ONNX session.
    for label in ("cold","warm"):
        import uuid

        job_id=f"bench-{label}-{uuid.uuid4().hex}"
        p0=time.perf_counter(); prep=rembg_worker.prepare.remote(job_id); pwall=time.perf_counter()-p0
        jobs.append({"label":label,"job_id":job_id,"prepare":prep,"prepare_client_wall":round(pwall,3)})
        print(f"PREPARED_{label.upper()}",json.dumps(jobs[-1],ensure_ascii=False,indent=2),flush=True)

    # Heavy GPU calls are now consecutive.
    for item in jobs:
        g0=time.perf_counter(); gen=worker.generate.remote(item["job_id"]); gwall=time.perf_counter()-g0
        item["sam3d"]=gen; item["sam3d_client_wall"]=round(gwall,3)
        print(f"SAM3D_{item['label'].upper()}",json.dumps(item,ensure_ascii=False,indent=2),flush=True)

    same_instance=jobs[0]["sam3d"]["instance_id"]==jobs[1]["sam3d"]["instance_id"]
    reuse={
        "same_instance":same_instance,
        "cold_instance":jobs[0]["sam3d"]["instance_id"],
        "warm_instance":jobs[1]["sam3d"]["instance_id"],
        "cold_client_wall":jobs[0]["sam3d_client_wall"],
        "warm_client_wall":jobs[1]["sam3d_client_wall"],
        "model_load_seconds":jobs[0]["sam3d"]["resident_model_load_seconds"],
        "cold_inference_seconds":jobs[0]["sam3d"]["inference_seconds"],
        "warm_inference_seconds":jobs[1]["sam3d"]["inference_seconds"],
    }
    print("SAM3D_WARM_REUSE",json.dumps(reuse,ensure_ascii=False),flush=True)
    if not same_instance:
        raise RuntimeError("warm benchmark did not reuse the resident SAM3D instance")

    # Downstream stages happen after the reuse measurement, so the heavy SAM3D
    # worker can naturally idle then scale to zero while xatlas uses CPU only.
    for item in jobs:
        down0=time.perf_counter()
        x0=time.perf_counter(); xr=mesh_worker.process.remote(item["job_id"]); xwall=time.perf_counter()-x0
        b0=time.perf_counter(); br=lite_worker.remote(item["job_id"]); bwall=time.perf_counter()-b0
        f0=time.perf_counter(); fr=finalizer.remote(item["job_id"]); fwall=time.perf_counter()-f0
        item.update({
            "xatlas":xr,"xatlas_client_wall":round(xwall,3),
            "lite_gpu":br,"lite_gpu_client_wall":round(bwall,3),
            "final":fr,"final_client_wall":round(fwall,3),
            "downstream_wall":round(time.perf_counter()-down0,3),
        })
        print(f"PIPELINE_{item['label'].upper()}",json.dumps(item,ensure_ascii=False,indent=2),flush=True)

    print("SPLIT_BENCHMARK_SUMMARY",json.dumps({"reuse":reuse,"jobs":jobs},ensure_ascii=False,indent=2),flush=True)
