"""Local/VPS control plane for the deployed EmbodiedGen Modal workers.

This module deliberately performs orchestration in the caller process.  It uses
Modal control-plane APIs to upload inputs, persist compact job state, update
worker autoscalers, and invoke the real compute workers directly.  It never
spawns a Modal gateway/orchestrator container.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import modal

APP_NAME = "modal-3d-embodiedgen"
AFFORDANCE_APP_NAME = "modal-3d-embodiedgen-affordance"
AFFORDANCE_SEMANTIC_APP_NAME = "modal-3d-embodiedgen-affordance-semantic"
ARTIFACT_VOLUME = "modal-3d-artifacts"
JOB_DICT = "modal-3d-embodiedgen-jobs"
TRAFFIC_DICT = "modal-3d-embodiedgen-traffic"
JOB_PREFIX = "job-"
JOB_VOLUME_PREFIX = "/embodiedgen/jobs"

AUTOSCALE_PROFILES = {
    "min_cost": {"rembg": 2, "sam3d": 2, "mesh": 2, "lite": 2, "finalize": 2},
    "cost_first": {"rembg": 60, "sam3d": 30, "mesh": 30, "lite": 10, "finalize": 2},
    "balanced": {"rembg": 120, "sam3d": 90, "mesh": 90, "lite": 30, "finalize": 10},
    "burst": {"rembg": 300, "sam3d": 180, "mesh": 120, "lite": 60, "finalize": 30},
}
TEXT2IMG_WINDOWS = {"min_cost": 2, "cost_first": 30, "balanced": 90, "burst": 180}
RETEXTURE_WINDOWS = dict(TEXT2IMG_WINDOWS)
RESULT_FILES = {
    "glb": "result/mesh/sample_00.glb",
    "obj": "result/mesh/sample_00.obj",
    "mtl": "result/mesh/material.mtl",
    "obj_texture": "result/mesh/material_0.png",
    "urdf": "result/sample_00.urdf",
    "video": "result/video.mp4",
    "validation": "validation_report.json",
}


def new_job_id() -> str:
    return f"{JOB_PREFIX}{uuid.uuid4().hex}"


def _stamp(state: dict) -> dict:
    now = time.time()
    state["updated_epoch"] = now
    state["updated_at"] = datetime.fromtimestamp(now, UTC).isoformat()
    return state


def _jobs():
    return modal.Dict.from_name(JOB_DICT, create_if_missing=True)


def _traffic():
    return modal.Dict.from_name(TRAFFIC_DICT, create_if_missing=True)


def _artifacts():
    return modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=False)


def get_job(job_id: str) -> dict | None:
    return _jobs().get(job_id)


def _put_job(job_id: str, **changes) -> dict:
    jobs = _jobs()
    state = dict(jobs.get(job_id) or {"job_id": job_id})
    state.update(changes)
    _stamp(state)
    jobs.put(job_id, state)
    return state


def select_profile(requested: str = "auto", *, now: float | None = None) -> str:
    if requested != "auto":
        if requested not in AUTOSCALE_PROFILES:
            raise ValueError(f"unknown autoscale profile: {requested}")
        return requested
    now = time.time() if now is None else float(now)
    traffic = _traffic()
    key = f"request:{now:.6f}:{uuid.uuid4().hex}"
    traffic.put(key, now)
    recent = 0
    stale = []
    for event_key, timestamp in traffic.items():
        try:
            age = now - float(timestamp)
        except (TypeError, ValueError):
            stale.append(event_key)
            continue
        if age <= 60.0:
            recent += 1
        else:
            stale.append(event_key)
    for event_key in stale:
        traffic.pop(event_key, None)
    return "cost_first" if recent >= 2 else "min_cost"


def _compute_handles():
    rembg = modal.Cls.from_name(APP_NAME, "RembgWorker")()
    sam3d = modal.Cls.from_name(APP_NAME, "Sam3DWorker")()
    mesh = modal.Cls.from_name(APP_NAME, "MeshWorker")()
    lite = modal.Function.from_name(APP_NAME, "lite_gpu_bake")
    finalize = modal.Function.from_name(APP_NAME, "cpu_finalize")
    return rembg, sam3d, mesh, lite, finalize


def apply_profile(profile: str):
    cfg = AUTOSCALE_PROFILES[profile]
    handles = _compute_handles()
    common = {"min_containers": 0, "max_containers": 1, "buffer_containers": 0}
    for stage, target in zip(("rembg", "sam3d", "mesh", "lite", "finalize"), handles, strict=True):
        target.update_autoscaler(scaledown_window=cfg[stage], **common)
    return handles


def _run_stage(job_id: str, stage: str, invoke: Callable[[], object], timings: dict) -> object:
    _put_job(job_id, status="running", stage=stage)
    started = time.perf_counter()
    try:
        result = invoke()
    except Exception as exc:
        timings[stage] = round(time.perf_counter() - started, 3)
        _put_job(
            job_id,
            status="failed",
            stage=stage,
            stage_seconds=dict(timings),
            error_type=type(exc).__name__,
            error=str(exc)[:2000],
        )
        raise
    timings[stage] = round(time.perf_counter() - started, 3)
    return result


def upload_image(image_path: str | Path, job_id: str | None = None) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    job_id = job_id or new_job_id()
    remote = f"{JOB_VOLUME_PREFIX}/{job_id}/input_image"
    with _artifacts().batch_upload(force=True) as batch:
        batch.put_file(path, remote)
    now = time.time()
    _jobs().put(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "created_epoch": now,
            "created_at": datetime.fromtimestamp(now, UTC).isoformat(),
            "updated_epoch": now,
            "updated_at": datetime.fromtimestamp(now, UTC).isoformat(),
            "input": {"type": "image", "bytes": path.stat().st_size, "name": path.name},
        },
    )
    return job_id


def generate_image3d(image_path: str | Path, profile: str = "auto") -> dict:
    job_id = upload_image(image_path)
    selected = select_profile(profile)
    _put_job(job_id, profile=selected, requested_profile=profile)
    rembg, sam3d, mesh, lite, finalize = apply_profile(selected)
    timings: dict[str, float] = {}
    _run_stage(job_id, "rembg", lambda: rembg.prepare.remote(job_id), timings)
    _run_stage(job_id, "sam3d", lambda: sam3d.generate.remote(job_id), timings)
    _run_stage(job_id, "mesh", lambda: mesh.process.remote(job_id), timings)
    _run_stage(job_id, "texture", lambda: lite.remote(job_id), timings)
    validation = _run_stage(job_id, "finalize", lambda: finalize.remote(job_id, True), timings)
    return _put_job(
        job_id,
        status="succeeded",
        stage="done",
        profile=selected,
        stage_seconds=timings,
        files=sorted(RESULT_FILES),
        validation=validation,
    )


def generate_text3d(prompt: str, seed: int = 0, profile: str = "auto") -> dict:
    prompt = str(prompt).strip()
    if not prompt or len(prompt) > 1000:
        raise ValueError("prompt must contain 1..1000 characters")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 100000:
        raise ValueError("seed must be an integer in 0..100000")
    job_id = new_job_id()
    selected = select_profile(profile)
    now = time.time()
    _jobs().put(job_id, _stamp({
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "profile": selected,
        "requested_profile": profile,
        "created_epoch": now,
        "created_at": datetime.fromtimestamp(now, UTC).isoformat(),
        "input": {"type": "text", "prompt_chars": len(prompt), "seed": seed},
    }))
    rembg, sam3d, mesh, lite, finalize = apply_profile(selected)
    text = modal.Cls.from_name(APP_NAME, "Text2ImageWorker")()
    text.update_autoscaler(
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=TEXT2IMG_WINDOWS[selected],
    )
    timings: dict[str, float] = {}
    _run_stage(job_id, "text2image", lambda: text.generate.remote(job_id, prompt, seed), timings)
    _run_stage(job_id, "rembg", lambda: rembg.prepare.remote(job_id), timings)
    _run_stage(job_id, "sam3d", lambda: sam3d.generate.remote(job_id), timings)
    _run_stage(job_id, "mesh", lambda: mesh.process.remote(job_id), timings)
    _run_stage(job_id, "texture", lambda: lite.remote(job_id), timings)
    validation = _run_stage(job_id, "finalize", lambda: finalize.remote(job_id, True), timings)
    return _put_job(
        job_id,
        status="succeeded",
        stage="done",
        profile=selected,
        stage_seconds=timings,
        files=sorted(RESULT_FILES),
        validation=validation,
    )


def retexture(source_job_id: str, prompt: str, seed: int = 0, profile: str = "auto") -> dict:
    source = get_job(source_job_id)
    if not source or source.get("status") != "succeeded":
        raise ValueError("source job must exist and be succeeded")
    prompt = str(prompt).strip()
    if not prompt or len(prompt) > 1000:
        raise ValueError("prompt must contain 1..1000 characters")
    selected = select_profile(profile)
    job_id = new_job_id()
    now = time.time()
    _jobs().put(job_id, _stamp({
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "profile": selected,
        "requested_profile": profile,
        "created_epoch": now,
        "created_at": datetime.fromtimestamp(now, UTC).isoformat(),
        "workflow": "retexture",
        "source_job_id": source_job_id,
    }))
    worker = modal.Cls.from_name(APP_NAME, "RetextureWorker")()
    worker.update_autoscaler(
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=RETEXTURE_WINDOWS[selected],
    )
    timings: dict[str, float] = {}
    validation = _run_stage(
        job_id,
        "retexture",
        lambda: worker.generate.remote(job_id, source_job_id, prompt, seed),
        timings,
    )
    return _put_job(
        job_id,
        status="succeeded",
        stage="done",
        profile=selected,
        stage_seconds=timings,
        files=sorted(RESULT_FILES),
        validation=validation,
    )


AFFORDANCE_PROFILE = "part-evidence-only"
AFFORDANCE_SEMANTIC_PROFILE = "semantic-evidence-v1"
AFFORDANCE_DEFAULTS = {
    "point_num": 20000,
    "prompt_num": 64,
    "prompt_bs": 8,
    "grasp_num_points": 2024,
    "num_grasps": 80,
    "topk": 20,
    "seed": 42,
}
AFFORDANCE_RESULT_FILES = {
    "source_glb": "source/sample_00.glb",
    "source_urdf": "source/sample_00.urdf",
    "part_segmentation": "affordance/agentscape_part_segmentation.v1.json",
    "raw_grasps": "affordance/raw_grasps.franka.v1.json",
    "segment_validation": "affordance/validation_report.json",
    "grasp_validation": "affordance/graspgen_validation_report.json",
    "affordance_bundle": "affordance/bundle.v1.json",
    "affordance_validation": "validation_report.json",
}
AFFORDANCE_SEMANTIC_RESULT_FILES = {
    "semantic_inputs": "affordance/semantic_inputs/semantic_inputs.v1.json",
    "semantic_rgb_grid": "affordance/semantic_inputs/rgb_grid.png",
    "semantic_mask_grid": "affordance/semantic_inputs/mask_grid.png",
    "semantic_part_atlas": "affordance/semantic_inputs/part_atlas.png",
    "part_semantics": "affordance/part_semantics.v1.json",
    "semantic_validation": "affordance/semantic_validation_report.json",
}


def normalize_affordance_options(payload: dict | None) -> dict:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError("affordance payload must be an object")
    allowed = {
        "profile", "point_num", "prompt_num", "prompt_bs", "grasp_num_points",
        "num_grasps", "topk", "seed", "category",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported affordance options: {unknown}")
    profile = payload.get("profile", AFFORDANCE_PROFILE)
    if profile not in {AFFORDANCE_PROFILE, AFFORDANCE_SEMANTIC_PROFILE}:
        raise ValueError(f"unsupported affordance profile: {profile!r}")
    if profile == AFFORDANCE_PROFILE and "category" in payload:
        raise ValueError("category is only supported by semantic-evidence-v1")
    options = {"profile": profile, **AFFORDANCE_DEFAULTS}
    for key in AFFORDANCE_DEFAULTS:
        if key in payload:
            options[key] = payload[key]
    if profile == AFFORDANCE_SEMANTIC_PROFILE:
        category = str(payload.get("category") or "unknown object").strip()
        if not category or len(category) > 160:
            raise ValueError("category must contain 1..160 characters")
        options["category"] = category
    for key in (
        "point_num", "prompt_num", "prompt_bs", "grasp_num_points",
        "num_grasps", "topk", "seed",
    ):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
    if not 1000 <= options["point_num"] <= 200000:
        raise ValueError("point_num must be in 1000..200000")
    if not 8 <= options["prompt_num"] <= 800:
        raise ValueError("prompt_num must be in 8..800")
    if not 1 <= options["prompt_bs"] <= 64:
        raise ValueError("prompt_bs must be in 1..64")
    if not 512 <= options["grasp_num_points"] <= 20000:
        raise ValueError("grasp_num_points must be in 512..20000")
    if not 1 <= options["num_grasps"] <= 1000:
        raise ValueError("num_grasps must be in 1..1000")
    if not 1 <= options["topk"] <= min(options["num_grasps"], 200):
        raise ValueError("topk must be in 1..min(num_grasps, 200)")
    if not 0 <= options["seed"] <= 2**31 - 1:
        raise ValueError("seed must be a non-negative 32-bit integer")
    return options


def generate_affordance(source_job_id: str, payload: dict | None = None) -> dict:
    source = get_job(source_job_id)
    if not source or source.get("status") != "succeeded":
        raise ValueError("source job must exist and be succeeded")
    options = normalize_affordance_options(payload)
    profile = options["profile"]
    job_id = new_job_id()
    now = time.time()
    _jobs().put(
        job_id,
        _stamp(
            {
                "job_id": job_id,
                "status": "queued",
                "stage": "queued",
                "profile": profile,
                "workflow": "asset.affordance",
                "source_job_id": source_job_id,
                "created_epoch": now,
                "created_at": datetime.fromtimestamp(now, UTC).isoformat(),
                "input": {"type": "affordance", "source_job_id": source_job_id, "options": options},
            }
        ),
    )

    segment = modal.Function.from_name(AFFORDANCE_APP_NAME, "segment_job")
    grasp = modal.Function.from_name(AFFORDANCE_APP_NAME, "raw_grasp_job")
    semantic_inputs = modal.Function.from_name(APP_NAME, "prepare_affordance_semantic_inputs")
    semantic = modal.Function.from_name(AFFORDANCE_SEMANTIC_APP_NAME, "annotate_semantics")
    finalize = modal.Function.from_name(APP_NAME, "finalize_affordance_bundle")
    timings: dict[str, float] = {}

    _run_stage(
        job_id,
        "segment",
        lambda: segment.remote(
            source_job_id,
            point_num=options["point_num"],
            prompt_num=options["prompt_num"],
            prompt_bs=options["prompt_bs"],
            output_job_id=job_id,
        ),
        timings,
    )
    _run_stage(
        job_id,
        "grasp_raw",
        lambda: grasp.remote(
            source_job_id,
            num_points=options["grasp_num_points"],
            num_grasps=options["num_grasps"],
            topk=options["topk"],
            seed=options["seed"],
            output_job_id=job_id,
        ),
        timings,
    )
    if profile == AFFORDANCE_SEMANTIC_PROFILE:
        _run_stage(
            job_id,
            "semantic_inputs",
            lambda: semantic_inputs.remote(job_id, options["category"]),
            timings,
        )
        _run_stage(job_id, "semantic_annotate", lambda: semantic.remote(job_id), timings)

    validation = _run_stage(
        job_id,
        "finalize",
        lambda: finalize.remote(job_id, source_job_id, options),
        timings,
    )
    files = dict(AFFORDANCE_RESULT_FILES)
    if profile == AFFORDANCE_SEMANTIC_PROFILE:
        files.update(AFFORDANCE_SEMANTIC_RESULT_FILES)
    return _put_job(
        job_id,
        status="succeeded",
        stage="done",
        profile=profile,
        workflow="asset.affordance",
        source_job_id=source_job_id,
        options=options,
        stage_seconds=timings,
        files=sorted(files),
        validation=validation,
    )


def download_result(job_id: str, name: str, destination: str | Path) -> Path:
    state = get_job(job_id)
    if not state or state.get("status") != "succeeded":
        raise ValueError("job must exist and be succeeded")
    if name not in (state.get("files") or []):
        raise ValueError(f"result is not available for this job: {name}")

    paths = {**RESULT_FILES, **AFFORDANCE_RESULT_FILES, **AFFORDANCE_SEMANTIC_RESULT_FILES}
    if name not in paths:
        raise ValueError(f"unknown result name: {name}")
    remote = f"{JOB_VOLUME_PREFIX}/{job_id}/{paths[name]}"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        for chunk in _artifacts().read_file(remote):
            stream.write(chunk)
    return destination
