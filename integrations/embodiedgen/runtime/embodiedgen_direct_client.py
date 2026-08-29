"""Local/VPS control plane for the deployed EmbodiedGen Modal compute workers.

This module is a plain local control client, not a deployed Modal application. Every
request is orchestrated by the caller process and goes directly to the real compute
worker (CPU or GPU) via ``Cls.from_name`` / ``Function.from_name``.
"""
from __future__ import annotations

import io
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import modal
from PIL import Image

APP_NAME = "modal-3d-embodiedgen"
AFFORDANCE_APP_NAME = "modal-3d-embodiedgen-affordance"
AFFORDANCE_SEMANTIC_APP_NAME = "modal-3d-embodiedgen-affordance-semantic"
ARTIFACT_VOLUME = "modal-3d-artifacts"
JOB_STATE_DICT = "modal-3d-embodiedgen-jobs"
TRAFFIC_DICT = "modal-3d-embodiedgen-traffic"
JOB_PREFIX = "job-"
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_INPUT_PIXELS = 40_000_000
MAX_PROMPT_CHARS = 1000
AUTO_TRAFFIC_WINDOW_SECONDS = 60.0
AUTO_COST_FIRST_REQUESTS = 2
TRAFFIC_EVENT_PREFIX = "request:"

AUTOSCALE_PROFILES = {
    "min_cost": {"rembg": 2, "sam3d": 2, "mesh": 2, "lite": 2, "finalize": 2},
    "cost_first": {"rembg": 60, "sam3d": 30, "mesh": 30, "lite": 10, "finalize": 2},
    "balanced": {"rembg": 120, "sam3d": 90, "mesh": 90, "lite": 30, "finalize": 10},
    "burst": {"rembg": 300, "sam3d": 180, "mesh": 120, "lite": 60, "finalize": 30},
}
TEXT2IMG_SCALEDOWN_WINDOWS = {
    "min_cost": 2,
    "cost_first": 30,
    "balanced": 90,
    "burst": 180,
}
RETEXTURE_SCALEDOWN_WINDOWS = dict(TEXT2IMG_SCALEDOWN_WINDOWS)

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
ALL_RESULT_FILES = {
    **RESULT_FILES,
    **AFFORDANCE_RESULT_FILES,
    **AFFORDANCE_SEMANTIC_RESULT_FILES,
}
AFFORDANCE_PROFILE = "part-evidence-only"
AFFORDANCE_SEMANTIC_PROFILE = "semantic-evidence-v1"
AFFORDANCE_DEFAULT_OPTIONS = {
    "point_num": 20000,
    "prompt_num": 64,
    "prompt_bs": 8,
    "grasp_num_points": 2024,
    "num_grasps": 80,
    "topk": 20,
    "seed": 42,
}


def _modal_kwargs(client):
    return {"client": client} if client is not None else {}


def new_job_id() -> str:
    return f"{JOB_PREFIX}{uuid.uuid4().hex}"


def _job_id(value: str) -> str:
    if not isinstance(value, str) or not value.startswith(JOB_PREFIX):
        raise ValueError("invalid EmbodiedGen job id")
    suffix = value[len(JOB_PREFIX) :]
    if len(suffix) != 32 or any(ch not in "0123456789abcdef" for ch in suffix):
        raise ValueError("invalid EmbodiedGen job id")
    return value


def _prompt(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prompt must be a string")
    value = value.strip()
    if not value:
        raise ValueError("prompt must not be empty")
    if len(value) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100000:
        raise ValueError("seed must be an integer in 0..100000")
    return value


def _semantic_category(value: object) -> str:
    category = "unknown object" if value is None else str(value).strip()
    if not category or len(category) > 160:
        raise ValueError("semantic category must be a non-empty string <= 160 chars")
    return category


def normalize_affordance_options(payload: dict | None) -> dict:
    payload = {} if payload is None else payload
    if not isinstance(payload, dict):
        raise TypeError("affordance payload must be an object")
    allowed = {
        "profile",
        "point_num",
        "prompt_num",
        "prompt_bs",
        "grasp_num_points",
        "num_grasps",
        "topk",
        "seed",
        "category",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported affordance options: {unknown}")
    profile = payload.get("profile", AFFORDANCE_PROFILE)
    if profile not in {AFFORDANCE_PROFILE, AFFORDANCE_SEMANTIC_PROFILE}:
        raise ValueError(f"unsupported affordance profile: {profile!r}")
    if profile == AFFORDANCE_PROFILE and "category" in payload:
        raise ValueError("category is only supported by semantic-evidence-v1")
    d = AFFORDANCE_DEFAULT_OPTIONS
    out = {
        "profile": profile,
        "point_num": payload.get("point_num", d["point_num"]),
        "prompt_num": payload.get("prompt_num", d["prompt_num"]),
        "prompt_bs": payload.get("prompt_bs", d["prompt_bs"]),
        "grasp_num_points": payload.get("grasp_num_points", d["grasp_num_points"]),
        "num_grasps": payload.get("num_grasps", d["num_grasps"]),
        "topk": payload.get("topk", d["topk"]),
        "seed": payload.get("seed", d["seed"]),
    }
    if profile == AFFORDANCE_SEMANTIC_PROFILE:
        out["category"] = _semantic_category(payload.get("category"))
    for key in (
        "point_num",
        "prompt_num",
        "prompt_bs",
        "grasp_num_points",
        "num_grasps",
        "topk",
        "seed",
    ):
        if isinstance(out[key], bool) or not isinstance(out[key], int):
            raise TypeError(f"{key} must be an integer")
    if not 1000 <= out["point_num"] <= 200000:
        raise ValueError("point_num must be in 1000..200000")
    if not 8 <= out["prompt_num"] <= 800:
        raise ValueError("prompt_num must be in 8..800")
    if not 1 <= out["prompt_bs"] <= 64:
        raise ValueError("prompt_bs must be in 1..64")
    if not 512 <= out["grasp_num_points"] <= 20000:
        raise ValueError("grasp_num_points must be in 512..20000")
    if not 1 <= out["num_grasps"] <= 1000:
        raise ValueError("num_grasps must be in 1..1000")
    if not 1 <= out["topk"] <= min(out["num_grasps"], 200):
        raise ValueError("topk must be in 1..min(num_grasps, 200)")
    if not 0 <= out["seed"] <= 2**31 - 1:
        raise ValueError("seed must be a non-negative 32-bit integer")
    return out


class EmbodiedGenDirectClient:
    """Synchronous orchestration intended to run inside the local VPS service."""

    def __init__(self, *, modal_client=None) -> None:
        kwargs = _modal_kwargs(modal_client)
        self._kwargs = kwargs
        self.artifacts = modal.Volume.from_name(ARTIFACT_VOLUME, **kwargs)
        self.job_states = modal.Dict.from_name(JOB_STATE_DICT, **kwargs)
        self.traffic_events = modal.Dict.from_name(TRAFFIC_DICT, **kwargs)

    def _cls(self, name: str, app_name: str = APP_NAME):
        return modal.Cls.from_name(app_name, name, **self._kwargs)()

    def _fn(self, name: str, app_name: str = APP_NAME):
        return modal.Function.from_name(app_name, name, **self._kwargs)

    @staticmethod
    def _now() -> tuple[float, str]:
        epoch = time.time()
        return epoch, datetime.fromtimestamp(epoch, UTC).isoformat()

    def _put_state(self, job_id: str, **updates) -> dict:
        _job_id(job_id)
        state = dict(self.job_states.get(job_id) or {"job_id": job_id})
        epoch, iso = self._now()
        state.update(updates)
        state["updated_epoch"] = epoch
        state["updated_at"] = iso
        if "created_epoch" not in state:
            state["created_epoch"] = epoch
            state["created_at"] = iso
        self.job_states.put(job_id, state)
        return state

    def _profile(self, requested: str) -> str:
        if requested != "auto":
            if requested not in AUTOSCALE_PROFILES:
                raise ValueError(f"unknown autoscale profile: {requested!r}")
            return requested
        now = time.time()
        self.traffic_events.put(
            f"{TRAFFIC_EVENT_PREFIX}{now:.6f}:{uuid.uuid4().hex}", now
        )
        recent = 0
        stale = []
        for key, timestamp in self.traffic_events.items():
            if not str(key).startswith(TRAFFIC_EVENT_PREFIX):
                continue
            age = now - float(timestamp)
            if 0.0 <= age <= AUTO_TRAFFIC_WINDOW_SECONDS:
                recent += 1
            elif age > AUTO_TRAFFIC_WINDOW_SECONDS:
                stale.append(key)
        for key in stale:
            self.traffic_events.pop(key, None)
        return "cost_first" if recent >= AUTO_COST_FIRST_REQUESTS else "min_cost"

    def _core_handles(self):
        return (
            self._cls("RembgWorker"),
            self._cls("Sam3DWorker"),
            self._cls("MeshWorker"),
            self._fn("lite_gpu_bake"),
            self._fn("cpu_finalize"),
        )

    def _apply_profile(self, profile: str, handles=None):
        cfg = AUTOSCALE_PROFILES[profile]
        handles = handles or self._core_handles()
        for stage, target in zip(
            ("rembg", "sam3d", "mesh", "lite", "finalize"), handles, strict=True
        ):
            target.update_autoscaler(
                min_containers=0,
                max_containers=1,
                buffer_containers=0,
                scaledown_window=cfg[stage],
            )
        return handles

    def _upload_image(self, job_id: str, image_path: str | Path) -> dict:
        path = Path(image_path)
        size = path.stat().st_size
        if not 0 < size <= MAX_INPUT_BYTES:
            raise ValueError("image must be 1..20 MiB")
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as probe:
            width, height = probe.size
            fmt = probe.format
        if width * height > MAX_INPUT_PIXELS:
            raise ValueError("image exceeds 40 megapixels")
        remote_path = f"embodiedgen/jobs/{job_id}/input_image"
        with self.artifacts.batch_upload(force=False) as batch:
            batch.put_file(path, remote_path)
        return {"bytes": size, "width": width, "height": height, "format": fmt}

    def _run_core(
        self,
        job_id: str,
        profile: str,
        *,
        prompt: str | None = None,
        text_seed: int = 0,
    ) -> dict:
        rembg, sam3d, mesh, lite, finalize = self._apply_profile(profile)
        stages = []
        if prompt is not None:
            prompt = _prompt(prompt)
            text_seed = _seed(text_seed)
            text = self._cls("Text2ImageWorker")
            text.update_autoscaler(
                min_containers=0,
                max_containers=1,
                buffer_containers=0,
                scaledown_window=TEXT2IMG_SCALEDOWN_WINDOWS[profile],
            )
            stages.append(("text2image", lambda: text.generate.remote(job_id, prompt, text_seed)))
        stages.extend(
            [
                ("rembg", lambda: rembg.prepare.remote(job_id)),
                ("sam3d", lambda: sam3d.generate.remote(job_id)),
                ("mesh", lambda: mesh.process.remote(job_id)),
                ("texture", lambda: lite.remote(job_id)),
                ("finalize", lambda: finalize.remote(job_id, True)),
            ]
        )
        timings: dict[str, float] = {}
        stage = "dispatch"
        try:
            for stage, invoke in stages:
                self._put_state(job_id, status="running", stage=stage, profile=profile)
                started = time.perf_counter()
                invoke()
                timings[stage] = round(time.perf_counter() - started, 3)
            return self._put_state(
                job_id,
                status="succeeded",
                stage="done",
                profile=profile,
                stage_seconds=timings,
                files=sorted(RESULT_FILES),
            )
        except Exception as exc:
            self._put_state(
                job_id,
                status="failed",
                stage=stage,
                profile=profile,
                stage_seconds=timings,
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
            raise

    def run_image(self, image_path: str | Path, *, profile: str = "auto") -> dict:
        """Upload one source image, then call compute workers directly from this process."""
        job_id = new_job_id()
        selected = self._profile(profile)
        input_info = self._upload_image(job_id, image_path)
        self._put_state(
            job_id,
            status="queued",
            stage="queued",
            profile=selected,
            requested_profile=profile,
            input=input_info,
        )
        return self._run_core(job_id, selected)

    def run_text(self, prompt: str, *, seed: int = 0, profile: str = "auto") -> dict:
        job_id = new_job_id()
        prompt = _prompt(prompt)
        seed = _seed(seed)
        selected = self._profile(profile)
        with self.artifacts.batch_upload(force=False) as batch:
            batch.put_file(
                io.BytesIO((prompt + "\n").encode()),
                f"embodiedgen/jobs/{job_id}/prompt.txt",
            )
        self._put_state(
            job_id,
            status="queued",
            stage="queued",
            profile=selected,
            requested_profile=profile,
            input={"type": "text", "prompt": prompt, "seed": seed},
        )
        return self._run_core(job_id, selected, prompt=prompt, text_seed=seed)

    def run_retexture(
        self,
        source_job_id: str,
        prompt: str,
        *,
        seed: int = 0,
        profile: str = "auto",
    ) -> dict:
        source_job_id = _job_id(source_job_id)
        source = self.job_states.get(source_job_id)
        if not source or source.get("status") != "succeeded":
            raise ValueError("source job must exist and be succeeded")
        prompt = _prompt(prompt)
        seed = _seed(seed)
        selected = self._profile(profile)
        job_id = new_job_id()
        self._put_state(
            job_id,
            status="queued",
            stage="queued",
            workflow="asset.retexture",
            source_job_id=source_job_id,
            profile=selected,
            requested_profile=profile,
        )
        worker = self._cls("RetextureWorker")
        worker.update_autoscaler(
            min_containers=0,
            max_containers=1,
            buffer_containers=0,
            scaledown_window=RETEXTURE_SCALEDOWN_WINDOWS[selected],
        )
        started = time.perf_counter()
        try:
            self._put_state(job_id, status="running", stage="retexture")
            result = worker.generate.remote(job_id, source_job_id, prompt, seed)
            return self._put_state(
                job_id,
                status="succeeded",
                stage="done",
                stage_seconds={"retexture": round(time.perf_counter() - started, 3)},
                files=sorted(RESULT_FILES),
                validation=result,
            )
        except Exception as exc:
            self._put_state(
                job_id,
                status="failed",
                stage="retexture",
                stage_seconds={"retexture": round(time.perf_counter() - started, 3)},
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
            raise

    def run_affordance(self, source_job_id: str, options: dict | None = None) -> dict:
        source_job_id = _job_id(source_job_id)
        source = self.job_states.get(source_job_id)
        if not source or source.get("status") != "succeeded":
            raise ValueError("source job must exist and be succeeded")
        options = normalize_affordance_options(options)
        profile = options["profile"]
        job_id = new_job_id()
        self._put_state(
            job_id,
            status="queued",
            stage="queued",
            workflow="asset.affordance",
            source_job_id=source_job_id,
            options=options,
        )
        segment = self._fn("segment_job", AFFORDANCE_APP_NAME)
        grasp = self._fn("raw_grasp_job", AFFORDANCE_APP_NAME)
        stages = [
            (
                "segment",
                lambda: segment.remote(
                    source_job_id,
                    point_num=options["point_num"],
                    prompt_num=options["prompt_num"],
                    prompt_bs=options["prompt_bs"],
                    output_job_id=job_id,
                ),
            ),
            (
                "grasp_raw",
                lambda: grasp.remote(
                    source_job_id,
                    num_points=options["grasp_num_points"],
                    num_grasps=options["num_grasps"],
                    topk=options["topk"],
                    seed=options["seed"],
                    output_job_id=job_id,
                ),
            ),
        ]
        if profile == AFFORDANCE_SEMANTIC_PROFILE:
            semantic_inputs = self._fn("prepare_affordance_semantic_inputs")
            semantic = self._fn("annotate_semantics", AFFORDANCE_SEMANTIC_APP_NAME)
            stages.extend(
                [
                    (
                        "semantic_inputs",
                        lambda: semantic_inputs.remote(job_id, options["category"]),
                    ),
                    ("semantic_annotate", lambda: semantic.remote(job_id)),
                ]
            )
        finalize = self._fn("finalize_affordance_bundle")
        stages.append(
            ("finalize", lambda: finalize.remote(job_id, source_job_id, options))
        )
        timings: dict[str, float] = {}
        stage = "dispatch"
        try:
            for stage, invoke in stages:
                self._put_state(job_id, status="running", stage=stage)
                started = time.perf_counter()
                invoke()
                timings[stage] = round(time.perf_counter() - started, 3)
            files = sorted(
                {
                    **AFFORDANCE_RESULT_FILES,
                    **(
                        AFFORDANCE_SEMANTIC_RESULT_FILES
                        if profile == AFFORDANCE_SEMANTIC_PROFILE
                        else {}
                    ),
                }
            )
            return self._put_state(
                job_id,
                status="succeeded",
                stage="done",
                workflow="asset.affordance",
                source_job_id=source_job_id,
                options=options,
                stage_seconds=timings,
                files=files,
            )
        except Exception as exc:
            self._put_state(
                job_id,
                status="failed",
                stage=stage,
                workflow="asset.affordance",
                source_job_id=source_job_id,
                options=options,
                stage_seconds=timings,
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
            raise

    def get_job(self, job_id: str) -> dict | None:
        value = self.job_states.get(_job_id(job_id))
        return dict(value) if value else None

    def download(self, job_id: str, role: str, destination: str | Path) -> Path:
        """Download a persisted result directly from the Modal Volume."""
        job_id = _job_id(job_id)
        state = self.get_job(job_id)
        if not state or state.get("status") != "succeeded":
            raise ValueError("job is not succeeded")
        available = set(state.get("files") or RESULT_FILES)
        if role not in available or role not in ALL_RESULT_FILES:
            raise ValueError(f"result role is unavailable: {role}")
        relative = PurePosixPath(ALL_RESULT_FILES[role])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe result path")
        remote = f"embodiedgen/jobs/{job_id}/{relative.as_posix()}"
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for chunk in self.artifacts.read_file(remote):
                stream.write(chunk)
        return target
