"""Isolated GPT semantic worker contract for EmbodiedGen Affordance.

The worker consumes immutable, hash-bound semantic inputs prepared by an upstream
render stage. It never reruns P3-SAM/GraspGen, never mutates the source URDF, and
never promotes semantic proposals into joint/action truth.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import modal

APP_NAME = "modal-3d-embodiedgen-affordance-semantic"
SEMANTIC_SECRET_NAME = "modal-3d-embodiedgen-affordance-semantic"
SEMANTIC_PROFILE = "part-semantics-v1"
SEMANTIC_INPUT_VERSION = 1
SEMANTIC_OUTPUT_VERSION = 1
JOB_ROOT = Path("/artifacts/embodiedgen/jobs")
SEMANTIC_INPUT_PATH = Path("affordance/semantic_inputs/semantic_inputs.v1.json")
SEMANTIC_OUTPUT_PATH = Path("affordance/part_semantics.v1.json")
SEMANTIC_VALIDATION_PATH = Path("affordance/semantic_validation_report.json")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_COMPLETION_TOKENS = 8192
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a strict 3D-part semantic annotator for robotic asset understanding.
You receive an RGB multi-view grid, an aligned colored global part-mask grid, and an isolated part atlas.
The isolated atlas contains three canonical silhouette views per part and may reveal parts hidden in all global views.
Every requested part id must appear exactly once in your JSON result. Use only the supplied part ids
and mask colors. Describe visible semantics conservatively. Graspable means a plausible robot-gripper
contact region, not a guarantee that pickup succeeds. Do not infer or output joints, joint types, axes,
anchors, limits, motors, runtime actions, or verified interaction claims. Return JSON only."""

USER_PROMPT_TEMPLATE = """Object category: {category}
Parts (id -> mask color): {parts}
Return exactly one JSON object with key \"parts\". For each part include:
- id: exact supplied id
- mask_color: exact supplied mask color
- part_name: concise semantic name
- graspable: boolean
- grasp_scenarios: list of {{scenario, confidence}} where confidence is 0..1
- functional_labels: 1..8 short functional phrases
- semantic_description: concise visible/functional description
Do not add any joint/action/physics fields."""

PROMPT_REVISION = hashlib.sha256(
    (SYSTEM_PROMPT + "\n---\n" + USER_PROMPT_TEMPLATE).encode("utf-8")
).hexdigest()

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name("modal-3d-artifacts", create_if_missing=True)
semantic_secret = modal.Secret.from_name(
    SEMANTIC_SECRET_NAME,
    required_keys=["ENDPOINT", "API_KEY", "MODEL_NAME"],
)
semantic_image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"PYTHONUNBUFFERED": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    .uv_pip_install("openai==1.101.0", "pillow==11.3.0")
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_job_relative_path(value: str) -> Path:
    text = str(value or "").strip()
    if not text or len(text) > 512 or "://" in text or "?" in text or "#" in text:
        raise ValueError(f"invalid semantic artifact path: {text!r}")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"semantic artifact path must stay under the job root: {text!r}")
    return path


def _require_sha256(value: str, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return digest


def validate_semantic_input_manifest(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("version") != SEMANTIC_INPUT_VERSION:
        raise ValueError("semantic input manifest must be version=1")
    source_job_id = str(payload.get("sourceJobId") or "").strip()
    output_job_id = str(payload.get("outputJobId") or "").strip()
    for label, job_id in (("sourceJobId", source_job_id), ("outputJobId", output_job_id)):
        if not job_id.startswith("job-") or len(job_id) != 36 or any(
            ch not in "0123456789abcdef" for ch in job_id[4:]
        ):
            raise ValueError(f"invalid {label}: {job_id!r}")
    category = str(payload.get("category") or "unknown object").strip()
    if not category or len(category) > 160:
        raise ValueError("category must be a non-empty string <= 160 chars")

    segmentation = payload.get("segmentation")
    if not isinstance(segmentation, dict):
        raise ValueError("semantic input requires segmentation descriptor")
    segmentation_desc = {
        "path": str(safe_job_relative_path(segmentation.get("path"))),
        "sha256": _require_sha256(segmentation.get("sha256"), "segmentation.sha256"),
    }

    images = payload.get("images")
    if not isinstance(images, dict):
        raise ValueError("semantic input requires images object")
    normalized_images = {}
    for role in ("rgbGrid", "maskGrid", "partAtlas"):
        item = images.get(role)
        if not isinstance(item, dict):
            raise ValueError(f"semantic input requires images.{role}")
        media_type = str(item.get("mediaType") or "").lower()
        if media_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError(f"unsupported images.{role}.mediaType: {media_type!r}")
        normalized_images[role] = {
            "path": str(safe_job_relative_path(item.get("path"))),
            "sha256": _require_sha256(item.get("sha256"), f"images.{role}.sha256"),
            "mediaType": media_type,
        }

    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("semantic input requires non-empty parts[]")
    normalized_parts = []
    seen_ids = set()
    seen_colors = set()
    for raw in parts:
        if not isinstance(raw, dict):
            raise ValueError("semantic input part must be an object")
        part_id = str(raw.get("id") or "").strip()
        color = str(raw.get("maskColor") or "").strip()
        if not part_id or len(part_id) > 80 or part_id in seen_ids:
            raise ValueError(f"invalid or duplicate semantic part id: {part_id!r}")
        if not color or len(color) > 80 or color in seen_colors:
            raise ValueError(f"invalid or duplicate mask color: {color!r}")
        seen_ids.add(part_id)
        seen_colors.add(color)
        normalized_parts.append({"id": part_id, "maskColor": color})

    return {
        "version": SEMANTIC_INPUT_VERSION,
        "sourceJobId": source_job_id,
        "outputJobId": output_job_id,
        "category": category,
        "segmentation": segmentation_desc,
        "images": normalized_images,
        "parts": normalized_parts,
    }


def validate_semantic_response(payload: dict, expected_parts: list[dict]) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"parts"}:
        raise ValueError("semantic response must contain exactly one top-level key: parts")
    raw_parts = payload.get("parts")
    if not isinstance(raw_parts, list):
        raise ValueError("semantic response parts must be a list")
    expected = {item["id"]: item["maskColor"] for item in expected_parts}
    if len(raw_parts) != len(expected):
        raise ValueError(f"semantic response part count mismatch: {len(raw_parts)} != {len(expected)}")
    normalized = []
    seen = set()
    forbidden = {
        "joint",
        "joint_type",
        "axis",
        "anchors",
        "limits",
        "motor",
        "actions",
        "pickup_verified",
        "runtime_verified",
    }
    for raw in raw_parts:
        if not isinstance(raw, dict):
            raise ValueError("semantic response part must be an object")
        bad = forbidden.intersection(raw)
        if bad:
            raise ValueError(f"semantic response contains forbidden executable fields: {sorted(bad)}")
        allowed = {
            "id",
            "mask_color",
            "part_name",
            "graspable",
            "grasp_scenarios",
            "functional_labels",
            "semantic_description",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"semantic response contains unknown fields: {sorted(unknown)}")
        raw_part_id = raw.get("id")
        if isinstance(raw_part_id, bool):
            raise ValueError(f"semantic response has invalid boolean part id: {raw_part_id!r}")
        if isinstance(raw_part_id, int):
            part_id = str(raw_part_id)
        elif isinstance(raw_part_id, str):
            part_id = raw_part_id.strip()
        else:
            raise ValueError(f"semantic response has invalid part id type: {type(raw_part_id).__name__}")
        if part_id not in expected or part_id in seen:
            raise ValueError(f"semantic response has invalid or duplicate part id: {part_id!r}")
        seen.add(part_id)
        mask_color = str(raw.get("mask_color") or "").strip()
        if mask_color != expected[part_id]:
            raise ValueError(
                f"semantic response mask color mismatch for {part_id}: {mask_color!r} != {expected[part_id]!r}"
            )
        part_name = str(raw.get("part_name") or "").strip()
        description = str(raw.get("semantic_description") or "").strip()
        if not part_name or len(part_name) > 160:
            raise ValueError(f"invalid part_name for {part_id}")
        if not description or len(description) > 800:
            raise ValueError(f"invalid semantic_description for {part_id}")
        graspable = raw.get("graspable")
        if not isinstance(graspable, bool):
            raise ValueError(f"graspable must be boolean for {part_id}")
        scenarios = raw.get("grasp_scenarios")
        if scenarios is None and graspable is False:
            scenarios = []
        if not isinstance(scenarios, list) or len(scenarios) > 8:
            raise ValueError(f"invalid grasp_scenarios for {part_id}")
        if graspable and not scenarios:
            raise ValueError(f"graspable part {part_id} requires at least one grasp scenario")
        normalized_scenarios = []
        for scenario in scenarios:
            if not isinstance(scenario, dict) or set(scenario) != {"scenario", "confidence"}:
                raise ValueError(f"invalid grasp scenario schema for {part_id}")
            text = str(scenario.get("scenario") or "").strip()
            confidence = scenario.get("confidence")
            if not text or len(text) > 280:
                raise ValueError(f"invalid grasp scenario text for {part_id}")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"invalid grasp scenario confidence for {part_id}")
            normalized_scenarios.append({"scenario": text, "confidence": float(confidence)})
        if not graspable and normalized_scenarios:
            raise ValueError(f"non-graspable part {part_id} must not contain grasp_scenarios")
        labels = raw.get("functional_labels")
        if not isinstance(labels, list) or not 1 <= len(labels) <= 8:
            raise ValueError(f"functional_labels must contain 1..8 items for {part_id}")
        normalized_labels = []
        for label in labels:
            text = str(label or "").strip()
            if not text or len(text) > 160:
                raise ValueError(f"invalid functional label for {part_id}")
            normalized_labels.append(text)
        normalized.append(
            {
                "id": part_id,
                "mask_color": mask_color,
                "part_name": part_name,
                "graspable": graspable,
                "grasp_scenarios": normalized_scenarios,
                "functional_labels": normalized_labels,
                "semantic_description": description,
            }
        )
    if seen != set(expected):
        raise ValueError("semantic response does not cover all expected parts")
    return sorted(normalized, key=lambda item: item["id"])


def _data_url(path: Path, media_type: str) -> str:
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"semantic image size out of bounds: {path.stat().st_size}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _semantic_client():
    from openai import AzureOpenAI, OpenAI

    endpoint = os.environ["ENDPOINT"].strip()
    api_key = os.environ["API_KEY"]
    model_name = os.environ["MODEL_NAME"].strip()
    api_version = os.environ.get("API_VERSION", "").strip()
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("ENDPOINT must be an absolute HTTP(S) URL")
    if not model_name or len(model_name) > 200:
        raise ValueError("MODEL_NAME must be non-empty")
    if api_version:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=120,
            max_retries=0,
        )
        style = "azure-openai"
    else:
        client = OpenAI(base_url=endpoint, api_key=api_key, timeout=120, max_retries=0)
        style = "openai-compatible"
    return client, style, model_name


@app.function(
    image=semantic_image,
    secrets=[semantic_secret],
    volumes={"/artifacts": artifacts},
    cpu=2.0,
    memory=4096,
    timeout=15 * 60,
    max_containers=4,
    scaledown_window=10,
)
def annotate_semantics(job_id: str, input_path: str = str(SEMANTIC_INPUT_PATH)) -> dict:
    """Annotate immutable semantic inputs. Safe to retry without rerunning geometry stages."""
    if not job_id.startswith("job-") or len(job_id) != 36:
        raise ValueError("invalid API job id")
    relative_input = safe_job_relative_path(input_path)
    artifacts.reload()
    root = JOB_ROOT / job_id
    manifest_path = root / relative_input
    if not manifest_path.is_file():
        raise FileNotFoundError(f"semantic input manifest is missing: {relative_input}")
    manifest_sha256 = sha256_file(manifest_path)
    manifest = validate_semantic_input_manifest(json.loads(manifest_path.read_text()))
    if manifest["outputJobId"] != job_id:
        raise ValueError("semantic input outputJobId does not match target job")

    segmentation_path = root / safe_job_relative_path(manifest["segmentation"]["path"])
    if not segmentation_path.is_file() or sha256_file(segmentation_path) != manifest["segmentation"]["sha256"]:
        raise RuntimeError("semantic input segmentation hash mismatch")
    image_paths = {}
    for role, descriptor in manifest["images"].items():
        path = root / safe_job_relative_path(descriptor["path"])
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"semantic input {role} hash mismatch")
        image_paths[role] = path

    client, api_style, model_name = _semantic_client()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        category=manifest["category"],
        parts=", ".join(f"{item['id']} -> {item['maskColor']}" for item in manifest["parts"]),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(
                            image_paths["rgbGrid"], manifest["images"]["rgbGrid"]["mediaType"]
                        )
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(
                            image_paths["maskGrid"], manifest["images"]["maskGrid"]["mediaType"]
                        )
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(
                            image_paths["partAtlas"], manifest["images"]["partAtlas"]["mediaType"]
                        )
                    },
                },
            ],
        },
    ]
    last_error = None
    normalized_parts = None
    request_ids = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        kwargs = {"model": model_name, "messages": messages}
        if "gpt-5" in model_name.lower() or "gpt5" in model_name.lower():
            kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        else:
            kwargs.update(temperature=0.1, max_tokens=MAX_COMPLETION_TOKENS)
        response = client.chat.completions.create(**kwargs)
        request_id = getattr(response, "_request_id", None)
        if request_id:
            request_ids.append(str(request_id)[:200])
        content = response.choices[0].message.content
        try:
            parsed = json.loads(str(content).strip())
            normalized_parts = validate_semantic_response(parsed, manifest["parts"])
            last_error = None
            break
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = str(exc)[:800]
            if attempt == MAX_ATTEMPTS:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON failed strict validation: "
                        f"{last_error}. Return a corrected full JSON object only."
                    ),
                }
            )
    if normalized_parts is None:
        raise RuntimeError(f"semantic response failed validation after {MAX_ATTEMPTS} attempts: {last_error}")

    output = {
        "version": SEMANTIC_OUTPUT_VERSION,
        "source": "embodiedgen/gpt-part-semantics",
        "profile": SEMANTIC_PROFILE,
        "sourceJobId": manifest["sourceJobId"],
        "outputJobId": job_id,
        "input": {
            "manifestSha256": manifest_sha256,
            "segmentationSha256": manifest["segmentation"]["sha256"],
            "rgbGridSha256": manifest["images"]["rgbGrid"]["sha256"],
            "maskGridSha256": manifest["images"]["maskGrid"]["sha256"],
            "partAtlasSha256": manifest["images"]["partAtlas"]["sha256"],
        },
        "provenance": {
            "apiStyle": api_style,
            "model": model_name,
            "promptRevision": PROMPT_REVISION,
            "requestIds": request_ids,
            "attempts": len(request_ids) if request_ids else 1,
        },
        "parts": normalized_parts,
    }
    output_path = root / SEMANTIC_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    validation = {
        "job_id": job_id,
        "profile": SEMANTIC_PROFILE,
        "part_count": len(normalized_parts),
        "prompt_revision": PROMPT_REVISION,
        "output_sha256": sha256_file(output_path),
        "result": "AFFORDANCE_PART_SEMANTICS_OK",
    }
    validation_path = root / SEMANTIC_VALIDATION_PATH
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    artifacts.commit()
    print("AFFORDANCE_PART_SEMANTICS_OK", json.dumps(validation), flush=True)
    return validation
