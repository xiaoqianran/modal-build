# ruff: noqa: ISC004
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

APP_NAME = "modal-build-glm53-b300-precompile"
MODEL_VOLUME_NAME = "glm53-flash-dflash2-models"
CACHE_VOLUME_NAME = "glm53-flash-compile-cache-v1"

MODEL_MOUNT = Path("/models")
TARGET_DIR = MODEL_MOUNT / "target"
DRAFTER_DIR = MODEL_MOUNT / "drafter"
CACHE_DIR = Path("/compile-cache")
OVERLAY_DIR = Path("/opt/glm53-overlay")

VLLM_PYTHON = "/usr/bin/python3"
VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SERVER_PORT = 8000

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
compile_cache = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("vllm/vllm-openai:glm53-flash", add_python="3.12")
    .entrypoint([])
    .env(
        {
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            "VLLM_LOG_STATS_INTERVAL": "1",
            "VLLM_CACHE_ROOT": "/compile-cache/vllm",
            "TILELANG_CACHE_DIR": "/compile-cache/tilelang",
            "TRITON_CACHE_DIR": "/compile-cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/compile-cache/torchinductor",
            "CUDA_CACHE_PATH": "/compile-cache/nv",
            "OMP_NUM_THREADS": "4",
            "FLASHINFER_WORKSPACE_BASE": "/compile-cache/flashinfer-workspace",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_dir(
        Path(__file__).parents[1] / "overlay",
        OVERLAY_DIR,
        copy=False,
    )
)


def _run_vllm_python(*args: str) -> None:
    cmd = [VLLM_PYTHON, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _prepare_runtime() -> None:
    qwen2_dst = VLLM_ROOT / "model_executor/models/qwen3_dflash2.py"
    if not qwen2_dst.exists():
        shutil.copy2(OVERLAY_DIR / "qwen3_dflash2.py", qwen2_dst)

    dflash_dst = VLLM_ROOT / "v1/worker/gpu/spec_decode/dflash2"
    if not (dflash_dst / "speculator.py").exists():
        dflash_dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OVERLAY_DIR / "dflash2", dflash_dst, dirs_exist_ok=True)

    for patch in (
        "patch_registry_and_select.py",
        "patch_glm_aux_capture.py",
        "patch_kv_page_lcm2.py",
        "patch_glm5_drafter_group.py",
    ):
        _run_vllm_python(str(OVERLAY_DIR / patch))

    _run_vllm_python(
        "-c",
        (
            "from vllm.model_executor.models.registry import ModelRegistry; "
            'assert "DFlash2DraftModel" in ModelRegistry.get_supported_archs(); '
            'print("DFlash2 overlay registry check OK")'
        ),
    )
    _run_vllm_python(str(OVERLAY_DIR / "sim_glm5_drafter_hades.py"))


def _wait_until_ready(process: subprocess.Popen, timeout_s: int = 45 * 60) -> None:
    url = f"http://127.0.0.1:{SERVER_PORT}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM exited before readiness (exit={code})")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("timed out waiting for vLLM readiness")


def _chat(prompt: str, *, max_tokens: int) -> dict:
    payload = json.dumps(
        {
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning_effort": "low",
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15 * 60) as response:
        return json.loads(response.read().decode("utf-8"))


@app.function(
    image=image,
    gpu="B300",
    cpu=16,
    memory=(32768, 65536),
    volumes={
        str(MODEL_MOUNT): models.with_mount_options(read_only=True),
        str(CACHE_DIR): compile_cache,
    },
    timeout=60 * 60,
    max_containers=1,
)
def precompile() -> dict:
    """One-time B300 cache generation for the pinned GLM53 runtime."""
    _prepare_runtime()

    speculative = json.dumps(
        {
            "method": "dflash",
            "model": str(DRAFTER_DIR),
            "num_speculative_tokens": 7,
        },
        separators=(",", ":"),
    )
    cmd = [
        "vllm",
        "serve",
        str(TARGET_DIR),
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--tensor-parallel-size",
        "1",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        "1000000",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.92",
        "--speculative-config",
        speculative,
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm47",
        "--enable-auto-tool-choice",
        "--served-model-name",
        "glm-5.3-flash",
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0}, separators=(",", ":")),
    ]

    started = time.monotonic()
    process = subprocess.Popen(cmd, env=os.environ.copy())
    try:
        _wait_until_ready(process)
        ready_s = time.monotonic() - started

        # Cover the common speculative-decoding/rejection-sampler shapes.
        warmups = [
            ("Reply with exactly: OK", 16),
            ("Write a short Python function that adds two integers.", 64),
            (
                "Return a compact JSON object with keys status, language, and "
                "result. status must be ok; language must be python.",
                128,
            ),
            (
                "Inspect this pseudo-code and state one bug, then provide a "
                "corrected 5-line implementation: def f(xs): return xs[1]",
                256,
            ),
        ]
        outputs = []
        for prompt, max_tokens in warmups:
            t0 = time.monotonic()
            response = _chat(prompt, max_tokens=max_tokens)
            outputs.append(
                {
                    "max_tokens": max_tokens,
                    "elapsed_s": round(time.monotonic() - t0, 3),
                    "completion_tokens": response.get("usage", {}).get(
                        "completion_tokens"
                    ),
                }
            )

        marker = {
            "tag": (
                "glm53-flash-dflash2-vllm0281rc1-fi0618-"
                "sm103a-b300-cache-v1"
            ),
            "ready_s": round(ready_s, 3),
            "warmups": outputs,
            "created_at_unix": int(time.time()),
        }
        (CACHE_DIR / ".glm53-precompile.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Explicit persistence is mandatory: the release builder reads this
        # Volume from a later CPU-only container.
        compile_cache.commit()
        return marker
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)





