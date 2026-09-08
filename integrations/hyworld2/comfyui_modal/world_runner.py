from __future__ import annotations

import json
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def build_prompt(
    *, image_name: str, filename: str, target_size: int, use_gsplat: bool
) -> dict[str, dict]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "VNCCS_LoadWorldMirrorV2Model",
            "inputs": {"device": "cuda", "precision": "bf16"},
        },
        "3": {
            "class_type": "VNCCS_WorldMirrorV2_3D",
            "inputs": {
                "model": ["2", 0],
                "images": ["1", 0],
                "use_gsplat": use_gsplat,
                "target_size": target_size,
                "low_vram_mode": False,
                "head_frame_chunk_size": 1,
                "head_compute_mode": "depth+gs",
                "gs_param_chunk_size": 1,
                "transformer_mlp_chunk_size": 8192,
                "filter_edges": True,
                "voxel_prune_splats": True,
                "debug_log": False,
            },
        },
        "4": {
            "class_type": "VNCCS_SavePLY",
            "inputs": {"ply_data": ["3", 0], "filename": filename},
        },
    }


class ComfyClient:
    def __init__(self, port: int, output_dir: pathlib.Path) -> None:
        self.base_url = f"http://127.0.0.1:{port}"
        self.output_dir = output_dir

    def wait_until_ready(self, process: subprocess.Popen, timeout: int = 180) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._get_json("/system_stats", timeout=2)
                return
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError(f"ComfyUI exited with code {process.returncode}")
                time.sleep(1)
        raise TimeoutError("ComfyUI did not become ready")

    def run(self, prompt: dict[str, dict], timeout: int) -> dict:
        filename = str(prompt["4"]["inputs"]["filename"])
        client_id = uuid.uuid4().hex
        result = self._post_json("/prompt", {"prompt": prompt, "client_id": client_id})
        prompt_id = result["prompt_id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self._get_json(f"/history/{prompt_id}")
            if prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False))
                if status.get("completed"):
                    return self._result(prompt_id, item, filename)
            time.sleep(2)
        raise TimeoutError(f"workflow {prompt_id} exceeded {timeout}s")

    def _result(self, prompt_id: str, history: dict, filename: str) -> dict:
        candidates = sorted(
            self.output_dir.glob(f"{filename}_*.ply"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            diagnostic = json.dumps(history, ensure_ascii=False, default=str)
            raise RuntimeError(
                f"workflow {prompt_id} completed without a PLY output; history={diagnostic}"
            )
        path = candidates[0]
        return {
            "prompt_id": prompt_id,
            "path": str(path),
            "bytes": path.stat().st_size,
            "status": history.get("status", {}),
        }

    def _get_json(self, path: str, timeout: int = 10) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=timeout) as response:
            return json.load(response)

    def _post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            raise RuntimeError(f"ComfyUI {error.code}: {body}") from error
