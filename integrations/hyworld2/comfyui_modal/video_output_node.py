import os
import re
import subprocess
import time
import uuid

import folder_paths
import torch


class HYWorld2SaveMP4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "filename_prefix": ("STRING", {"default": "HYWorld2_API"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 40}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "HYWorld2/Modal"

    def save(self, images, fps=8.0, filename_prefix="HYWorld2_API", crf=18):
        frames = images.detach().clamp(0, 1).mul(255).to("cpu", dtype=torch.uint8).numpy()
        if frames.ndim == 3:
            frames = frames[None, ...]
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError(f"Expected IMAGE batch [N,H,W,C], got {frames.shape}")
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        frames = frames.copy(order="C")
        n, h, w, _ = frames.shape
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(filename_prefix)).strip("._") or "HYWorld2_API"
        subfolder = "hyworld2_api"
        out_dir = os.path.join(folder_paths.get_output_directory(), subfolder)
        os.makedirs(out_dir, exist_ok=True)
        name = f"{safe}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        path = os.path.join(out_dir, name)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(float(fps)), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
        ]
        proc = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[-4000:]}")
        rel = f"{subfolder}/{name}"
        print(f"[HYWorld2 Modal] Saved MP4: {path} ({n} frames, {w}x{h} @ {fps} fps)", flush=True)
        return {
            "ui": {
                "text": [rel],
                "videos": [{"filename": name, "subfolder": subfolder, "type": "output"}],
            },
            "result": (rel,),
        }


NODE_CLASS_MAPPINGS = {"HYWorld2_SaveMP4": HYWorld2SaveMP4}
NODE_DISPLAY_NAME_MAPPINGS = {"HYWorld2_SaveMP4": "HYWorld2 Save MP4 (Modal)"}
