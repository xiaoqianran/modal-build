import os

APP_NAME = "comfyui-hyworld2"
GPU_ARCHITECTURES = {
    "L4": "8.9",
    "L40S": "8.9",
    "RTX-PRO-6000": "12.0",
}
GPU = os.environ.get("MODAL_GPU", "RTX-PRO-6000")
if GPU not in GPU_ARCHITECTURES:
    allowed = ", ".join(GPU_ARCHITECTURES)
    raise ValueError(f"Unsupported MODAL_GPU={GPU!r}; allowed: {allowed}")

CUDA_ARCH = GPU_ARCHITECTURES[GPU]
SCALEDOWN_WINDOW_SECONDS = 60
MAX_SESSION_SECONDS = 2 * 60 * 60
