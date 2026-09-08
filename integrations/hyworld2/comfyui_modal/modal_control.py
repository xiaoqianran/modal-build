import pathlib

import modal

from modal_config import APP_NAME, CUDA_ARCH, GPU

LOCAL_DIR = pathlib.Path(__file__).parent
app = modal.App(f"{APP_NAME}-control")
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    LOCAL_DIR / "modal_config.py", "/root/modal_config.py", copy=True
)


@app.function(image=image, timeout=60, min_containers=0, scaledown_window=2)
def config_probe():
    result = {
        "app": APP_NAME,
        "gpu": GPU,
        "cuda_arch": CUDA_ARCH,
        "min_containers": 0,
    }
    print(result, flush=True)
    return result


@app.function(
    image=image,
    gpu=GPU,
    timeout=60,
    min_containers=0,
    scaledown_window=2,
    max_containers=1,
)
def gpu_probe():
    import subprocess

    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    result = {"requested": GPU, "detected": name, "cuda_arch": CUDA_ARCH}
    print(result, flush=True)
    return result
