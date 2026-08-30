import os
import pathlib
import shutil
import subprocess

import modal

APP_NAME = "comfyui-hyworld2"
GPU = os.environ.get("MODAL_GPU", "H100")
CUDA_ARCH = {"H100": "9.0", "H200": "9.0", "A100": "8.0", "A10G": "8.6", "L40S": "8.9", "L4": "8.9"}.get(GPU)
COMFYUI_DIR = pathlib.Path("/opt/ComfyUI")
DATA_DIR = pathlib.Path("/data")
PORT = 8188

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("comfyui-hyworld2-data", create_if_missing=True)

# HYWorld2 的原生扩展需要 nvcc，并且仓库安装脚本会执行 CUDA smoke test。
# 因此使用 CUDA 13 devel 镜像，并让原生扩展安装步骤在带 GPU 的 image build 中执行。
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "git-lfs",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libx11-6",
        "ninja-build",
        "pkg-config",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging",
        # 与仓库现有 cu130 / pt291 构建目标保持一致，Linux 下原生扩展会重新编译。
        "python -m pip install --index-url https://download.pytorch.org/whl/cu130 "
        "torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1",
        "git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI",
        "python -m pip install -r /opt/ComfyUI/requirements.txt",
        "git clone --depth=1 https://github.com/xiaoqianran/ComfyUI_HYWorld2.git "
        "/opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2",
        "python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/requirements.txt",
        "python -m pip check",
    )
    .run_commands(
        # 仓库会把内置的 Windows wheel 误判成 Linux 可用，先删除后强制源码构建。
        "find /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/gsplat -maxdepth 1 "
        "-type f -name \"*win_amd64.whl\" -delete",
        "rm -rf /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/hyworld2/worldgen/third_party/gsplat_maskgaussian/gsplat/cuda/csrc/third_party/glm/doc",
    )
    .run_commands(
        "cd /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2 && "
        f"CC=gcc CXX=g++ CUDAHOSTCXX=g++ TORCH_CUDA_ARCH_LIST={CUDA_ARCH} PIP_VERBOSE=1 MAX_JOBS=8 python scripts/build_gsplat.py --gsplat-only",
        gpu=GPU,
    )
    .run_commands(
        "cd /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2 && "
        "CC=gcc CXX=g++ CUDAHOSTCXX=g++ MAX_JOBS=8 python scripts/build_gsplat.py --fused-ssim-only",
        gpu=GPU,
    )
    .run_commands(
        "cd /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2 && "
        "python -m pip install pybind11 && "
        "git clone --depth 1 https://github.com/recastnavigation/recastnavigation.git hyworld2/worldgen/third_party/recastnavigation && "
        "RECAST_PATH=/opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/hyworld2/worldgen/third_party/recastnavigation "
        "CC=gcc CXX=g++ python -m pip wheel --no-deps --no-build-isolation "
        "-w /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/gsplat "
        "/opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/hyworld2/worldgen/third_party/navmesh && "
        "python -m pip install --force-reinstall /opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/gsplat/recast-*.whl && "
        "python -c \"import recast; print('recast import OK')\"",
    )
    .add_local_file(
        "/workspace/hyworld2-modal/pytorch3d_nopulsar_build.py",
        "/opt/pytorch3d_nopulsar_build.py",
        copy=True,
    )
    .run_commands(
        f"CC=gcc CXX=g++ CUDAHOSTCXX=g++ TORCH_CUDA_ARCH_LIST={CUDA_ARCH} "
        "PYTORCH3D_NO_NINJA=0 MAX_JOBS=8 python /opt/pytorch3d_nopulsar_build.py",
        gpu=GPU,
    )
    .run_commands(
        "python -m pip install iopath",
        "python -m pip check",
        "python - <<'PY_CHECK'\n"
        "import torch\n"
        "print('torch:', torch.__version__, 'cuda:', torch.version.cuda)\n"
        "import gsplat, pytorch3d, recast, fused_ssim\n"
        "print('HYWorld2 native modules import OK')\n"
        "PY_CHECK",
    )
    .run_commands(
        "python -m pip install 'diffusers==0.36.0' 'transformers==5.2.0'",
        "python -m pip check",
    )
    .run_commands(
        "python -m pip install 'torchao==0.15.0'",
        "python -m pip check",
        "python - <<'PY_DEP_CHECK'\n"
        "import torch, torchao, diffusers, transformers\n"
        "import diffusers.pipelines.pipeline_utils\n"
        "print('dependency smoke OK', torch.__version__, torchao.__version__, diffusers.__version__, transformers.__version__)\n"
        "PY_DEP_CHECK",
    )
    .add_local_file(
        "/workspace/hyworld2-modal/runtime_patch.py",
        "/opt/runtime_patch.py",
        copy=True,
    )
    .run_commands("python /opt/runtime_patch.py")
    .run_commands(
        "python -m pip install 'peft==0.18.1' 'timm==1.0.11' 'ftfy' 'loguru==0.7.3' 'matplotlib==3.10.3'",
        "python -m pip check",
        "python - <<'PY_WS_IMPORT'\n"
        "import sys\n"
        "sys.path.insert(0, '/opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2/worldstereo')\n"
        "from models.worldstereo_wrapper import WorldStereo\n"
        "print('WorldStereo import smoke OK', WorldStereo)\n"
        "PY_WS_IMPORT",
    )
    .add_local_file(
        "/workspace/hyworld2-modal/video_output_node.py",
        "/opt/ComfyUI/custom_nodes/HYWorld2_Modal_Video/__init__.py",
        copy=True,
    )
)



@app.function(image=image, gpu=GPU, timeout=60 * 60)
def build_probe():
    import torch
    import gsplat
    import fused_ssim
    import recast
    import pytorch3d
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "native_modules": ["gsplat", "fused_ssim", "recast", "pytorch3d"],
    }

def _persist_dir(name: str) -> None:
    """把 ComfyUI 可变目录迁移到 /data，并用软链接接回去。"""
    src = COMFYUI_DIR / name
    dst = DATA_DIR / name
    dst.mkdir(parents=True, exist_ok=True)

    if src.is_symlink():
        return

    if src.exists() and src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        shutil.rmtree(src)
    elif src.exists():
        src.unlink()

    src.symlink_to(dst, target_is_directory=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_volume},
    timeout=24 * 60 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
    env={
        "HF_HOME": "/data/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data/huggingface/hub",
        "TORCH_HOME": "/data/torch",
        "XDG_CACHE_HOME": "/data/cache",
        "PYTHONUNBUFFERED": "1",
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=PORT, startup_timeout=300)
def comfyui():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dirname in ("models", "input", "output", "user"):
        _persist_dir(dirname)

    for cache_dir in (
        DATA_DIR / "huggingface",
        DATA_DIR / "torch",
        DATA_DIR / "cache",
    ):
        cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "main.py",
        "--listen",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    print("Starting:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, cwd=str(COMFYUI_DIR), env=os.environ.copy())


@app.function(image=image, gpu=GPU, timeout=20 * 60)
def import_probe():
    import subprocess
    import sys
    import textwrap
    import time

    modules = [
        "world_mirror_v1",
        "world_mirror_v2",
        "panorama_mapper",
        "world_stereo",
        "splat_patch_poc",
        "worldgen",
        "hyworld2_native",
        "qwen_pano",
        "utils",
    ]
    results = {}
    for name in modules:
        print(f"PROBE START {name}", flush=True)
        code = textwrap.dedent(f"""
            import importlib, sys, types, time
            repo = '/opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2'
            comfy = '/opt/ComfyUI'
            sys.path.insert(0, repo)
            sys.path.insert(0, comfy)
            pkg = types.ModuleType('hywcustom')
            pkg.__path__ = [repo]
            sys.modules['hywcustom'] = pkg
            npkg = types.ModuleType('hywcustom.nodes')
            npkg.__path__ = [repo + '/nodes']
            sys.modules['hywcustom.nodes'] = npkg
            t = time.time()
            m = importlib.import_module('hywcustom.nodes.{name}')
            print('IMPORTED {name}', round(time.time()-t, 3), len(getattr(m, 'NODE_CLASS_MAPPINGS', {{}})))
        """)
        t0 = time.time()
        try:
            cp = subprocess.run(
                [sys.executable, "-c", code],
                cwd="/opt/ComfyUI",
                text=True,
                capture_output=True,
                timeout=120,
            )
            results[name] = {
                "returncode": cp.returncode,
                "seconds": round(time.time() - t0, 3),
                "stdout": cp.stdout[-4000:],
                "stderr": cp.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as e:
            results[name] = {
                "timeout": True,
                "seconds": round(time.time() - t0, 3),
                "stdout": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "")[-4000:] if isinstance(e.stderr, str) else "",
            }
        print(f"PROBE END {name}: {results[name]}", flush=True)
    return results


@app.function(image=image, timeout=10 * 60)
def dep_probe():
    import importlib.metadata as md
    import os
    from pathlib import Path

    names = [
        "torch", "torchvision", "diffusers", "transformers", "accelerate",
        "torchao", "torchcodec", "optimum", "peft", "safetensors",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    refs = []
    root = Path("/usr/local/lib/python3.12/site-packages")
    for p in root.rglob("*.py"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if "ScalingType" in txt:
            lines = [f"{i+1}:{line}" for i, line in enumerate(txt.splitlines()) if "ScalingType" in line]
            refs.append((str(p), lines[:20]))
    try:
        import diffusers.pipelines.pipeline_utils
        diffusers_import = "ok"
    except Exception as e:
        import traceback
        diffusers_import = traceback.format_exc()
    return {"versions": versions, "refs": refs[:100], "diffusers_import": diffusers_import}


@app.function(image=image, timeout=5 * 60)
def dep_probe_fast():
    import importlib.metadata as md
    import subprocess
    names = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "torchao", "torchcodec", "peft"]
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    print("DEP_VERSIONS", versions, flush=True)
    cmd = [
        "grep", "-RIn", "ScalingType",
        "/usr/local/lib/python3.12/site-packages/diffusers",
        "/usr/local/lib/python3.12/site-packages/transformers",
        "/usr/local/lib/python3.12/site-packages/torchvision",
        "/usr/local/lib/python3.12/site-packages/torchao",
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    print("SCALING_REFS", cp.stdout[-12000:], flush=True)
    try:
        import diffusers.pipelines.pipeline_utils
        print("DIFFUSERS_IMPORT_OK", flush=True)
    except Exception:
        import traceback
        print("DIFFUSERS_IMPORT_FAIL\n" + traceback.format_exc(), flush=True)
    return versions
