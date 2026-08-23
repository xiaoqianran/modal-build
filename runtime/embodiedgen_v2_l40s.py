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
APP_NAME = "embodiedgen-v2-l40s-consumer"

app = modal.App(APP_NAME)
data = modal.Volume.from_name("embodiedgen-v2-data", create_if_missing=True)

ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "HF_HOME": "/data/hf",
    "MODELSCOPE_CACHE": "/data/modelscope",
    "TORCH_HOME": "/data/torch",
    "U2NET_HOME": "/data/u2net",
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
        "git clone --depth 1 --branch v2.0.0 https://github.com/HorizonRobotics/EmbodiedGen.git /workspace/EmbodiedGen",
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
        "python -m pip install --no-deps 'clip@git+https://github.com/openai/CLIP.git'",
        "python -m pip install --no-deps 'segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f'",
        "python -m pip install --no-deps 'kolors@git+https://github.com/HochCC/Kolors.git'",
        "python -m pip install --no-deps 'MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734'",
        "python -m pip install plyfile moderngl glcontext ftfy fvcore iopath",
        "python -m pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 'pillow<12'",
        "python -m pip install --no-deps kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html",
        "python -m pip install pygltflib warp-lang usd-core ipycanvas ipyevents 'jupyter_client<8' tornado",
        "python -m pip install --no-deps gsplat==1.5.3",
    )
    # Consume release artifacts: no source builds.
    .run_commands(
        f"mkdir -p /opt/embodiedgen-release/wheels /root/.cache/torch_extensions && curl -fL '{RELEASE}/{TAG}.wheels.zip' -o /tmp/wheels.zip",
        f"curl -fL '{RELEASE}/{TAG}.torch-extensions.zip' -o /tmp/ext.zip",
        "unzip -q /tmp/wheels.zip -d /opt/embodiedgen-release/wheels",
        "unzip -q /tmp/ext.zip -d /root/.cache/torch_extensions",
        "python -m pip install --no-deps /opt/embodiedgen-release/wheels/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl /opt/embodiedgen-release/wheels/nvdiffrast-0.3.3-py3-none-any.whl",
        "rm -f /tmp/wheels.zip /tmp/ext.zip",
    )
    # Replace JIT loaders with direct .so loaders. On this image nvcc does not exist,
    # so a cache miss is a hard failure rather than an accidental expensive compile.
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import gsplat.cuda._backend as b\n"
        "p=Path(b.__file__)\n"
        "p.write_text('''import importlib.util, pathlib, sys\n"
        "so=pathlib.Path('/root/.cache/torch_extensions/py310_cu126/gsplat_cuda/gsplat_cuda.so')\n"
        "if not so.exists(): raise ImportError(f'missing precompiled gsplat extension: {so}')\n"
        "spec=importlib.util.spec_from_file_location('gsplat_cuda', so)\n"
        "_C=importlib.util.module_from_spec(spec); sys.modules['gsplat_cuda']=_C; spec.loader.exec_module(_C)\n"
        "__all__=['_C']\n''')\n"
        "print('patched gsplat direct loader', p)\n"
        "PY",
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import nvdiffrast.torch.ops as ops\n"
        "p=Path(ops.__file__); s=p.read_text()\n"
        "old='''    # Compile and load.\n    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]\n    torch.utils.cpp_extension.load(name=plugin_name, sources=source_paths, extra_cflags=common_opts+cc_opts, extra_cuda_cflags=common_opts+['-lineinfo'], extra_ldflags=ldflags, with_cuda=True, verbose=False)\n\n    # Import, cache, and return the compiled module.\n    _cached_plugin[gl] = importlib.import_module(plugin_name)\n'''\n"
        "new='''    # Release-consumer runtime: direct-load precompiled CUDA plugin; never JIT compile.\n    if gl:\n        raise RuntimeError('nvdiffrast GL plugin is not shipped in the EmbodiedGen consumer release')\n    import importlib.util, sys\n    so = '/root/.cache/torch_extensions/py310_cu126/nvdiffrast_plugin/nvdiffrast_plugin.so'\n    if not os.path.exists(so):\n        raise ImportError(f'missing precompiled nvdiffrast plugin: {so}')\n    spec = importlib.util.spec_from_file_location(plugin_name, so)\n    module = importlib.util.module_from_spec(spec)\n    sys.modules[plugin_name] = module\n    spec.loader.exec_module(module)\n    _cached_plugin[gl] = module\n'''\n"
        "if old not in s: raise SystemExit('nvdiffrast compile block not found')\n"
        "p.write_text(s.replace(old,new,1)); print('patched nvdiffrast direct loader', p)\n"
        "PY",
    )
    .workdir("/workspace/EmbodiedGen")
)

# Apply only the validated headless/source patches after all packages are installed.
image = (
    image
    .add_local_file("patches/embodiedgen-v2.0.0/headless-l40s.patch", "/tmp/headless-l40s.patch", copy=True)
    .add_local_file("patches/embodiedgen-v2.0.0/modal_postprocess.py", "/workspace/EmbodiedGen/embodied_gen/scripts/modal_postprocess.py", copy=True)
    .add_local_file("patches/embodiedgen-v2.0.0/inference_sam3d_only.py", "/workspace/EmbodiedGen/embodied_gen/utils/inference.py", copy=True)
    .run_commands(
        "cd /workspace/EmbodiedGen && git apply /tmp/headless-l40s.patch",
        "cd /workspace/EmbodiedGen && grep -RIl '@spaces.GPU' embodied_gen --include='*.py' | xargs -r sed -i '/^[[:space:]]*@spaces.GPU[[:space:]]*$/d'",
        "cd /workspace/EmbodiedGen && python -m pip install --no-deps -e .",
        "cd /workspace/EmbodiedGen && python -m py_compile embodied_gen/scripts/imageto3d.py embodied_gen/scripts/modal_postprocess.py embodied_gen/utils/inference.py",
        "! command -v nvcc",
    )
)

# Final hardening: every nvdiffrast.torch import overrides _get_plugin with a
# release-only loader. This works in the Modal parent process and every child subprocess.
image = (
    image
    .add_local_file(
        "patches/embodiedgen-v2.0.0/patch_nvdiffrast_init_release.py",
        "/tmp/patch_nvdiffrast_init_release.py",
        copy=True,
    )
    .add_local_file(
        "patches/embodiedgen-v2.0.0/gsplat_backend_release.py",
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


def _prepare_weights() -> dict:
    os.environ.update({"TORCH_HOME": "/data/torch", "U2NET_HOME": "/data/u2net"})
    target = Path("/data/weights/sam-3d-objects")
    marker = target / "checkpoints/pipeline.yaml"
    if not marker.exists():
        raise RuntimeError("SAM3D weights missing; run preload_weights first")
    return {"path": str(target), "size": subprocess.check_output(["du", "-sh", str(target)], text=True).split()[0]}


@app.function(image=image, volumes={"/data": data}, timeout=60 * 60, cpu=4.0, memory=16384)
def preload_weights():
    os.environ.update({"TORCH_HOME": "/data/torch", "U2NET_HOME": "/data/u2net"})
    u2net = Path("/data/u2net/u2net.onnx")
    if not u2net.exists():
        import urllib.request
        u2net.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", str(u2net))
    dino = Path("/data/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
    repo = Path("/data/torch/hub/facebookresearch_dinov2_main")
    if not dino.exists() or not repo.exists():
        import torch
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True, trust_repo=True)
        del m
    target = Path("/data/weights/sam-3d-objects")
    marker = target / "checkpoints/pipeline.yaml"
    if not marker.exists():
        from modelscope import snapshot_download
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download("facebook/sam-3d-objects", local_dir=str(target))
    data.commit()
    return _prepare_weights()


@app.function(
    image=image,
    gpu="L40S",
    timeout=5 * 60,
    cpu=2.0,
    memory=8192,
    min_containers=0,
    scaledown_window=10,
)
def extensions_smoke() -> dict:
    """Cheap proof that release .so files execute without nvcc/c++ available."""
    t0=time.perf_counter()
    assert subprocess.run(["bash","-lc","command -v nvcc || command -v c++"],capture_output=True).returncode != 0
    import torch
    from gsplat import rasterization
    import nvdiffrast.torch as dr

    # nvdiffrast: rasterize one clip-space triangle.
    ctx=dr.RasterizeCudaContext()
    pos=torch.tensor([[[-0.5,-0.5,0.0,1.0],[0.5,-0.5,0.0,1.0],[0.0,0.5,0.0,1.0]]],device="cuda",dtype=torch.float32)
    tri=torch.tensor([[0,1,2]],device="cuda",dtype=torch.int32)
    rast,_=dr.rasterize(ctx,pos,tri,resolution=[64,64])
    torch.cuda.synchronize()

    # gsplat: rasterize a tiny cloud.
    n=64
    means=torch.randn(n,3,device="cuda"); means[:,2].abs_().add_(3.0)
    quats=torch.randn(n,4,device="cuda"); quats=quats/quats.norm(dim=-1,keepdim=True)
    scales=torch.full((n,3),0.03,device="cuda")
    op=torch.full((n,),0.5,device="cuda")
    colors=torch.rand(n,3,device="cuda")
    view=torch.eye(4,device="cuda")[None]
    K=torch.tensor([[100.,0.,32.],[0.,100.,32.],[0.,0.,1.]],device="cuda")[None]
    rgb,alpha,_=rasterization(means,quats,scales,op,colors,view,K,64,64)
    torch.cuda.synchronize()
    out={
        "gpu":torch.cuda.get_device_name(0),
        "torch":str(torch.__version__),
        "cuda":str(torch.version.cuda),
        "cc":list(torch.cuda.get_device_capability(0)),
        "nvcc_present":False,
        "nvdiffrast_shape":list(rast.shape),
        "gsplat_shape":list(rgb.shape),
        "seconds":round(time.perf_counter()-t0,3),
    }
    print("EXTENSIONS_SMOKE_OK",json.dumps(out),flush=True)
    return out


@app.function(
    image=image, gpu="L40S", volumes={"/data": data}, timeout=60 * 60,
    cpu=6.0, memory=32768, min_containers=0, scaledown_window=300,
)
def benchmark_once(label: str = "run") -> dict:
    """One end-to-end request. Repeated calls can reuse the same warm container."""
    t0 = time.perf_counter()
    os.chdir("/workspace/EmbodiedGen")
    w = _prepare_weights()
    local = Path("weights/sam-3d-objects")
    local.parent.mkdir(exist_ok=True)
    if local.is_symlink(): local.unlink()
    if not local.exists(): local.symlink_to(Path(w["path"]), target_is_directory=True)

    # Hard proof that this is a release consumer, not a builder.
    assert subprocess.run(["bash", "-lc", "command -v nvcc"], capture_output=True).returncode != 0
    import torch
    from gsplat.cuda._backend import _C as gsplat_C
    import nvdiffrast.torch as dr
    cuda_ready = time.perf_counter()
    _ = gsplat_C
    _ = dr.RasterizeCudaContext()
    ext_ready = time.perf_counter()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"/data/benchmarks/{stamp}-{label}")
    out.mkdir(parents=True, exist_ok=True)

    cmd = ["img3d-cli", "--image_path", "apps/assets/example_image/sample_00.jpg", "--output_root", str(out), "--n_retry", "1", "--image3d_model", "SAM3D", "--keep_intermediate"]
    env = os.environ.copy(); env["MODAL_GENERATE_ONLY"] = "1"
    g0 = time.perf_counter(); subprocess.run(cmd, check=True, env=env); g1 = time.perf_counter()

    post = ["python", "-m", "embodied_gen.scripts.modal_postprocess", "--output_root", str(out), "--filename", "sample_00", "--texture_size", "1024", "--video_frames", "60", "--fast_bake", "--skip_fix_mesh", "--disable_decompose_convex", "--skip_aesthetic"]
    p0 = time.perf_counter(); subprocess.run(post, check=True); p1 = time.perf_counter()

    report = json.loads((out / "validation_report.json").read_text())
    t1 = time.perf_counter()
    result = {
        "label": label,
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvcc_present": False,
        "weights": w,
        "seconds": {
            "function_start_to_cuda_ready": round(cuda_ready - t0, 3),
            "precompiled_extensions_ready": round(ext_ready - cuda_ready, 3),
            "sam3d_generate": round(g1 - g0, 3),
            "postprocess": round(p1 - p0, 3),
            "function_total": round(t1 - t0, 3),
        },
        "validation": report.get("checks", {}),
        "output": str(out),
    }
    (out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    data.commit()
    print("BENCHMARK_RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    return result


@app.local_entrypoint()
def benchmark():
    print("CPU preload/check:", preload_weights.remote(), flush=True)
    rows = []
    for label in ("cold", "warm"):
        wall0 = time.perf_counter()
        r = benchmark_once.remote(label)
        wall = time.perf_counter() - wall0
        r["seconds"]["client_wall"] = round(wall, 3)
        rows.append(r)
        print(label.upper(), json.dumps(r, indent=2, ensure_ascii=False), flush=True)
    print("BENCHMARK_SUMMARY", json.dumps(rows, indent=2, ensure_ascii=False), flush=True)
