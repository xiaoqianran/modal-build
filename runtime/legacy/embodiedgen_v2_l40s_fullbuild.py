import os
import subprocess
from pathlib import Path
import modal

app = modal.App("embodiedgen-v2-l40s")
data = modal.Volume.from_name("embodiedgen-v2-data", create_if_missing=True)

ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "CUDA_HOME": "/usr/local/cuda",
    "TORCH_CUDA_ARCH_LIST": "8.9",
    "TCNN_CUDA_ARCHITECTURES": "89",
    "MAX_JOBS": "2",
    "CC": "gcc",
    "CXX": "g++",
    "HF_HOME": "/data/hf",
    "MODELSCOPE_CACHE": "/data/modelscope",
    "PYOPENGL_PLATFORM": "egl",
}

# All image building is CPU-only. CUDA *devel* is used only so nvcc can build
# target-SM89 extensions before we ever rent an L40S.
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .env(ENV)
    .apt_install(
        "git", "build-essential", "gcc", "g++", "cmake", "ninja-build", "ffmpeg",
        "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libegl1", "libegl1-mesa", "libgomp1", "libx11-6", "libxrandr2", "libxi6",
    )
    # Layer 1: source only.
    .run_commands(
        "git clone --depth 1 --branch v2.0.0 https://github.com/HorizonRobotics/EmbodiedGen.git /workspace/EmbodiedGen",
        "cd /workspace/EmbodiedGen && git submodule update --init --recursive --progress thirdparty/sam3d",
    )
    # Layer 2: core ABI. Never rebuild this because a later optional package fails.
    .run_commands(
        "python -m pip install --upgrade 'pip>=25' setuptools==80.10.2 wheel packaging 'Cython>=0.29.37'",
        "python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126",
        "python -m pip install xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu126",
        "python - <<'PY'\nimport torch\nprint('CORE', torch.__version__, torch.version.cuda)\nassert torch.__version__.startswith('2.8.0') and torch.version.cuda == '12.6'\nPY",
    )
    # Layer 3: EmbodiedGen's pinned runtime requirements. The original script uses
    # legacy resolver; constrain packages whose 2026 latest releases conflict with numpy 1.26.
    .run_commands(
        "printf 'numpy==1.26.4\\nopencv-python==4.11.0.86\\nopencv-python-headless==4.11.0.86\\npillow<12\\n' >/tmp/eg-constraints.txt",
        "cd /workspace/EmbodiedGen && PIP_CONSTRAINT=/tmp/eg-constraints.txt python -m pip install -r requirements.txt --use-deprecated=legacy-resolver",
    )
    # Layer 4: small git packages. --no-deps prevents 2026 transitive packages from
    # silently replacing the project's numpy pin.
    .run_commands(
        "python -m pip install --no-deps 'utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15'",
        "python -m pip install --no-deps 'clip@git+https://github.com/openai/CLIP.git'",
        "python -m pip install --no-deps 'segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f'",
        "python -m pip install --no-deps 'nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@729261d'",
        "python -m pip install --no-deps 'kolors@git+https://github.com/HochCC/Kolors.git'",
        "python -m pip install --no-deps 'MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734'",
        "python -m pip install plyfile moderngl glcontext ftfy fvcore iopath",
        "python -m pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 'pillow<12'",
    )
    # Layer 5: exact prebuilt Kaolin wheel for torch 2.8/cu126/cp310. No source build.
    .run_commands(
        "python -m pip install --no-deps kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html",
        "python -m pip install pygltflib warp-lang usd-core ipycanvas ipyevents 'jupyter_client<8' tornado",
        "python -c \"from kaolin.utils.testing import check_tensor; import kaolin; print('KAOLIN OK', kaolin.__version__)\"",
    )
    # Layer 6: gsplat's PyPI wheel ships CUDA sources; compile its JIT extension now on
    # CPU builder against SM89 so the paid L40S does not spend minutes compiling it.
    .run_commands(
        "python -m pip install --no-deps gsplat==1.5.3",
        "FAST_COMPILE=1 VERBOSE=0 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 python -c \"from gsplat.cuda._backend import _C; print('GSPLAT CUDA EXT OK', _C is not None)\"",
    )
    # Layer 7: PyTorch3D is genuinely required by SAM3D. Force an SM89 CUDA build even
    # though the image builder has no physical GPU.
    .run_commands(
        "FORCE_CUDA=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-deps --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'",
        "python -c \"import pytorch3d; from pytorch3d.transforms import Transform3d; print('PYTORCH3D OK')\"",
    )
    # Layer 8: install only the application (not .[dev], tests, mkdocs, pre-commit).
    .run_commands(
        # SAM3D still reuses TRELLIS representation/render-state classes in trender.py.
        # Pull only the 35MB TRELLIS source submodule; no TRELLIS model weights are downloaded.
        "cd /workspace/EmbodiedGen && git submodule update --init --recursive --depth 1 thirdparty/TRELLIS",
        # Keep TRELLIS optional at import time for the SAM3D-only runtime.
        "echo IyBQcm9qZWN0IEVtYm9kaWVkR2VuCiMKIyBDb3B5cmlnaHQgKGMpIDIwMjUgSG9yaXpvbiBSb2JvdGljcy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC4KIwojIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSAiTGljZW5zZSIpOwojIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS4KIyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKIwojICAgICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMAojCiMgVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQojIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuICJBUyBJUyIgQkFTSVMsCiMgV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yCiMgaW1wbGllZC4gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nCiMgcGVybWlzc2lvbnMgYW5kIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLgoKCnRyeToKICAgIGZyb20gZW1ib2RpZWRfZ2VuLnV0aWxzLm1vbmtleV9wYXRjaC50cmVsbGlzIGltcG9ydCBtb25rZXlfcGF0aF90cmVsbGlzCiAgICBtb25rZXlfcGF0aF90cmVsbGlzKCkKZXhjZXB0IE1vZHVsZU5vdEZvdW5kRXJyb3I6CiAgICBtb25rZXlfcGF0aF90cmVsbGlzID0gTm9uZQppbXBvcnQgcmFuZG9tCgppbXBvcnQgdG9yY2gKZnJvbSBQSUwgaW1wb3J0IEltYWdlCmZyb20gZW1ib2RpZWRfZ2VuLmRhdGEudXRpbHMgaW1wb3J0IHRyZWxsaXNfcHJlcHJvY2Vzcwpmcm9tIGVtYm9kaWVkX2dlbi5tb2RlbHMuc2FtM2QgaW1wb3J0IFNhbTNkSW5mZXJlbmNlCmZyb20gZW1ib2RpZWRfZ2VuLnV0aWxzLnRyZW5kZXIgaW1wb3J0IHBhY2tfc3RhdGUsIHVucGFja19zdGF0ZQp0cnk6CiAgICBmcm9tIHRoaXJkcGFydHkuVFJFTExJUy50cmVsbGlzLnBpcGVsaW5lcyBpbXBvcnQgVHJlbGxpc0ltYWdlVG8zRFBpcGVsaW5lCmV4Y2VwdCBNb2R1bGVOb3RGb3VuZEVycm9yOgogICAgY2xhc3MgVHJlbGxpc0ltYWdlVG8zRFBpcGVsaW5lOiAgIyBTQU0zRC1vbmx5IHJ1bnRpbWUgc2VudGluZWwKICAgICAgICBwYXNzCgpfX2FsbF9fID0gWwogICAgImltYWdlM2RfbW9kZWxfaW5mZXIiLApdCgoKZGVmIGltYWdlM2RfbW9kZWxfaW5mZXIoCiAgICBwaXBlOiBUcmVsbGlzSW1hZ2VUbzNEUGlwZWxpbmUgfCBTYW0zZEluZmVyZW5jZSwKICAgIHNlZ19pbWFnZTogSW1hZ2UuSW1hZ2UsCiAgICBzZWVkOiBpbnQgPSBOb25lLAogICAgKiprd2FyZ3M6IGRpY3QsCikgLT4gZGljdFtzdHIsIGFueV06CiAgICAiIiJFeGVjdXRlIDNEIGdlbmVyYXRpb24gdXNpbmcgVHJlbGxpcyBvciBTQU0zRCBwaXBlbGluZSBvbiBpbnB1dCBpbWFnZS4iIiIKICAgIGlmIGlzaW5zdGFuY2UocGlwZSwgVHJlbGxpc0ltYWdlVG8zRFBpcGVsaW5lKToKICAgICAgICBwaXBlLmN1ZGEoKQogICAgICAgIHNlZ19pbWFnZSA9IHRyZWxsaXNfcHJlcHJvY2VzcyhzZWdfaW1hZ2UpCiAgICAgICAgb3V0cHV0cyA9IHBpcGUucnVuKAogICAgICAgICAgICBzZWdfaW1hZ2UsCiAgICAgICAgICAgIHByZXByb2Nlc3NfaW1hZ2U9RmFsc2UsCiAgICAgICAgICAgIHNlZWQ9KHJhbmRvbS5yYW5kaW50KDAsIDEwMDAwMCkgaWYgc2VlZCBpcyBOb25lIGVsc2Ugc2VlZCksCiAgICAgICAgICAgICMgT3B0aW9uYWwgcGFyYW1ldGVycwogICAgICAgICAgICAjIHNwYXJzZV9zdHJ1Y3R1cmVfc2FtcGxlcl9wYXJhbXM9ewogICAgICAgICAgICAjICAgICAic3RlcHMiOiAxMiwKICAgICAgICAgICAgIyAgICAgImNmZ19zdHJlbmd0aCI6IDcuNSwKICAgICAgICAgICAgIyB9LAogICAgICAgICAgICAjIHNsYXRfc2FtcGxlcl9wYXJhbXM9ewogICAgICAgICAgICAjICAgICAic3RlcHMiOiAxMiwKICAgICAgICAgICAgIyAgICAgImNmZ19zdHJlbmd0aCI6IDMsCiAgICAgICAgICAgICMgfSwKICAgICAgICAgICAgKiprd2FyZ3MsCiAgICAgICAgKQogICAgICAgIHBpcGUuY3B1KCkKICAgIGVsaWYgaXNpbnN0YW5jZShwaXBlLCBTYW0zZEluZmVyZW5jZSk6CiAgICAgICAgb3V0cHV0cyA9IHBpcGUucnVuKAogICAgICAgICAgICBzZWdfaW1hZ2UsCiAgICAgICAgICAgIHNlZWQ9KHJhbmRvbS5yYW5kaW50KDAsIDEwMDAwMCkgaWYgc2VlZCBpcyBOb25lIGVsc2Ugc2VlZCksCiAgICAgICAgICAgICMgc3RhZ2UxX2luZmVyZW5jZV9zdGVwcz0yNSwKICAgICAgICAgICAgIyBzdGFnZTJfaW5mZXJlbmNlX3N0ZXBzPTI1LAogICAgICAgICAgICAqKmt3YXJncywKICAgICAgICApCiAgICAgICAgc3RhdGUgPSBwYWNrX3N0YXRlKG91dHB1dHNbImdhdXNzaWFuIl1bMF0sIG91dHB1dHNbIm1lc2giXVswXSkKICAgICAgICAjIEFsaWduIEdTM0QgZnJvbSBTQU0zRCB3aXRoIFRSRUxMSVMgZm9ybWF0LgogICAgICAgIG91dHB1dHNbImdhdXNzaWFuIl1bMF0sIF8gPSB1bnBhY2tfc3RhdGUoc3RhdGUsIGRldmljZT0iY3VkYSIpCiAgICBlbHNlOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJVbnN1cHBvcnRlZCBwaXBlbGluZSB0eXBlOiB7dHlwZShwaXBlKX0iKQoKICAgIHRvcmNoLmN1ZGEuZW1wdHlfY2FjaGUoKQoKICAgIHJldHVybiBvdXRwdXRzCg== | base64 -d > /workspace/EmbodiedGen/embodied_gen/utils/inference.py",
        "cd /workspace/EmbodiedGen && python -m pip install --no-deps -e .",
        "cd /workspace/EmbodiedGen && python -c \"import embodied_gen; print('EMBODIEDGEN PACKAGE OK')\"",
        "cd /workspace/EmbodiedGen && python -c \"from embodied_gen.utils.trender import pack_state, unpack_state; print('TRELLIS RENDER BRIDGE IMPORT OK')\"",
        "python -m pip cache purge || true",
    )
    # Final CPU-only optimization layer: replace the earlier quick -O0 gsplat JIT
    # artifact with the real -O3/SM89 binary. This layer is intentionally last so
    # PyTorch3D/Kaolin/etc. remain cached if we tweak it.
    .run_commands(
        "python - <<'PY'\nfrom torch.utils.cpp_extension import _get_build_directory\nimport shutil\np=_get_build_directory('gsplat_cuda', verbose=False)\nprint('Removing old gsplat JIT dir:', p)\nshutil.rmtree(p, ignore_errors=True)\nPY",
        "FAST_COMPILE=0 VERBOSE=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 python -c \"from gsplat.cuda._backend import _C; print('GSPLAT O3 CUDA EXT OK', _C is not None)\"",
        "python - <<'PY'\nfrom torch.utils.cpp_extension import _get_build_directory\nfrom pathlib import Path\np=Path(_get_build_directory('gsplat_cuda', verbose=False))\nprint('GSPLAT O3 BUILD DIR', p)\nfor f in p.rglob('*'):\n    if f.is_file() and (f.suffix == '.so' or f.name == 'build.ninja'):\n        print(f, f.stat().st_size)\nPY",
    )
    # Modal already owns the real L40S. Disable Hugging Face ZeroGPU wrappers.
    # This is source-only and intentionally AFTER all compiled artifacts.
    .run_commands(
        "cd /workspace/EmbodiedGen && grep -RIl '@spaces.GPU' embodied_gen --include='*.py' | xargs -r sed -i '/^[[:space:]]*@spaces.GPU[[:space:]]*$/d'",
        "cd /workspace/EmbodiedGen && python -c \"from embodied_gen.scripts.render_gs import entrypoint; print('RENDER_GS DIRECT OK')\"",
        "cd /workspace/EmbodiedGen && python -m py_compile embodied_gen/scripts/imageto3d.py",
    )
    .workdir("/workspace/EmbodiedGen")
)
# Source-only headless patches are baked last; changing them never invalidates CUDA builds.
image = (
    image
    .add_local_file(
        "embodied_gen/scripts/imageto3d.py",
        "/workspace/EmbodiedGen/embodied_gen/scripts/imageto3d.py",
        copy=True,
    )
    .add_local_file(
        "embodied_gen/scripts/modal_postprocess.py",
        "/workspace/EmbodiedGen/embodied_gen/scripts/modal_postprocess.py",
        copy=True,
    )
    .add_local_file(
        "embodied_gen/models/gs_model.py",
        "/workspace/EmbodiedGen/embodied_gen/models/gs_model.py",
        copy=True,
    )
    .add_local_file(
        "embodied_gen/data/backproject_v3.py",
        "/workspace/EmbodiedGen/embodied_gen/data/backproject_v3.py",
        copy=True,
    )
    .run_commands(
        "cd /workspace/EmbodiedGen && python -m py_compile embodied_gen/scripts/imageto3d.py embodied_gen/scripts/modal_postprocess.py embodied_gen/models/gs_model.py embodied_gen/data/backproject_v3.py"
    )
)

@app.function(image=image, volumes={"/data": data}, timeout=60 * 60, cpu=4.0, memory=16384)
def preload_weights():
    """CPU-only: download checkpoints exactly once into a persistent Volume."""
    # Keep every runtime download off the paid GPU path.
    os.environ["TORCH_HOME"] = "/data/torch"
    os.environ["U2NET_HOME"] = "/data/u2net"

    # rembg/u2net (176 MB)
    u2net = Path("/data/u2net/u2net.onnx")
    if not u2net.exists():
        u2net.parent.mkdir(parents=True, exist_ok=True)
        print("CPU ONLY: caching rembg u2net.onnx...", flush=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            str(u2net),
        )

    # DINOv2 ViT-L/14 reg4 (~1.13 GB), required by SAM3D/MoGe.
    dino_ckpt = Path("/data/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
    dino_repo = Path("/data/torch/hub/facebookresearch_dinov2_main")
    if not dino_ckpt.exists() or not dino_repo.exists():
        print("CPU ONLY: caching DINOv2 repo + vitl14_reg4 checkpoint...", flush=True)
        import torch
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14_reg",
            pretrained=True,
            trust_repo=True,
        )
        del model

    target = Path("/data/weights/sam-3d-objects")
    marker = target / "checkpoints/pipeline.yaml"
    if marker.exists():
        data.commit()
        size = subprocess.check_output(["du", "-sh", str(target)], text=True).split()[0]
        print(f"SAM3D already cached: {target} ({size})")
        return {"cached": True, "path": str(target), "size": size}
    target.parent.mkdir(parents=True, exist_ok=True)
    print("CPU ONLY: downloading facebook/sam-3d-objects into persistent Volume...", flush=True)
    from modelscope import snapshot_download
    p = snapshot_download("facebook/sam-3d-objects", local_dir=str(target))
    if not marker.exists():
        raise RuntimeError(f"download completed but {marker} is missing: {p}")
    data.commit()
    size = subprocess.check_output(["du", "-sh", str(target)], text=True).split()[0]
    print("SAM3D cached:", p, size)
    return {"cached": False, "path": str(target), "size": size}

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": data},
    timeout=60 * 60,
    cpu=6.0,
    memory=32768,
    min_containers=0,
    scaledown_window=15,
)
def run_img3d():
    """The only function in this app that is allowed to rent a GPU."""
    # gsplat is precompiled -O3/SM89 in the image; keep default FAST_COMPILE=0.
    os.environ.pop("FAST_COMPILE", None)
    os.environ["TORCH_HOME"] = "/data/torch"
    os.environ["U2NET_HOME"] = "/data/u2net"
    os.chdir("/workspace/EmbodiedGen")
    weights = Path("/data/weights/sam-3d-objects")
    marker = weights / "checkpoints/pipeline.yaml"
    if not marker.exists():
        raise RuntimeError("Refusing to rent GPU without preloaded SAM3D weights")

    local_weights = Path("weights/sam-3d-objects")
    local_weights.parent.mkdir(exist_ok=True)
    if local_weights.is_symlink():
        local_weights.unlink()
    elif local_weights.exists():
        raise RuntimeError(f"unexpected baked-in weights dir: {local_weights}")
    local_weights.symlink_to(weights, target_is_directory=True)

    out = Path("/data/outputs/test")
    if out.exists():
        # Make the smoke test deterministic and avoid confusing old outputs with success.
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], check=False)
    subprocess.run(["python", "-c", "import torch; print('torch=',torch.__version__,'runtime=',torch.version.cuda,'gpu=',torch.cuda.get_device_name(0),'cc=',torch.cuda.get_device_capability(0))"], check=True)

    # Fail fast before loading 13 GB SAM3D if gsplat itself is unhealthy.
    gs_smoke = r"""
import time, torch
from gsplat.rendering import rasterization
N=256; dev='cuda'
torch.manual_seed(0)
means=torch.randn(N,3,device=dev); means[:,2].abs_().add_(3.0)
quats=torch.randn(N,4,device=dev); quats=quats/quats.norm(dim=-1,keepdim=True)
scales=torch.full((N,3),0.03,device=dev)
op=torch.full((N,),0.5,device=dev)
colors=torch.rand(N,3,device=dev)
view=torch.eye(4,device=dev)[None]
K=torch.tensor([[300.,0.,128.],[0.,300.,128.],[0.,0.,1.]],device=dev)[None]
t0=time.time(); rgb,alpha,_=rasterization(means,quats,scales,op,colors,view,K,256,256); torch.cuda.synchronize(); dt=time.time()-t0
print('GSPLAT_SMOKE_SECONDS=',dt,'rgb=',tuple(rgb.shape),'alpha=',tuple(alpha.shape),flush=True)
assert dt < 20, f'gsplat tiny render unexpectedly slow: {dt:.2f}s'
"""
    subprocess.run(["python", "-c", gs_smoke], check=True, timeout=30)

    # Same user-requested smoke test, with only two cost/debug flags:
    # n_retry=1 prevents paid retries; keep_intermediate preserves diagnostics.
    cmd = [
        "img3d-cli",
        "--image_path", "apps/assets/example_image/sample_00.jpg",
        "--output_root", "/data/outputs/test",
        "--n_retry", "1",
        "--image3d_model", "SAM3D",
        "--keep_intermediate",
    ]
    print("GPU PHASE 1 (SAM3D generate-only):", " ".join(cmd), flush=True)
    gen_env = os.environ.copy()
    gen_env["MODAL_GENERATE_ONLY"] = "1"
    proc = subprocess.run(cmd, text=True, env=gen_env)
    if proc.returncode != 0:
        data.commit()
        raise RuntimeError(f"img3d-cli generate phase exited {proc.returncode}")

    state_file = out / "sample_00_state.pkl"
    if not state_file.exists():
        data.commit()
        raise RuntimeError("generate phase returned 0 but state pickle is missing")

    print("GPU PHASE 2 (fresh CUDA postprocess process)", flush=True)
    post_cmd = [
        "python", "-m", "embodied_gen.scripts.modal_postprocess",
        "--output_root", "/data/outputs/test",
        "--filename", "sample_00",
        "--texture_size", "1024",
        "--video_frames", "60",
        "--fast_bake",
        "--skip_fix_mesh",
        "--disable_decompose_convex",
        "--skip_aesthetic",
    ]
    print("POST:", " ".join(post_cmd), flush=True)
    post = subprocess.run(post_cmd, text=True)
    proc = post
    data.commit()
    files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()]
    print("OUTPUT FILES (%d):" % len(files))
    print("\n".join(files[:300]))
    if proc.returncode != 0:
        raise RuntimeError(f"img3d-cli exited {proc.returncode}")
    # Do not call a silent no-op a success.
    geometry = [f for f in files if f.lower().endswith((".ply", ".obj", ".glb", ".urdf"))]
    if not geometry:
        raise RuntimeError("img3d-cli returned 0 but produced no geometry artifact")
    return {"returncode": 0, "geometry": geometry, "files": files}

@app.local_entrypoint()
def main():
    print("[1/2] CPU-only model cache")
    print(preload_weights.remote())
    print("[2/2] L40S inference (first and only GPU allocation)")
    print(run_img3d.remote())


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": data},
    timeout=30 * 60,
    cpu=6.0,
    memory=32768,
    min_containers=0,
    scaledown_window=10,
)
def postprocess_only():
    """Resume from persisted SAM3D state without re-running the 13GB model."""
    os.chdir("/workspace/EmbodiedGen")
    state = Path("/data/outputs/test/sample_00_state.pkl")
    if not state.exists():
        raise RuntimeError(f"missing persisted state: {state}")
    cmd = [
        "python", "-m", "embodied_gen.scripts.modal_postprocess",
        "--output_root", "/data/outputs/test",
        "--filename", "sample_00",
        "--texture_size", "1024",
        "--video_frames", "60",
        "--fast_bake",
        "--skip_fix_mesh",
        "--disable_decompose_convex",
        "--skip_aesthetic",
    ]
    print("RESUME POSTPROCESS:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    data.commit()
    report = Path("/data/outputs/test/validation_report.json")
    if not report.exists():
        raise RuntimeError("postprocess returned 0 but validation report is missing")
    print(report.read_text()[:12000])
    return report.read_text()

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": data},
    timeout=10 * 60,
    cpu=4.0,
    memory=16384,
    min_containers=0,
    scaledown_window=10,
)
def diagnose_real_gs_render():
    """Benchmark real SAM3D PLY with isolated gsplat parameter sets.

    Each variant runs in a child process with a hard timeout so one pathological
    gsplat setting cannot burn the GPU indefinitely.
    """
    import json, textwrap
    ply = "/data/outputs/test/sample_00_gs_aligned.ply"
    if not Path(ply).exists():
        raise RuntimeError(f"missing real SAM3D PLY: {ply}")
    print(subprocess.check_output(["du", "-h", ply], text=True).strip(), flush=True)

    variants = [
        # Current EmbodiedGen v2.0.0 settings, first with a sane far plane.
        dict(name="exact_far1e9", packed=False, absgrad=True, rasterize_mode="antialiased", render_mode="RGB+ED", far_plane=1_000_000_000.0, radius_clip=0.0),
        dict(name="exact_far100", packed=False, absgrad=True, rasterize_mode="antialiased", render_mode="RGB+ED", far_plane=100.0, radius_clip=0.0),
        dict(name="classic_unpacked_rgb", packed=False, absgrad=False, rasterize_mode="classic", render_mode="RGB", far_plane=100.0, radius_clip=0.0),
        dict(name="classic_packed_rgb", packed=True, absgrad=False, rasterize_mode="classic", render_mode="RGB", far_plane=100.0, radius_clip=0.0),
        dict(name="classic_packed_rgb_ed", packed=True, absgrad=False, rasterize_mode="classic", render_mode="RGB+ED", far_plane=100.0, radius_clip=0.0),
        dict(name="aa_packed_rgb", packed=True, absgrad=False, rasterize_mode="antialiased", render_mode="RGB", far_plane=100.0, radius_clip=0.0),
        dict(name="aa_packed_rgb_ed", packed=True, absgrad=False, rasterize_mode="antialiased", render_mode="RGB+ED", far_plane=100.0, radius_clip=0.0),
        # Screen-size culling is a practical fallback if a few pathological huge splats exist.
        dict(name="classic_packed_clip1", packed=True, absgrad=False, rasterize_mode="classic", render_mode="RGB", far_plane=100.0, radius_clip=1.0),
    ]

    child = r'''
import json, sys, time, torch
from embodied_gen.models.gs_model import load_gs_model
from embodied_gen.data.utils import CameraSetting, init_kal_camera
from gsplat import rasterization
cfg=json.loads(sys.argv[1]); ply=sys.argv[2]
model=load_gs_model(ply, pre_quat=[0.0,0.0,1.0,0.0])
print('N=', model._means.shape[0], 'device=', model._means.device, flush=True)
cp=CameraSetting(num_images=4,elevation=[30,-30],distance=5,resolution_hw=(512,512),fov=0.5235987755982988,device='cuda')
cam=init_kal_camera(cp, flip_az=True)
mv=cam.view_matrix(); mv[:,:3,3]=-mv[:,:3,3]
c2w=torch.linalg.inv(mv[0].to('cuda')); K=torch.tensor(cp.Ks,device='cuda')
gs=model.get_gaussians(c2w, apply_activate=True)
kwargs=dict(
 means=gs._means, quats=gs._quats, scales=gs._scales,
 opacities=gs._opacities.squeeze(), colors=gs._rgbs,
 viewmats=torch.linalg.inv(c2w)[None,...], Ks=K[None,...], width=512, height=512,
 near_plane=0.01,
 packed=cfg['packed'], absgrad=cfg['absgrad'], sparse_grad=False,
 rasterize_mode=cfg['rasterize_mode'], render_mode=cfg['render_mode'],
 far_plane=cfg['far_plane'], radius_clip=cfg['radius_clip'])
t0=time.perf_counter(); out=rasterization(**kwargs); torch.cuda.synchronize(); dt=time.perf_counter()-t0
print('OK',cfg['name'],'seconds=',dt,'render_shape=',tuple(out[0].shape),flush=True)
'''
    results=[]
    for cfg in variants:
        print("\n=== VARIANT", cfg["name"], "===", flush=True)
        try:
            p=subprocess.run(["python","-c",child,json.dumps(cfg),ply], text=True, capture_output=True, timeout=25)
            print(p.stdout, end="")
            if p.stderr:
                print("STDERR:", p.stderr[-2500:])
            results.append((cfg["name"], p.returncode, p.stdout[-1000:]))
        except subprocess.TimeoutExpired as e:
            print("TIMEOUT >25s", cfg["name"], flush=True)
            results.append((cfg["name"], 124, "TIMEOUT"))
    print("\n=== SUMMARY ===")
    for r in results: print(r)
    return results


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": data},
    timeout=5 * 60,
    cpu=4.0,
    memory=16384,
    min_containers=0,
    scaledown_window=10,
)
def render_probe():
    code = r"""
import time

def mark(x):
    print(f'PROBE {time.time():.3f} {x}', flush=True)
mark('start')
import torch
mark(f'torch import cuda={torch.cuda.is_available()} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
from PIL import Image
mark('PIL import')
from gsplat import rasterization
mark('gsplat import')
from embodied_gen.data.utils import CameraSetting, init_kal_camera
mark('camera imports')
from embodied_gen.models.gs_model import load_gs_model
mark('gs_model import')
import math
ply='/data/outputs/test/sample_00_gs_aligned.ply'
model=load_gs_model(ply, pre_quat=[0.0,0.0,1.0,0.0])
mark(f'load_gs_model N={model._means.shape[0]}')
cp=CameraSetting(num_images=4,elevation=[30,-30],distance=5,resolution_hw=(512,512),fov=math.radians(30),device='cuda')
mark('CameraSetting')
cam=init_kal_camera(cp, flip_az=True)
mark('init_kal_camera')
mv=cam.view_matrix(); mv[:,:3,3]=-mv[:,:3,3]
mark('view_matrix')
c2w=torch.linalg.inv(mv[0].to('cuda')); K=torch.tensor(cp.Ks,device='cuda')
mark('c2w/K')
gs=model.get_gaussians(c2w, apply_activate=True)
mark('get_gaussians')
kwargs=dict(means=gs._means,quats=gs._quats,scales=gs._scales,opacities=gs._opacities.squeeze(),colors=gs._rgbs,viewmats=torch.linalg.inv(c2w)[None,...],Ks=K[None,...],width=512,height=512,packed=False,absgrad=True,sparse_grad=False,rasterize_mode='antialiased',near_plane=0.01,far_plane=1000000000,radius_clip=0.0,render_mode='RGB+ED')
mark('before rasterization')
t0=time.perf_counter(); out=rasterization(**kwargs); mark(f'rasterization returned lazy dt={time.perf_counter()-t0:.3f}')
torch.cuda.synchronize(); mark(f'synchronized total={time.perf_counter()-t0:.3f}')
rgb=(out[0][0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy(); mark(f'cpu rgb {rgb.shape}')
"""
    subprocess.run(["python", "-c", code], check=True, timeout=90)
    return "ok"

@app.function(
    image=image,
    timeout=10 * 60,
    cpu=4.0,
    memory=16384,
)
def probe_backproject_import():
    """CPU-only import profiler: no GPU rental."""
    import subprocess, time
    probes = [
        "import nvdiffrast.torch",
        "import utils3d",
        "import xatlas",
        "import embodied_gen.data.mesh_operator",
        "import embodied_gen.models.delight_model",
        "import embodied_gen.models.sr_model",
        "import embodied_gen.data.backproject_v3",
    ]
    for code in probes:
        print(f"\n=== {code} ===", flush=True)
        t=time.perf_counter()
        try:
            p=subprocess.run(
                ["python","-c",code], text=True, capture_output=True, timeout=90
            )
            print(f"rc={p.returncode} seconds={time.perf_counter()-t:.3f}", flush=True)
            if p.stdout: print("STDOUT",p.stdout[-3000:],flush=True)
            if p.stderr: print("STDERR",p.stderr[-5000:],flush=True)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT >90s for {code}",flush=True)
