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
APP_NAME = "modal-3d-embodiedgen"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("modal-3d-embodiedgen-weights", create_if_missing=True)
artifacts = modal.Volume.from_name("modal-3d-artifacts", create_if_missing=True)

ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "HF_HOME": "/weights/hf",
    "MODELSCOPE_CACHE": "/weights/modelscope",
    "TORCH_HOME": "/weights/torch",
    "U2NET_HOME": "/weights/u2net",
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
        "PIP_CONSTRAINT=/tmp/eg-constraints.txt python -m pip install plyfile moderngl glcontext ftfy fvcore iopath",
        "python -m pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 'pillow<12'",
        "python -m pip install --no-deps kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html",
        "python -m pip install pygltflib warp-lang usd-core ipycanvas ipyevents 'jupyter_client<8' tornado",
        "python -m pip install --no-deps gsplat==1.5.3",
        "python -m pip install --no-deps fast-simplification==0.2.0",
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



def _weights_info() -> dict:
    target = Path("/weights/sam-3d-objects")
    marker = target / "checkpoints/pipeline.yaml"
    if not marker.exists():
        raise RuntimeError("SAM3D weights missing; run preload_weights first")
    return {
        "path": str(target),
        "size": subprocess.check_output(["du", "-sh", str(target)], text=True).split()[0],
    }


@app.function(
    image=image,
    volumes={"/weights": weights},
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
)
def preload_weights():
    """CPU-only model/cache pull for a fresh Modal workspace."""
    os.environ.update({"TORCH_HOME": "/weights/torch", "U2NET_HOME": "/weights/u2net"})
    t0 = time.perf_counter()
    u2net = Path("/weights/u2net/u2net.onnx")
    if not u2net.exists():
        import urllib.request
        u2net.parent.mkdir(parents=True, exist_ok=True)
        print("CPU ONLY: downloading U2Net", flush=True)
        urllib.request.urlretrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            str(u2net),
        )

    dino = Path("/weights/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
    dino_repo = Path("/weights/torch/hub/facebookresearch_dinov2_main")
    if not dino.exists() or not dino_repo.exists():
        print("CPU ONLY: downloading DINOv2 repo + ViT-L/14 reg4", flush=True)
        import torch
        m = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14_reg",
            pretrained=True,
            trust_repo=True,
        )
        del m

    target = Path("/weights/sam-3d-objects")
    marker = target / "checkpoints/pipeline.yaml"
    if not marker.exists():
        print("CPU ONLY: downloading SAM3D weights", flush=True)
        from modelscope import snapshot_download
        snapshot_download("facebook/sam-3d-objects", local_dir=str(target))
    weights.commit()
    info = _weights_info()
    info["seconds"] = round(time.perf_counter() - t0, 3)
    print("WEIGHTS_READY", json.dumps(info), flush=True)
    return info


def _rembg_load(worker, cpu_label: str) -> None:
    import uuid
    import rembg

    t0=time.perf_counter()
    os.chdir("/workspace/EmbodiedGen")
    os.environ.update({"U2NET_HOME":"/weights/u2net", "TORCH_HOME":"/weights/torch"})
    worker.session=rembg.new_session("u2net", providers=["CPUExecutionProvider"])
    worker.session_load_seconds=time.perf_counter()-t0
    worker.instance_id=uuid.uuid4().hex
    worker.cpu_label=cpu_label
    print(
        f"REMBG_RESIDENT_READY cpu={cpu_label} instance_id={worker.instance_id} "
        f"load_seconds={worker.session_load_seconds:.3f}",
        flush=True,
    )


def _rembg_prepare(worker, job_id: str) -> dict:
    from PIL import Image
    import rembg

    t0=time.perf_counter()
    artifacts.reload()
    root=Path("/artifacts/embodiedgen/jobs")/job_id
    root.mkdir(parents=True,exist_ok=True)
    src=Path("apps/assets/example_image/sample_00.jpg")
    raw=root/"sample_00_raw.png"
    cond=root/"sample_00_cond.png"
    image_in=Image.open(src)
    image_in.save(raw)
    current_max=max(image_in.size)
    scale=min(1.0,1024.0/current_max)
    if scale < 1.0:
        image_in=image_in.resize(
            (int(image_in.width*scale),int(image_in.height*scale)),
            Image.Resampling.LANCZOS,
        )
    r0=time.perf_counter()
    rembg.remove(image_in,session=worker.session).save(cond)
    r1=time.perf_counter()
    artifacts.commit()
    out={
        "job_id":job_id,
        "raw":str(raw),
        "cond":str(cond),
        "cpu":worker.cpu_label,
        "instance_id":worker.instance_id,
        "session_load_seconds":round(worker.session_load_seconds,3),
        "remove_seconds":round(r1-r0,3),
        "method_seconds":round(time.perf_counter()-t0,3),
    }
    print("REMBG_PREPARE_OK",json.dumps(out),flush=True)
    return out


@app.cls(
    image=image,
    volumes={"/weights": weights, "/artifacts": artifacts},
    cpu=1.0,
    memory=4096,
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
    timeout=10 * 60,
)
class RembgWorker:
    """Production rembg worker: 1 CPU + 4 GiB, warm for 120 seconds."""

    @modal.enter()
    def load(self):
        _rembg_load(self,"1cpu-4g")

    @modal.method()
    def prepare(self, job_id: str) -> dict:
        return _rembg_prepare(self,job_id)


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/weights": weights, "/artifacts": artifacts},
    cpu=6.0,
    memory=32768,
    min_containers=0,
    max_containers=1,
    scaledown_window=90,
    timeout=30 * 60,
)
class Sam3DWorker:
    """Heavy L40S worker: keep the 13GB SAM3D pipeline resident for 90s."""

    @modal.enter()
    def load(self):
        t0=time.perf_counter()
        os.chdir("/workspace/EmbodiedGen")
        os.environ.update({"TORCH_HOME":"/weights/torch", "U2NET_HOME":"/weights/u2net"})
        assert subprocess.run(["bash","-lc","command -v nvcc"],capture_output=True).returncode != 0
        import torch
        from embodied_gen.models.sam3d import Sam3dInference
        import uuid
        self.torch=torch
        self.pipeline=Sam3dInference(local_dir="/weights/sam-3d-objects")
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-t0
        self.instance_id=uuid.uuid4().hex
        print(f"SAM3D_RESIDENT_READY instance_id={self.instance_id} load_seconds={self.load_seconds:.3f}",flush=True)

    @modal.method()
    def generate(self, job_id: str, seed: int = 0) -> dict:
        import pickle
        import numpy as np
        from PIL import Image
        from embodied_gen.models.gs_model import GaussianOperator
        from embodied_gen.utils.trender import pack_state

        t0=time.perf_counter()
        artifacts.reload()
        root=Path("/artifacts/embodiedgen/jobs")/job_id
        cond=root/"sample_00_cond.png"
        if not cond.exists(): raise FileNotFoundError(cond)
        image=Image.open(cond).convert("RGBA")
        i0=time.perf_counter()
        outputs=self.pipeline.run(image,seed=seed)
        self.torch.cuda.synchronize()
        i1=time.perf_counter()
        gs=outputs["gaussian"][0]; mesh=outputs["mesh"][0]
        gs_path=root/"sample_00_gs.ply"
        gs.save_ply(str(gs_path))
        rot_matrix=np.array([[0,0,-1],[0,1,0],[1,0,0]])
        gs_add_rot=np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        pose=GaussianOperator.trans_to_quatpose(gs_add_rot @ rot_matrix)
        aligned=root/"sample_00_gs_aligned.ply"
        GaussianOperator.resave_ply(str(gs_path),str(aligned),instance_pose=pose,device="cpu")
        state=pack_state(gs,mesh)
        with (root/"sample_00_state.pkl").open("wb") as f:
            pickle.dump(state,f,protocol=pickle.HIGHEST_PROTOCOL)
        del outputs,gs,mesh,state
        self.torch.cuda.empty_cache()
        artifacts.commit()
        result={
            "job_id":job_id,
            "instance_id":self.instance_id,
            "resident_model_load_seconds":round(self.load_seconds,3),
            "inference_seconds":round(i1-i0,3),
            "method_seconds":round(time.perf_counter()-t0,3),
            "gpu":self.torch.cuda.get_device_name(0),
        }
        print("SAM3D_GENERATE_OK",json.dumps(result),flush=True)
        return result


@app.cls(
    image=image,
    volumes={"/artifacts": artifacts},
    cpu=4.0,
    memory=8192,
    min_containers=0,
    max_containers=1,
    scaledown_window=90,
    timeout=15 * 60,
)
class MeshWorker:
    """Persistent CPU mesh worker: fast-simplification -> 50k -> xatlas."""

    @modal.enter()
    def load(self):
        import uuid
        t0=time.perf_counter()
        # Warm imports once per container. The C++ modules stay resident.
        import numpy  # noqa: F401
        import fast_simplification  # noqa: F401
        import xatlas  # noqa: F401
        self.instance_id=uuid.uuid4().hex
        self.load_seconds=time.perf_counter()-t0
        print(
            f"MESH_RESIDENT_READY instance_id={self.instance_id} "
            f"load_seconds={self.load_seconds:.3f}",
            flush=True,
        )

    @modal.method()
    def process(self, job_id: str) -> dict:
        import pickle
        import numpy as np
        import fast_simplification
        import xatlas

        t0=time.perf_counter()
        artifacts.reload()
        root=Path("/artifacts/embodiedgen/jobs")/job_id
        with (root/"sample_00_state.pkl").open("rb") as f:
            state=pickle.load(f)
        vertices=np.asarray(state["mesh"]["vertices"],dtype=np.float32)
        faces=np.asarray(state["mesh"]["faces"],dtype=np.int32)
        input_vertices,input_faces=len(vertices),len(faces)

        mesh_add_rot=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=np.float32)
        rot_matrix=np.array([[0,0,-1],[0,1,0],[1,0,0]],dtype=np.float32)
        vertices=vertices @ mesh_add_rot @ rot_matrix

        # Critical path deliberately does not export the 884k-face raw OBJ.
        # state.pkl remains the lossless high-poly intermediate if it is needed later.
        d0=time.perf_counter()
        vertices,faces=fast_simplification.simplify(
            vertices,
            faces,
            target_count=50000,
            agg=7.0,
            preserve_border=False,
        )
        d1=time.perf_counter()
        vertices=np.asarray(vertices,dtype=np.float32)
        faces=np.asarray(faces,dtype=np.int32)

        bbmin=vertices.min(0); bbmax=vertices.max(0)
        center=(bbmin+bbmax)*0.5
        scale=np.float32(2.0/(bbmax-bbmin).max())
        norm=(vertices-center)*scale
        x_rot=np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=np.float32)
        z_rot=np.array([[0,1,0],[-1,0,0],[0,0,1]],dtype=np.float32)
        norm=norm @ x_rot @ z_rot

        x0=time.perf_counter()
        vmapping,indices,uvs=xatlas.parametrize(norm,faces)
        x1=time.perf_counter()
        baked_vertices=norm[vmapping]

        # These arrays are only a few MB; compression burns CPU and adds latency.
        np.savez(
            root/"bake_mesh.npz",
            vertices=baked_vertices.astype(np.float32),
            faces=np.asarray(indices,dtype=np.int32),
            uvs=np.asarray(uvs,dtype=np.float32),
            scale=np.asarray(scale,dtype=np.float32),
            center=center.astype(np.float32),
            x_rot=x_rot,
            z_rot=z_rot,
        )
        artifacts.commit()
        result={
            "job_id":job_id,
            "instance_id":self.instance_id,
            "worker_load_seconds":round(self.load_seconds,3),
            "input_vertices":int(input_vertices),
            "input_faces":int(input_faces),
            "dec_vertices":int(len(vertices)),
            "dec_faces":int(len(faces)),
            "simplify_seconds":round(d1-d0,3),
            "uv_vertices":int(len(baked_vertices)),
            "uv_faces":int(len(indices)),
            "xatlas_seconds":round(x1-x0,3),
            "method_seconds":round(time.perf_counter()-t0,3),
        }
        print("MESH_PROCESS_OK",json.dumps(result),flush=True)
        return result


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/artifacts": artifacts},
    timeout=15 * 60,
    cpu=4.0,
    memory=16384,
    min_containers=0,
    scaledown_window=30,
)
def lite_gpu_bake(job_id: str) -> dict:
    """Light L40S: gsplat multiview render + texture bake; no SAM3D model."""
    import math
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from gsplat import rasterization
    from embodied_gen.data.utils import CameraSetting, init_kal_camera, post_process_texture
    from embodied_gen.data.backproject_v3 import TextureBaker
    from embodied_gen.models.gs_model import load_gs_model

    t0=time.perf_counter()
    artifacts.reload()
    root=Path("/artifacts/embodiedgen/jobs")/job_id
    d=np.load(root/"bake_mesh.npz")
    vertices=d["vertices"]; faces=d["faces"]; uvs=d["uvs"]
    cp=CameraSetting(num_images=24,elevation=[0],distance=5.0,resolution_hw=(512,512),fov=math.radians(30),device="cuda")
    cam=init_kal_camera(cp,flip_az=True)
    mv=cam.view_matrix(); mv[:,:3,3]=-mv[:,:3,3]
    K=torch.tensor(cp.Ks,device="cuda")
    model=load_gs_model(str(root/"sample_00_gs_aligned.ply"),pre_quat=[0.,0.,1.,0.])
    views=[]
    r0=time.perf_counter()
    for m in mv:
        c2w=torch.linalg.inv(m.to("cuda")); gs=model.get_gaussians(c2w,apply_activate=True)
        renders,_,_=rasterization(
            means=gs._means,quats=gs._quats,scales=gs._scales,
            opacities=gs._opacities.squeeze(),colors=gs._rgbs,
            viewmats=torch.linalg.inv(c2w)[None,...],Ks=K[None,...],width=512,height=512,
            packed=False,absgrad=True,sparse_grad=False,rasterize_mode="antialiased",
            near_plane=0.01,far_plane=1_000_000_000,radius_clip=0.0,render_mode="RGB")
        torch.cuda.synchronize()
        views.append((renders[0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy())
    r1=time.perf_counter()

    b0=time.perf_counter()
    baker=TextureBaker(vertices,faces,uvs,cp,device="cuda")
    texture=baker.bake_texture([v[...,:3] for v in views],texture_size=1024,mode="fast")
    texture=post_process_texture(texture)
    b1=time.perf_counter()
    Image.fromarray(texture).save(root/"texture.png")

    # Preview is nearly free compared with model inference; reuse horizontal gsplat orbit.
    preview=[]
    cpv=CameraSetting(num_images=60,elevation=[0],distance=5.0,resolution_hw=(512,512),fov=math.radians(30),device="cuda")
    camv=init_kal_camera(cpv,flip_az=True); mvv=camv.view_matrix(); mvv[:,:3,3]=-mvv[:,:3,3]
    Kv=torch.tensor(cpv.Ks,device="cuda")
    for m in mvv:
        c2w=torch.linalg.inv(m.to("cuda")); gs=model.get_gaussians(c2w,apply_activate=True)
        rr,_,_=rasterization(means=gs._means,quats=gs._quats,scales=gs._scales,
            opacities=gs._opacities.squeeze(),colors=gs._rgbs,
            viewmats=torch.linalg.inv(c2w)[None,...],Ks=Kv[None,...],width=512,height=512,
            packed=False,absgrad=True,sparse_grad=False,rasterize_mode="antialiased",
            near_plane=0.01,far_plane=1_000_000_000,radius_clip=0.0,render_mode="RGB")
        torch.cuda.synchronize()
        preview.append((rr[0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy())
    imageio.mimsave(str(root/"preview.mp4"),preview,fps=30,codec="libx264")
    artifacts.commit()
    result={
        "job_id":job_id,
        "gpu":torch.cuda.get_device_name(0),
        "render24_seconds":round(r1-r0,3),
        "bake_seconds":round(b1-b0,3),
        "total_seconds":round(time.perf_counter()-t0,3),
    }
    print("LITE_GPU_BAKE_OK",json.dumps(result),flush=True)
    return result


@app.function(
    image=image,
    volumes={"/artifacts": artifacts},
    timeout=15 * 60,
    cpu=4.0,
    memory=16384,
)
def cpu_finalize(job_id: str) -> dict:
    """Pure CPU: restore mesh scale, export OBJ/GLB, write fallback URDF and validate."""
    import json as _json
    import shutil
    import xml.etree.ElementTree as ET
    import numpy as np
    import trimesh
    from PIL import Image

    t0=time.perf_counter()
    artifacts.reload()
    root=Path("/artifacts/embodiedgen/jobs")/job_id
    d=np.load(root/"bake_mesh.npz")
    vertices=d["vertices"]; faces=d["faces"]; uvs=d["uvs"]
    scale=float(d["scale"]); center=d["center"]; x_rot=d["x_rot"]; z_rot=d["z_rot"]
    vertices=vertices @ np.linalg.inv(z_rot)
    vertices=vertices @ np.linalg.inv(x_rot)
    vertices=vertices/scale + center
    texture=Image.open(root/"texture.png").convert("RGB")
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs,image=texture),process=True)
    obj=root/"sample_00.obj"; glb=root/"sample_00.glb"
    mesh.export(obj); mesh.export(glb)

    result=root/"result"; meshdir=result/"mesh"
    if result.exists(): shutil.rmtree(result)
    meshdir.mkdir(parents=True,exist_ok=True)
    for pth in root.glob("sample_00*.*"):
        if pth.suffix.lower() in {".obj",".mtl",".glb",".ply"}: shutil.copy2(pth,meshdir/pth.name)
    for pth in root.glob("*.png"):
        if pth.name=="texture.png": shutil.copy2(pth,meshdir/pth.name)
    if (root/"preview.mp4").exists(): shutil.copy2(root/"preview.mp4",result/"video.mp4")

    # GPT-free fallback attributes match the upstream fallback semantics.
    robot=ET.Element("robot",{"name":"sample_00"})
    link=ET.SubElement(robot,"link",{"name":"sample_00"})
    visual=ET.SubElement(link,"visual"); ET.SubElement(visual,"origin",{"xyz":"0 0 0","rpy":"1.5708 0 1.5708"})
    geom=ET.SubElement(visual,"geometry"); ET.SubElement(geom,"mesh",{"filename":"mesh/sample_00.obj","scale":"1 1 1"})
    collision=ET.SubElement(link,"collision"); ET.SubElement(collision,"origin",{"xyz":"0 0 0","rpy":"1.5708 0 1.5708"})
    cgeom=ET.SubElement(collision,"geometry"); ET.SubElement(cgeom,"mesh",{"filename":"mesh/sample_00.obj","scale":"1 1 1"})
    inertial=ET.SubElement(link,"inertial"); ET.SubElement(inertial,"mass",{"value":"1.0"})
    extra=ET.SubElement(link,"extra_info")
    for k,v in {"category":"unknown","description":"unknown","real_height":"1.0","version":"2.0.0","gs_model":"mesh/sample_00_gs.ply"}.items(): ET.SubElement(extra,k).text=v
    urdf=result/"sample_00.urdf"; ET.ElementTree(robot).write(urdf,encoding="utf-8",xml_declaration=True)

    # Structural validation.
    objm=trimesh.load(obj,force="mesh"); glbs=trimesh.load(glb,force="scene")
    ET.parse(urdf)
    with (root/"sample_00_gs.ply").open("rb") as f: header=f.read(8192).decode("ascii","ignore")
    ply_vertices=next(int(x.split()[-1]) for x in header.splitlines() if x.startswith("element vertex "))
    checks={
        "ply_vertices":ply_vertices,
        "obj_vertices":int(len(objm.vertices)),
        "obj_faces":int(len(objm.faces)),
        "glb_geometries":int(len(glbs.geometry)),
        "urdf_mesh_exists":(result/"mesh/sample_00.obj").exists(),
        "video_exists":(result/"video.mp4").exists(),
    }
    if not all([checks["ply_vertices"]>0,checks["obj_vertices"]>0,checks["obj_faces"]>0,checks["glb_geometries"]>0,checks["urdf_mesh_exists"]]):
        raise RuntimeError(checks)
    report={"job_id":job_id,"checks":checks,"seconds":round(time.perf_counter()-t0,3)}
    (root/"validation_report.json").write_text(_json.dumps(report,indent=2)+"\n")
    artifacts.commit()
    print("VALIDATION_OK",_json.dumps(report),flush=True)
    return report



@app.local_entrypoint()
def benchmark_split():
    """Prepare both jobs first, then measure cold→warm resident SAM3D back-to-back."""
    print("WEIGHTS", preload_weights.remote(), flush=True)
    worker = Sam3DWorker()
    rembg_worker = RembgWorker()
    mesh_worker = MeshWorker()

    jobs=[]
    # Prepare both inputs before allocating the heavy GPU. The rembg session stays
    # resident for 120s, so the second input reuses the same U2Net/ONNX session.
    for label in ("cold","warm"):
        job_id=f"bench-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{label}"
        p0=time.perf_counter(); prep=rembg_worker.prepare.remote(job_id); pwall=time.perf_counter()-p0
        jobs.append({"label":label,"job_id":job_id,"prepare":prep,"prepare_client_wall":round(pwall,3)})
        print(f"PREPARED_{label.upper()}",json.dumps(jobs[-1],ensure_ascii=False,indent=2),flush=True)

    # Heavy GPU calls are now consecutive.
    for item in jobs:
        g0=time.perf_counter(); gen=worker.generate.remote(item["job_id"]); gwall=time.perf_counter()-g0
        item["sam3d"]=gen; item["sam3d_client_wall"]=round(gwall,3)
        print(f"SAM3D_{item['label'].upper()}",json.dumps(item,ensure_ascii=False,indent=2),flush=True)

    same_instance=jobs[0]["sam3d"]["instance_id"]==jobs[1]["sam3d"]["instance_id"]
    reuse={
        "same_instance":same_instance,
        "cold_instance":jobs[0]["sam3d"]["instance_id"],
        "warm_instance":jobs[1]["sam3d"]["instance_id"],
        "cold_client_wall":jobs[0]["sam3d_client_wall"],
        "warm_client_wall":jobs[1]["sam3d_client_wall"],
        "model_load_seconds":jobs[0]["sam3d"]["resident_model_load_seconds"],
        "cold_inference_seconds":jobs[0]["sam3d"]["inference_seconds"],
        "warm_inference_seconds":jobs[1]["sam3d"]["inference_seconds"],
    }
    print("SAM3D_WARM_REUSE",json.dumps(reuse,ensure_ascii=False),flush=True)
    if not same_instance:
        raise RuntimeError("warm benchmark did not reuse the resident SAM3D instance")

    # Downstream stages happen after the reuse measurement, so the heavy SAM3D
    # worker can naturally idle then scale to zero while xatlas uses CPU only.
    for item in jobs:
        down0=time.perf_counter()
        x0=time.perf_counter(); xr=mesh_worker.process.remote(item["job_id"]); xwall=time.perf_counter()-x0
        b0=time.perf_counter(); br=lite_gpu_bake.remote(item["job_id"]); bwall=time.perf_counter()-b0
        f0=time.perf_counter(); fr=cpu_finalize.remote(item["job_id"]); fwall=time.perf_counter()-f0
        item.update({
            "xatlas":xr,"xatlas_client_wall":round(xwall,3),
            "lite_gpu":br,"lite_gpu_client_wall":round(bwall,3),
            "final":fr,"final_client_wall":round(fwall,3),
            "downstream_wall":round(time.perf_counter()-down0,3),
        })
        print(f"PIPELINE_{item['label'].upper()}",json.dumps(item,ensure_ascii=False,indent=2),flush=True)

    print("SPLIT_BENCHMARK_SUMMARY",json.dumps({"reuse":reuse,"jobs":jobs},ensure_ascii=False,indent=2),flush=True)
