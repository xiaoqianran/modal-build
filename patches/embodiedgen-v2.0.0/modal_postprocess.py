"""Headless Modal post-processing in isolated CUDA subprocesses.

Each heavy CUDA stage is launched as its own Python process. This is deliberate:
SAM3D, gsplat, Warp/nvdiffrast and mesh repair can leave process-global CUDA
state that is harmless alone but pathological when mixed in one interpreter.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, env: dict | None = None, timeout: int | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, timeout=timeout)


def stage_render(args):
    """Render with exact raw gsplat path, with per-line diagnostics."""
    import time
    def mark(x): print(f"STAGE_RENDER {time.time():.3f} {x}", flush=True)
    mark("start")
    import math
    mark("math")
    import numpy as np
    mark("numpy")
    import torch
    mark(f"torch cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    from PIL import Image
    mark("PIL")
    from gsplat import rasterization
    mark("gsplat")
    from embodied_gen.data.utils import CameraSetting, init_kal_camera
    mark("camera imports")
    from embodied_gen.models.gs_model import load_gs_model
    mark("gs_model import")

    root = Path(args.output_root)
    aligned = root / f"{args.filename}_gs_aligned.ply"
    color = root / "color.png"
    mark(f"paths aligned={aligned} exists={aligned.exists()}")
    cp = CameraSetting(num_images=4,elevation=[30,-30],distance=5,resolution_hw=(512,512),fov=math.radians(30),device="cuda")
    mark("CameraSetting")
    cam = init_kal_camera(cp, flip_az=True)
    mark("init_kal_camera")
    mv = cam.view_matrix(); mv[:, :3, 3] = -mv[:, :3, 3]
    mark("view_matrix")
    c2ws = [torch.linalg.inv(m.to("cuda")) for m in mv]
    K = torch.tensor(cp.Ks, device="cuda")
    mark("c2ws/K")
    model = load_gs_model(str(aligned), pre_quat=[0.0,0.0,1.0,0.0])
    mark(f"load_gs_model N={model._means.shape[0]}")
    images=[]
    for i,c2w in enumerate(c2ws):
        mark(f"view={i} get_gaussians begin")
        gs=model.get_gaussians(c2w, apply_activate=True)
        mark(f"view={i} before rasterization")
        t0=time.perf_counter()
        renders,alphas,_=rasterization(
            means=gs._means, quats=gs._quats, scales=gs._scales,
            opacities=gs._opacities.squeeze(), colors=gs._rgbs,
            viewmats=torch.linalg.inv(c2w)[None,...], Ks=K[None,...],
            width=512,height=512,packed=False,absgrad=True,sparse_grad=False,
            rasterize_mode="antialiased",near_plane=0.01,far_plane=1_000_000_000,
            radius_clip=0.0,render_mode="RGB+ED")
        mark(f"view={i} raster returned dt={time.perf_counter()-t0:.3f}")
        torch.cuda.synchronize()
        mark(f"view={i} sync dt={time.perf_counter()-t0:.3f}")
        rgb=(renders[0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy()
        alpha=(alphas[0].clamp(0,1)*255).to(torch.uint8).cpu().numpy()
        rgba=np.concatenate([rgb,alpha],axis=-1)
        images.append(Image.fromarray(rgba,mode="RGBA"))
        mark(f"view={i} PIL done")
    grid=Image.new("RGBA",(1024,1024),(0,0,0,0))
    for i,img in enumerate(images):
        grid.paste(img.resize((512,512)),((i%2)*512,(i//2)*512))
    mark("grid composed")
    grid.save(color)
    mark(f"grid saved size={color.stat().st_size if color.exists() else -1}")
    if not color.exists() or color.stat().st_size==0:
        raise RuntimeError("render stage produced no color.png")
    print(f"STAGE_RENDER_OK {color} {color.stat().st_size}", flush=True)


def stage_video_mesh(args):
    """Export SAM3D mesh and make orbit preview with the proven gsplat backend.

    TRELLIS' legacy GaussianRenderer depends on diff_gaussian_rasterization; using
    gsplat here avoids compiling a second CUDA rasterizer solely for preview video.
    """
    import math, pickle, time
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import trimesh
    from gsplat import rasterization
    from embodied_gen.data.utils import CameraSetting, init_kal_camera
    from embodied_gen.models.gs_model import load_gs_model

    root=Path(args.output_root); fn=args.filename
    with (root/f"{fn}_state.pkl").open("rb") as f: state=pickle.load(f)
    verts=np.asarray(state["mesh"]["vertices"]); faces=np.asarray(state["mesh"]["faces"])
    print(f"STATE GS={len(state['gaussian']['_xyz'])} verts={len(verts)} faces={len(faces)}",flush=True)

    # Export the actual SAM3D mesh using upstream orientation transforms.
    rot=np.array([[0,0,-1],[0,1,0],[1,0,0]])
    add=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    mesh=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    mesh.vertices=mesh.vertices @ add
    mesh.vertices=mesh.vertices @ rot
    obj=root/f"{fn}.obj"; mesh.export(obj)
    print(f"RAW_OBJ_OK size={obj.stat().st_size}",flush=True)

    # Orbit color preview from aligned PLY with gsplat only.
    model=load_gs_model(str(root/f"{fn}_gs_aligned.ply"),pre_quat=[0.,0.,1.,0.])
    cp=CameraSetting(num_images=args.video_frames,elevation=[0],distance=5,resolution_hw=(512,512),fov=math.radians(30),device="cuda")
    cam=init_kal_camera(cp,flip_az=True); mv=cam.view_matrix(); mv[:,:3,3]=-mv[:,:3,3]
    K=torch.tensor(cp.Ks,device="cuda")
    frames=[]; t0=time.perf_counter()
    for i,m in enumerate(mv):
        c2w=torch.linalg.inv(m.to("cuda")); gs=model.get_gaussians(c2w,apply_activate=True)
        renders,_,_=rasterization(means=gs._means,quats=gs._quats,scales=gs._scales,
            opacities=gs._opacities.squeeze(),colors=gs._rgbs,
            viewmats=torch.linalg.inv(c2w)[None,...],Ks=K[None,...],width=512,height=512,
            packed=False,absgrad=True,sparse_grad=False,rasterize_mode="antialiased",
            near_plane=0.01,far_plane=1_000_000_000,radius_clip=0.0,render_mode="RGB")
        torch.cuda.synchronize()
        frame=(renders[0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy()
        frames.append(frame)
        if i==0 or (i+1)%15==0: print(f"VIDEO_GS {i+1}/{len(mv)}",flush=True)
    video=root/"gs_mesh.mp4"; imageio.mimsave(str(video),frames,fps=30,codec="libx264")
    print(f"STAGE_VIDEO_MESH_OK video={video.stat().st_size} obj={obj.stat().st_size} render_s={time.perf_counter()-t0:.2f}",flush=True)


def stage_backproject(args):
    """Texture/mesh stage isolated from both SAM3D and preview rendering."""
    import trimesh
    from embodied_gen.data.backproject_v3 import entrypoint as backproject_api

    root = Path(args.output_root)
    aligned = root / f"{args.filename}_gs_aligned.ply"
    obj = root / f"{args.filename}.obj"
    kwargs = dict(
        gs_path=str(aligned),
        mesh_path=str(obj),
        output_path=str(obj),
        # Cost-safe validation profile using the exact horizontal orbit already
        # proven stable for this real SAM3D asset (60/60 frames succeeded).
        num_images=24,
        elevation=[0],
        distance=5.0,
        skip_fix_mesh=args.skip_fix_mesh,
        texture_size=args.texture_size,
        baker_mode="fast" if args.fast_bake else "opt",
        delight=False,
    )
    mesh = backproject_api(**kwargs)
    glb = root / f"{args.filename}.glb"
    mesh.export(glb)
    if not obj.exists() or not glb.exists():
        raise RuntimeError("backproject stage output missing")
    # Validate parse immediately in the producing process.
    m = trimesh.load(glb, force="scene")
    ngeom = len(m.geometry) if hasattr(m, "geometry") else 1
    if ngeom < 1:
        raise RuntimeError("GLB contains no geometry")
    print(f"STAGE_BACKPROJECT_OK glb={glb.stat().st_size} geometry={ngeom}", flush=True)


def stage_urdf(args):
    """URDF and QA stage. GPT is forced onto the project's built-in fallback."""
    from glob import glob
    from shutil import copy, copytree, rmtree
    import trimesh

    from embodied_gen.models.gs_model import GaussianOperator
    from embodied_gen.utils.gpt_clients import GPT_CLIENT
    from embodied_gen.utils.tags import VERSION
    from embodied_gen.validators.quality_checkers import (
        BaseChecker,
        ImageAestheticChecker,
        ImageSegChecker,
        MeshGeoChecker,
    )
    from embodied_gen.validators.urdf_convertor import URDFGenerator

    root = Path(args.output_root)
    fn = args.filename
    GPT_CLIENT.query = lambda *a, **k: None
    obj = root / f"{fn}.obj"
    aligned = root / f"{fn}_gs_aligned.ply"

    urdf_gen = URDFGenerator(
        GPT_CLIENT,
        render_view_num=4,
        decompose_convex=not args.disable_decompose_convex,
    )
    urdf_root = root / f"URDF_{fn}"
    urdf_path = urdf_gen(
        mesh_path=str(obj),
        output_root=str(urdf_root),
        version=VERSION,
        gs_model=f"{urdf_gen.output_mesh_dir}/{fn}_gs.ply",
    )
    real_height = urdf_gen.get_attr_from_urdf(urdf_path, attr_name="real_height")
    out_gs = urdf_root / urdf_gen.output_mesh_dir / f"{fn}_gs.ply"
    GaussianOperator.resave_ply(
        in_ply=str(aligned), out_ply=str(out_gs), real_height=real_height, device="cpu"
    )
    mesh_out = urdf_root / urdf_gen.output_mesh_dir / f"{fn}.obj"
    trimesh.load(mesh_out).export(str(mesh_out).replace(".obj", ".glb"))

    # GPT-backed checkers return through fallback immediately. The local aesthetic
    # checker is optional in cheap smoke mode because it can trigger another model download.
    checkers = [MeshGeoChecker(GPT_CLIENT), ImageSegChecker(GPT_CLIENT)]
    if not args.skip_aesthetic:
        checkers.append(ImageAestheticChecker())
    image_dir = urdf_root / urdf_gen.output_render_dir / "image_color"
    image_paths = glob(str(image_dir / "*.png"))
    images_list = []
    for checker in checkers:
        if isinstance(checker, ImageSegChecker):
            images = [str(root / f"{fn}_raw.png"), str(root / f"{fn}_cond.png")]
        else:
            from PIL import Image
            pil = [Image.open(x).convert("RGB") for x in image_paths[:4]]
            if pil:
                grid = Image.new("RGB", (1024, 1024), (0, 0, 0))
                for i, im in enumerate(pil):
                    grid.paste(im.resize((512, 512)), ((i % 2) * 512, (i // 2) * 512))
                images = [grid]
            else:
                images = []
        images_list.append(images)
    qa = BaseChecker.validate(checkers, images_list)
    urdf_gen.add_quality_tag(urdf_path, qa)

    result = root / "result"
    if result.exists():
        rmtree(result, ignore_errors=True)
    result.mkdir(parents=True, exist_ok=True)
    copy(urdf_path, result / Path(urdf_path).name)
    copytree(urdf_root / urdf_gen.output_mesh_dir, result / urdf_gen.output_mesh_dir)
    copy(root / "gs_mesh.mp4", result / "video.mp4")
    print(f"STAGE_URDF_OK {result}", flush=True)


def validate(args):
    """CPU-ish structural validation; does not load generation models."""
    import json
    import xml.etree.ElementTree as ET
    import trimesh

    root = Path(args.output_root)
    result = root / "result"
    report = {"root": str(root), "files": {}, "checks": {}}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            report["files"][str(p.relative_to(root))] = p.stat().st_size

    required = [
        root / f"{args.filename}_gs.ply",
        root / f"{args.filename}_gs_aligned.ply",
        root / f"{args.filename}.obj",
        root / f"{args.filename}.glb",
        root / "color.png",
        root / "gs_mesh.mp4",
    ]
    for p in required:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"required artifact missing/empty: {p}")

    obj = trimesh.load(root / f"{args.filename}.obj", force="mesh")
    report["checks"]["obj_vertices"] = int(len(obj.vertices))
    report["checks"]["obj_faces"] = int(len(obj.faces))
    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        raise RuntimeError("OBJ has no geometry")

    glb = trimesh.load(root / f"{args.filename}.glb", force="scene")
    report["checks"]["glb_geometries"] = int(len(glb.geometry))
    if not glb.geometry:
        raise RuntimeError("GLB has no geometry")

    urdfs = list(result.glob("*.urdf"))
    if len(urdfs) != 1:
        raise RuntimeError(f"expected one result URDF, got {urdfs}")
    tree = ET.parse(urdfs[0])
    refs = [e.attrib.get("filename") for e in tree.findall(".//mesh") if e.attrib.get("filename")]
    report["checks"]["urdf_mesh_refs"] = refs
    for ref in refs:
        rp = urdfs[0].parent / ref
        if not rp.exists():
            raise RuntimeError(f"URDF mesh ref does not exist: {ref} -> {rp}")

    # PLY vertex declaration.
    with (root / f"{args.filename}_gs.ply").open("rb") as f:
        header = f.read(8192).decode("ascii", "ignore")
    verts = None
    for line in header.splitlines():
        if line.startswith("element vertex "):
            verts = int(line.split()[-1])
            break
    report["checks"]["ply_vertices"] = verts
    if not verts:
        raise RuntimeError("PLY vertex count missing/zero")

    # ffprobe if available.
    try:
        raw = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(root / "gs_mesh.mp4")],
            text=True,
        ).strip()
        report["checks"]["video_duration"] = float(raw)
        if float(raw) <= 0:
            raise RuntimeError("video duration <= 0")
    except FileNotFoundError:
        report["checks"]["video_duration"] = "ffprobe unavailable"

    report_path = root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["checks"], indent=2, ensure_ascii=False), flush=True)
    print(f"VALIDATION_OK {report_path}", flush=True)


def orchestrate(args):
    base = [sys.executable, "-m", "embodied_gen.scripts.modal_postprocess"]
    common = ["--output_root", args.output_root, "--filename", args.filename]
    # Each CUDA-heavy stage gets a fresh process/context.
    run(base + common + ["--stage", "render"], timeout=120)
    run(base + common + ["--stage", "video_mesh", "--video_frames", str(args.video_frames)], timeout=300)
    back = base + common + ["--stage", "backproject", "--texture_size", str(args.texture_size)]
    if args.fast_bake:
        back.append("--fast_bake")
    if args.skip_fix_mesh:
        back.append("--skip_fix_mesh")
    run(back, timeout=900)
    urdf = base + common + ["--stage", "urdf"]
    if args.disable_decompose_convex:
        urdf.append("--disable_decompose_convex")
    if args.skip_aesthetic:
        urdf.append("--skip_aesthetic")
    run(urdf, timeout=300)
    run(base + common + ["--stage", "validate"], timeout=120)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--filename", default="sample_00")
    ap.add_argument("--stage", choices=["all", "render", "video_mesh", "backproject", "urdf", "validate"], default="all")
    ap.add_argument("--video_frames", type=int, default=60)
    ap.add_argument("--texture_size", type=int, default=1024)
    ap.add_argument("--fast_bake", action="store_true")
    ap.add_argument("--skip_fix_mesh", action="store_true")
    ap.add_argument("--disable_decompose_convex", action="store_true")
    ap.add_argument("--skip_aesthetic", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.stage == "all": orchestrate(a)
    elif a.stage == "render": stage_render(a)
    elif a.stage == "video_mesh": stage_video_mesh(a)
    elif a.stage == "backproject": stage_backproject(a)
    elif a.stage == "urdf": stage_urdf(a)
    elif a.stage == "validate": validate(a)
