"""EmbodiedGen P3-SAM part-segmentation runtime for Modal L40S.

This is the GPT-free core of EmbodiedGen affordance annotation. It consumes a
completed EmbodiedGen job's URDF/mesh and produces deterministic part labels +
a segmented GLB. Full semantic annotation and grasp evaluation remain separate
because upstream requires a configured GPT endpoint and additional simulation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import modal

APP_NAME = "modal-3d-embodiedgen-affordance"
EMBODIEDGEN_COMMIT = "cc3015ca5ccdacf94df3428d9e65f79375982216"
HUNYUAN3D_PART_COMMIT = "e96be065375438962375b55326416291342958a7"
HUNYUAN3D_PART_MODEL_REVISION = "677174466c53571e8bacd5050dff5948734a1a4d"
SONATA_MODEL_REVISION = "df99897472c09f91ba9288da0a034aacffc0b010"
P3SAM_WEIGHT_SHA256 = "eb76550cfbe06f154c6e9b17167ccfc28222bb4a216ec7b12ac2bf7d762de38c"
SONATA_WEIGHT_SHA256 = "c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50"
GRASPGEN_COMMIT = "a56d518f3b76ea2a432b5b838b3c68027d29be49"
GRASPGEN_MODELS_REVISION = "ec1ccbb5eec0680db669246ac312a3636f16ee43"
GRASPGEN_CONFIG_SHA256 = "3b666d28ffb91001ddb6ba24a2e0c11458478a986b808b493cf6fa9a987c2abd"
GRASPGEN_GEN_SHA256 = "0597583b89b322d42ceb4e596967d6ed68d1b56cba4039895909ccd5bdc66eff"
GRASPGEN_DIS_SHA256 = "e47d703c63b54c2d11fbc1effd43898f251b4147250888541e3b16e9c0d19e1c"
AFFORDANCE_RELEASE_TAG = "embodiedgen-v2.0.0-affordance-py310-cu126-torch280-sm89-v1"
AFFORDANCE_RELEASE = (
    "https://github.com/xiaoqianran/modal-build/releases/download/"
    + AFFORDANCE_RELEASE_TAG
)
AFFORDANCE_WHEELS = {
    "chamfer_3d-0.0.0-cp310-cp310-linux_x86_64.whl": (
        "600791e5ff88f988114848ed2888cbf0f35513291247c439e3e65f94761de028"
    ),
    "pointnet2_ops-3.0.0-cp310-cp310-linux_x86_64.whl": (
        "052ba8a1cf9c4de22154b90bd9c4e49e1fc51edc6cd7e02af527ee540c834bc2"
    ),
    "torch_scatter-2.1.2+pt28cu126-cp310-cp310-linux_x86_64.whl": (
        "3585a1ef1f4886d037a76a21ff987fbcac354805dbae42c7992b0b6d7cf8ad54"
    ),
}
P3SAM_WEIGHT = Path("/weights/affordance/p3sam/p3sam.safetensors")
SONATA_WEIGHT = Path("/weights/affordance/sonata/sonata.pth")
WEIGHT_MANIFEST = Path("/weights/affordance/manifest.json")
GRASPGEN_ROOT = Path("/weights/affordance/graspgen/franka_panda")
GRASPGEN_CONFIG = GRASPGEN_ROOT / "graspgen_franka_panda.yml"
GRASPGEN_GEN = GRASPGEN_ROOT / "graspgen_franka_panda_gen.pth"
GRASPGEN_DIS = GRASPGEN_ROOT / "graspgen_franka_panda_dis.pth"
GRASPGEN_MANIFEST = GRASPGEN_ROOT / "manifest.json"
JOB_ROOT = Path("/artifacts/embodiedgen/jobs")

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("modal-3d-embodiedgen-weights", create_if_missing=True)
artifacts = modal.Volume.from_name("modal-3d-artifacts", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-runtime-ubuntu22.04", add_python="3.10")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "HF_HUB_OFFLINE": "1",
        }
    )
    .apt_install("git", "curl", "libgomp1", "clang")
    .run_commands(
        "! command -v nvcc",
        "python -m pip install --upgrade 'pip>=25' setuptools==80.10.2 wheel packaging",
        "python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126",
        "python -m pip install numpy==1.26.4 trimesh==4.5.3 scipy scikit-learn fpsample==1.0.2 numba tqdm addict timm==1.0.15 spconv-cu126==2.3.7 huggingface_hub==0.34.4 omegaconf==2.3.0",
    )
    .run_commands(
        "git init /workspace/Hunyuan3D-Part && cd /workspace/Hunyuan3D-Part && git remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git",
        f"cd /workspace/Hunyuan3D-Part && git fetch --depth 1 origin {HUNYUAN3D_PART_COMMIT} && git checkout --detach FETCH_HEAD",
    )
)

# Consume only L40S-smoke-validated immutable wheels. Runtime image has no nvcc.
for wheel_name, wheel_sha256 in AFFORDANCE_WHEELS.items():
    image = image.run_commands(
        f"curl -fL '{AFFORDANCE_RELEASE}/{wheel_name}' -o '/tmp/{wheel_name}'",
        f"echo '{wheel_sha256}  /tmp/{wheel_name}' | sha256sum -c -",
        f"python -m pip install --no-deps '/tmp/{wheel_name}'",
        f"rm -f '/tmp/{wheel_name}'",
    )

# Standalone P3-SAM must be fully offline: point Sonata at the pinned Volume file
# and disable flash attention. Exact source text is asserted before replacement.
image = image.run_commands(
    "python - <<'PY'\n"
    "import base64\n"
    "from pathlib import Path\n"
    "p=Path('/workspace/Hunyuan3D-Part/P3-SAM/model.py')\n"
    "s=p.read_text()\n"
    "old=base64.b64decode('c2VsZi5zb25hdGEgPSBzb25hdGEubG9hZCgic29uYXRhIiwgcmVwb19pZD0iZmFjZWJvb2svc29uYXRhIiwgZG93bmxvYWRfcm9vdD0nL3Jvb3Qvc29uYXRhJyk=').decode()\n"
    "new=base64.b64decode('c2VsZi5zb25hdGEgPSBzb25hdGEubG9hZCgiL3dlaWdodHMvYWZmb3JkYW5jZS9zb25hdGEvc29uYXRhLnB0aCIsIGN1c3RvbV9jb25maWc9eyJlbmFibGVfZmxhc2giOiBGYWxzZX0p').decode()\n"
    "if s.count(old) != 1: raise SystemExit('unexpected P3-SAM Sonata loader source')\n"
    "p.write_text(s.replace(old,new,1))\n"
    "PY",
    "grep -Fq 'custom_config={\"enable_flash\": False}' /workspace/Hunyuan3D-Part/P3-SAM/model.py",
    "! command -v nvcc",
).workdir("/workspace/Hunyuan3D-Part/P3-SAM/demo")

# GraspGen is kept on a separate image branch so its older pinned diffusers / HF
# runtime cannot perturb the already-validated P3-SAM environment.
grasp_image = (
    image
    .run_commands(
        "python -m pip install huggingface_hub==0.25.2 diffusers==0.11.1 h5py==3.11.0 pyyaml==6.0.2",
        "git init /workspace/GraspGen && cd /workspace/GraspGen && git remote add origin https://github.com/NVlabs/GraspGen.git",
        f"cd /workspace/GraspGen && git fetch --depth 1 origin {GRASPGEN_COMMIT} && git checkout --detach FETCH_HEAD",
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "p=Path('/workspace/GraspGen/grasp_gen/models/discriminator.py')\n"
        "s=p.read_text()\n"
        "old='from grasp_gen.dataset.dataset import MAPPING_ID2NAME'\n"
        "new='MAPPING_ID2NAME = {}  # training-only metric map; raw inference has no labels'\n"
        "if s.count(old) != 1: raise SystemExit('unexpected GraspGen discriminator import source')\n"
        "p.write_text(s.replace(old,new,1))\n"
        "robot=Path('/workspace/GraspGen/grasp_gen/robot.py')\n"
        "rs=robot.read_text()\n"
        "robot_old='from grasp_gen.dataset.eval_utils import load_urdf_scene'\n"
        "if rs.count(robot_old) != 1: raise SystemExit('unexpected GraspGen robot import source')\n"
        "robot.write_text(rs.replace(robot_old, '# raw-inference runtime: eval_utils/yourdfpy is not required', 1))\n"
        "PY",
        "grep -Fq 'enable_flash=False' /workspace/GraspGen/grasp_gen/models/generator.py",
        "grep -Fq 'enable_flash=False' /workspace/GraspGen/grasp_gen/models/discriminator.py",
        "! command -v nvcc",
    )
    .workdir("/workspace/GraspGen")
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_glb_document(glb_path: Path):
    """Parse a GLB v2 JSON/BIN pair without adding a runtime dependency."""
    import json as _json
    import struct

    data = glb_path.read_bytes()
    if len(data) < 20:
        raise RuntimeError(f"GLB is too small: {glb_path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise RuntimeError(
            f"invalid GLB header: magic={magic!r} version={version} "
            f"declared={declared_length} actual={len(data)}"
        )
    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset < len(data):
        if offset + 8 > len(data):
            raise RuntimeError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            raise RuntimeError("truncated GLB chunk")
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            if json_chunk is not None:
                raise RuntimeError("GLB contains multiple JSON chunks")
            json_chunk = chunk
        elif chunk_type == 0x004E4942:
            if bin_chunk is not None:
                raise RuntimeError("GLB contains multiple BIN chunks")
            bin_chunk = chunk
    if json_chunk is None or bin_chunk is None:
        raise RuntimeError("GLB requires exactly one JSON and one BIN chunk")
    doc = _json.loads(json_chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
    return doc, bin_chunk


def _glb_accessor_array(doc: dict, bin_chunk: bytes, accessor_index: int):
    import numpy as np

    accessors = doc.get("accessors", [])
    views = doc.get("bufferViews", [])
    if not (0 <= accessor_index < len(accessors)):
        raise RuntimeError(f"invalid GLB accessor index: {accessor_index}")
    accessor = accessors[accessor_index]
    if accessor.get("sparse") is not None:
        raise RuntimeError("sparse GLB accessors are not supported for segmentation alignment")
    if "bufferView" not in accessor:
        raise RuntimeError("segmentation alignment requires buffer-backed GLB accessors")
    view = views[accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise RuntimeError("segmentation alignment only supports the GLB BIN buffer")

    component_types = {
        5120: np.dtype("i1"),
        5121: np.dtype("u1"),
        5122: np.dtype("<i2"),
        5123: np.dtype("<u2"),
        5125: np.dtype("<u4"),
        5126: np.dtype("<f4"),
    }
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    dtype = component_types.get(accessor.get("componentType"))
    width = components.get(accessor.get("type"))
    if dtype is None or width is None:
        raise RuntimeError(
            f"unsupported GLB accessor format: componentType={accessor.get('componentType')} "
            f"type={accessor.get('type')}"
        )
    count = int(accessor.get("count", 0))
    if count <= 0:
        raise RuntimeError("GLB accessor count must be positive")
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    packed_stride = dtype.itemsize * width
    stride = int(view.get("byteStride", packed_stride))
    if stride < packed_stride:
        raise RuntimeError(f"invalid GLB accessor stride: {stride} < {packed_stride}")
    end = byte_offset + (count - 1) * stride + packed_stride
    if byte_offset < 0 or end > len(bin_chunk):
        raise RuntimeError("GLB accessor extends outside BIN chunk")
    array = np.ndarray(
        shape=(count, width),
        dtype=dtype,
        buffer=bin_chunk,
        offset=byte_offset,
        strides=(stride, dtype.itemsize),
    ).copy()
    if width == 1:
        return array[:, 0]
    return array


def _build_agentscape_segmentation_evidence(
    *,
    source_obj: Path,
    primary_glb: Path,
    face_labels,
    source_job_id: str,
) -> dict:
    """Align P3-SAM OBJ face labels to the exact primary GLB primitive triangles."""
    import numpy as np

    vertices = []
    faces = []
    with source_obj.open("r", encoding="utf-8", errors="strict") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("v "):
                values = [float(x) for x in line.split()[1:4]]
                if len(values) != 3 or not np.isfinite(values).all():
                    raise RuntimeError(f"invalid OBJ vertex: {line[:120]}")
                vertices.append(values)
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) != 3:
                    raise RuntimeError("compiler-native segmentation requires triangulated OBJ")
                indices = []
                for token in tokens:
                    raw_index = int(token.split("/", 1)[0])
                    if raw_index <= 0:
                        raise RuntimeError("compiler-native segmentation requires positive OBJ indices")
                    indices.append(raw_index - 1)
                faces.append(indices)
    obj_vertices = np.asarray(vertices, dtype=np.float64)
    obj_faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64).reshape(-1)
    if len(obj_vertices) == 0 or len(obj_faces) == 0:
        raise RuntimeError("source OBJ is empty")
    if len(labels) != len(obj_faces):
        raise RuntimeError(
            f"segmentation labels do not match source OBJ faces: {len(labels)} != {len(obj_faces)}"
        )
    if np.any(labels < 0):
        raise RuntimeError("compiler-native segmentation cannot contain unlabeled faces")

    doc, bin_chunk = _read_glb_document(primary_glb)
    nodes = doc.get("nodes", [])
    mesh_nodes = [(i, node) for i, node in enumerate(nodes) if "mesh" in node]
    if len(mesh_nodes) != 1:
        raise RuntimeError(
            f"compiler-native segmentation requires exactly one GLB mesh node, got {len(mesh_nodes)}"
        )
    _, node = mesh_nodes[0]
    source_node = str(node.get("name", "")).strip()
    if not source_node:
        raise RuntimeError("primary GLB mesh node requires a stable non-empty name")
    if sum(1 for item in nodes if str(item.get("name", "")).strip() == source_node) != 1:
        raise RuntimeError(f"primary GLB source node name is not unique: {source_node}")
    identity = np.eye(4, dtype=np.float64)
    if "matrix" in node:
        matrix_values = np.asarray(node["matrix"], dtype=np.float64)
        if matrix_values.size != 16 or not np.allclose(
            matrix_values.reshape((4, 4), order="F"), identity, atol=1e-8, rtol=0
        ):
            raise RuntimeError("primary GLB mesh node has a non-identity matrix transform")
    else:
        translation = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
        rotation = np.asarray(node.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
        scale = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
        if not (
            np.allclose(translation, [0, 0, 0], atol=1e-8, rtol=0)
            and np.allclose(rotation, [0, 0, 0, 1], atol=1e-8, rtol=0)
            and np.allclose(scale, [1, 1, 1], atol=1e-8, rtol=0)
        ):
            raise RuntimeError("primary GLB mesh node has a non-identity TRS transform")

    meshes = doc.get("meshes", [])
    mesh_index = int(node["mesh"])
    if not (0 <= mesh_index < len(meshes)):
        raise RuntimeError(f"primary GLB node references invalid mesh index: {mesh_index}")
    primitives = meshes[mesh_index].get("primitives", [])
    if not primitives:
        raise RuntimeError("primary GLB mesh has no primitives")

    source_face_by_key = {}
    for face_index, tri in enumerate(obj_faces):
        key = tuple(sorted(int(x) for x in tri))
        if key in source_face_by_key:
            raise RuntimeError(f"source OBJ contains duplicate triangle vertex set: {key}")
        source_face_by_key[key] = face_index

    primitive_specs = []
    used_source_faces = set()
    max_vertex_error = 0.0
    for primitive_index, primitive in enumerate(primitives):
        if int(primitive.get("mode", 4)) != 4:
            raise RuntimeError(f"GLB primitive {primitive_index} is not TRIANGLES")
        if primitive.get("extensions"):
            raise RuntimeError(f"GLB primitive {primitive_index} uses unsupported extensions")
        attributes = primitive.get("attributes", {})
        if "POSITION" not in attributes:
            raise RuntimeError(f"GLB primitive {primitive_index} has no POSITION accessor")
        positions = np.asarray(
            _glb_accessor_array(doc, bin_chunk, int(attributes["POSITION"])), dtype=np.float64
        )
        if positions.shape != obj_vertices.shape:
            raise RuntimeError(
                f"GLB primitive {primitive_index} POSITION shape {positions.shape} "
                f"does not preserve source OBJ vertices {obj_vertices.shape}"
            )
        vertex_error = float(np.max(np.abs(positions - obj_vertices)))
        max_vertex_error = max(max_vertex_error, vertex_error)
        if vertex_error > 1e-6:
            raise RuntimeError(
                f"GLB primitive {primitive_index} vertex identity drift exceeds tolerance: {vertex_error}"
            )
        if "indices" in primitive:
            indices = np.asarray(
                _glb_accessor_array(doc, bin_chunk, int(primitive["indices"])), dtype=np.int64
            ).reshape(-1)
        else:
            indices = np.arange(len(positions), dtype=np.int64)
        if len(indices) % 3:
            raise RuntimeError(f"GLB primitive {primitive_index} index count is not divisible by 3")
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(obj_vertices)):
            raise RuntimeError(f"GLB primitive {primitive_index} has out-of-range vertex indices")

        primitive_labels = []
        for offset in range(0, len(indices), 3):
            key = tuple(sorted(int(x) for x in indices[offset : offset + 3]))
            source_face_index = source_face_by_key.get(key)
            if source_face_index is None:
                raise RuntimeError(
                    f"GLB primitive {primitive_index} triangle has no source OBJ face: {key}"
                )
            if source_face_index in used_source_faces:
                raise RuntimeError(
                    f"source OBJ face is referenced more than once by primary GLB: {source_face_index}"
                )
            used_source_faces.add(source_face_index)
            primitive_labels.append(str(int(labels[source_face_index])))
        primitive_specs.append(
            {"primitive": primitive_index, "faceLabels": primitive_labels}
        )

    if len(used_source_faces) != len(obj_faces):
        missing = len(obj_faces) - len(used_source_faces)
        raise RuntimeError(f"primary GLB does not cover all source OBJ faces: missing={missing}")

    segment_ids, segment_counts = np.unique(labels, return_counts=True)
    segments = [
        {"id": str(int(segment_id)), "faceCount": int(count)}
        for segment_id, count in zip(segment_ids, segment_counts)
    ]
    evidence = {
        "version": 1,
        "source": "embodiedgen/p3sam",
        "faceCount": int(len(labels)),
        "segments": segments,
        "artifact": {
            "role": "primary_glb",
            "path": str(primary_glb.relative_to(JOB_ROOT / source_job_id)),
            "sha256": _sha256(primary_glb),
            "bytes": primary_glb.stat().st_size,
            "sourceObjSha256": _sha256(source_obj),
            "alignment": {
                "strategy": "verified-vertex-identity-triangle-index-set",
                "maxVertexAbsError": max_vertex_error,
            },
        },
        "materialization": {
            "sourceNode": source_node,
            "primitives": primitive_specs,
        },
    }
    return evidence


def _valid_job_id(job_id: str) -> bool:
    return bool(re.fullmatch(r"job-[0-9a-f]{32}", job_id))


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/weights": weights, "/artifacts": artifacts},
    timeout=60 * 60,
    cpu=8.0,
    memory=32768,
    max_containers=1,
    scaledown_window=30,
)
def segment_job(
    source_job_id: str,
    point_num: int = 100000,
    prompt_num: int = 400,
    prompt_bs: int = 8,
) -> dict:
    """Run GPT-free P3-SAM segmentation on a succeeded EmbodiedGen job."""
    import os
    import sys
    import time

    import numpy as np
    import torch
    import trimesh

    if not _valid_job_id(source_job_id):
        raise ValueError(f"invalid source job id: {source_job_id!r}")
    if not (1000 <= point_num <= 200000):
        raise ValueError("point_num must be in 1000..200000")
    if not (8 <= prompt_num <= 800):
        raise ValueError("prompt_num must be in 8..800")
    if not (1 <= prompt_bs <= 64):
        raise ValueError("prompt_bs must be in 1..64")

    weights.reload()
    artifacts.reload()
    if not P3SAM_WEIGHT.is_file() or not SONATA_WEIGHT.is_file() or not WEIGHT_MANIFEST.is_file():
        raise RuntimeError("affordance weights are not preloaded; run preload_affordance_weights")
    weight_manifest = json.loads(WEIGHT_MANIFEST.read_text())
    if weight_manifest.get("hunyuan3d_part_model_revision") != HUNYUAN3D_PART_MODEL_REVISION:
        raise RuntimeError("P3-SAM model revision marker mismatch")
    if weight_manifest.get("sonata_model_revision") != SONATA_MODEL_REVISION:
        raise RuntimeError("Sonata model revision marker mismatch")
    expected_weight_hashes = {
        "p3sam": P3SAM_WEIGHT_SHA256,
        "sonata": SONATA_WEIGHT_SHA256,
    }
    for key, path in (("p3sam", P3SAM_WEIGHT), ("sonata", SONATA_WEIGHT)):
        expected = expected_weight_hashes[key]
        if weight_manifest[key]["sha256"] != expected:
            raise RuntimeError(f"{key} manifest hash mismatch")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{key} weight hash mismatch: {actual} != {expected}")

    source_root = JOB_ROOT / source_job_id
    source_obj = source_root / "result/mesh/sample_00.obj"
    primary_glb = source_root / "result/mesh/sample_00.glb"
    source_urdf = source_root / "result/sample_00.urdf"
    if not source_obj.is_file() or not primary_glb.is_file() or not source_urdf.is_file():
        raise FileNotFoundError(f"source job has no validated OBJ/GLB/URDF: {source_job_id}")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    if capability != (8, 9):
        raise RuntimeError(f"expected NVIDIA L40S SM89, got {device_name} capability={capability}")

    p3sam_root = "/workspace/Hunyuan3D-Part/P3-SAM"
    partgen_root = "/workspace/Hunyuan3D-Part/XPart/partgen"
    demo_root = p3sam_root + "/demo"
    for path in (demo_root, p3sam_root, partgen_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(demo_root)

    from auto_mask import AutoMask

    mesh = trimesh.load(source_obj, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid source mesh: {source_obj}")
    unit_mesh = mesh.copy()
    extent = float(np.ptp(unit_mesh.vertices, axis=0).max())
    if not np.isfinite(extent) or extent <= 0.0:
        raise RuntimeError(f"invalid source mesh extent: {extent}")
    unit_mesh.apply_scale(1.0 / extent)

    started = time.perf_counter()
    pipeline = AutoMask(
        ckpt_path=str(P3SAM_WEIGHT),
        threshold=0.95,
        post_process=True,
    )
    model_load_seconds = time.perf_counter() - started
    infer_started = time.perf_counter()
    aabb, face_ids, _ = pipeline.predict_aabb(
        unit_mesh,
        point_num=point_num,
        prompt_num=prompt_num,
        threshold=0.95,
        post_process=True,
        save_path=None,
        save_mid_res=False,
        show_info=False,
        clean_mesh_flag=False,
        seed=42,
        is_parallel=False,
        prompt_bs=prompt_bs,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - infer_started

    face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    if len(face_ids) != len(mesh.faces):
        raise RuntimeError(f"face label count mismatch: {len(face_ids)} != {len(mesh.faces)}")
    valid = sorted(int(x) for x in np.unique(face_ids) if int(x) >= 0)
    if not valid:
        raise RuntimeError("P3-SAM returned no valid parts")
    dense = np.full_like(face_ids, -1)
    for new_id, old_id in enumerate(valid):
        dense[face_ids == old_id] = new_id

    palette = np.asarray(
        [
            [230, 25, 75, 255], [60, 180, 75, 255], [255, 225, 25, 255],
            [0, 130, 200, 255], [245, 130, 48, 255], [145, 30, 180, 255],
            [70, 240, 240, 255], [240, 50, 230, 255], [210, 245, 60, 255],
            [250, 190, 212, 255], [0, 128, 128, 255], [220, 190, 255, 255],
            [170, 110, 40, 255], [255, 250, 200, 255], [128, 0, 0, 255],
            [170, 255, 195, 255], [128, 128, 0, 255], [255, 215, 180, 255],
            [0, 0, 128, 255], [128, 128, 128, 255],
        ],
        dtype=np.uint8,
    )
    face_colors = np.zeros((len(dense), 4), dtype=np.uint8)
    face_colors[:, 3] = 255
    for part_id in range(len(valid)):
        face_colors[dense == part_id] = palette[part_id % len(palette)]
    segmented = mesh.copy()
    segmented.visual = trimesh.visual.ColorVisuals(mesh=segmented, face_colors=face_colors)

    output = source_root / "affordance"
    output.mkdir(parents=True, exist_ok=True)
    glb_path = output / "mesh_part_seg.glb"
    json_path = output / "part_segmentation.json"
    report_path = output / "validation_report.json"
    agentscape_path = output / "agentscape_part_segmentation.v1.json"
    segmented.export(glb_path)

    agentscape_evidence = _build_agentscape_segmentation_evidence(
        source_obj=source_obj,
        primary_glb=primary_glb,
        face_labels=dense,
        source_job_id=source_job_id,
    )
    agentscape_path.write_text(json.dumps(agentscape_evidence, separators=(",", ":")) + "\n")

    counts = {str(part_id): int(np.sum(dense == part_id)) for part_id in range(len(valid))}
    payload = {
        "source_job_id": source_job_id,
        "backend": "P3-SAM",
        "hunyuan3d_part_commit": HUNYUAN3D_PART_COMMIT,
        "hunyuan3d_part_model_revision": HUNYUAN3D_PART_MODEL_REVISION,
        "sonata_model_revision": SONATA_MODEL_REVISION,
        "point_num": point_num,
        "prompt_num": prompt_num,
        "prompt_bs": prompt_bs,
        "face_count": int(len(dense)),
        "part_count": int(len(valid)),
        "part_face_counts": counts,
        "aabb": np.asarray(aabb).tolist(),
        "face_ids": dense.tolist(),
    }
    json_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    check_scene = trimesh.load(glb_path, force="scene")
    checks = {
        "glb_exists": glb_path.is_file() and glb_path.stat().st_size > 0,
        "json_exists": json_path.is_file() and json_path.stat().st_size > 0,
        "agentscape_evidence_exists": agentscape_path.is_file() and agentscape_path.stat().st_size > 0,
        "agentscape_source_node": agentscape_evidence["materialization"]["sourceNode"],
        "agentscape_primitive_count": len(agentscape_evidence["materialization"]["primitives"]),
        "agentscape_glb_sha256": agentscape_evidence["artifact"]["sha256"],
        "source_faces": int(len(mesh.faces)),
        "label_faces": int(len(dense)),
        "part_count": int(len(valid)),
        "glb_geometries": int(len(check_scene.geometry)),
    }
    if not (
        checks["glb_exists"]
        and checks["json_exists"]
        and checks["agentscape_evidence_exists"]
        and checks["agentscape_primitive_count"] > 0
        and checks["source_faces"] == checks["label_faces"]
        and checks["part_count"] > 0
        and checks["glb_geometries"] > 0
    ):
        raise RuntimeError(f"affordance validation failed: {checks}")

    report = {
        "source_job_id": source_job_id,
        "gpu": device_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "checks": checks,
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "result": "P3SAM_PART_SEGMENTATION_OK",
        "files": {
            "part_seg_glb": str(glb_path.relative_to(source_root)),
            "part_seg_json": str(json_path.relative_to(source_root)),
            "agentscape_part_segmentation": str(agentscape_path.relative_to(source_root)),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    artifacts.commit()
    print("P3SAM_PART_SEGMENTATION_OK", json.dumps(report), flush=True)
    return report

def _load_urdf_collision_mesh(urdf_path: Path, allowed_root: Path):
    """Load all mesh collisions in a URDF into the URDF link frame."""
    import xml.etree.ElementTree as ET

    import numpy as np
    import trimesh

    root = ET.parse(urdf_path).getroot()
    meshes = []
    allowed_root = allowed_root.resolve()
    for collision in root.findall(".//collision"):
        geometry = collision.find("geometry")
        mesh_node = geometry.find("mesh") if geometry is not None else None
        if mesh_node is None:
            continue
        filename = mesh_node.attrib.get("filename", "").strip()
        if not filename or "://" in filename or filename.startswith("package:"):
            raise RuntimeError(f"unsupported URDF collision mesh reference: {filename!r}")
        mesh_path = (urdf_path.parent / filename).resolve()
        if mesh_path != allowed_root and allowed_root not in mesh_path.parents:
            raise RuntimeError(f"URDF collision mesh escapes job root: {mesh_path}")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"URDF collision mesh missing: {mesh_path}")

        loaded = trimesh.load(mesh_path, force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            geometry_items = [g.copy() for g in loaded.geometry.values()]
            if not geometry_items:
                raise RuntimeError(f"collision mesh scene is empty: {mesh_path}")
            part = trimesh.util.concatenate(geometry_items)
        elif isinstance(loaded, trimesh.Trimesh):
            part = loaded.copy()
        else:
            raise RuntimeError(f"unsupported collision mesh type: {type(loaded).__name__}")

        scale_values = [float(x) for x in mesh_node.attrib.get("scale", "1 1 1").split()]
        if len(scale_values) != 3 or not np.isfinite(scale_values).all() or min(scale_values) <= 0:
            raise RuntimeError(f"invalid URDF mesh scale: {scale_values}")
        part.vertices = part.vertices * np.asarray(scale_values, dtype=np.float64)[None, :]

        origin = collision.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = [float(x) for x in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in origin.attrib.get("rpy", "0 0 0").split()]
        if len(xyz) != 3 or len(rpy) != 3 or not np.isfinite(xyz + rpy).all():
            raise RuntimeError(f"invalid URDF collision origin: xyz={xyz} rpy={rpy}")
        transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
        transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
        part.apply_transform(transform)
        meshes.append(part)

    if not meshes:
        raise RuntimeError(f"URDF has no mesh collision geometry: {urdf_path}")
    merged = trimesh.util.concatenate(meshes)
    if len(merged.vertices) == 0 or len(merged.faces) == 0:
        raise RuntimeError("merged URDF collision mesh is empty")
    return merged


@app.function(
    image=grasp_image,
    gpu="L40S",
    volumes={"/weights": weights, "/artifacts": artifacts},
    timeout=60 * 60,
    cpu=8.0,
    memory=32768,
    max_containers=1,
    scaledown_window=30,
)
def raw_grasp_job(
    source_job_id: str,
    num_points: int = 2024,
    num_grasps: int = 200,
    topk: int = 50,
    seed: int = 42,
) -> dict:
    """Run raw GraspGen generator+discriminator without GPT semantic filtering."""
    import os
    import sys
    import time

    import numpy as np
    import torch
    import trimesh
    from omegaconf import OmegaConf

    if not _valid_job_id(source_job_id):
        raise ValueError(f"invalid source job id: {source_job_id!r}")
    if not (512 <= num_points <= 20000):
        raise ValueError("num_points must be in 512..20000")
    if not (1 <= num_grasps <= 1000):
        raise ValueError("num_grasps must be in 1..1000")
    if not (1 <= topk <= min(num_grasps, 200)):
        raise ValueError("topk must be in 1..min(num_grasps, 200)")
    if not (0 <= seed <= 2**31 - 1):
        raise ValueError("seed must be a non-negative 32-bit integer")

    weights.reload()
    artifacts.reload()
    expected_files = {
        "config": (GRASPGEN_CONFIG, GRASPGEN_CONFIG_SHA256),
        "generator": (GRASPGEN_GEN, GRASPGEN_GEN_SHA256),
        "discriminator": (GRASPGEN_DIS, GRASPGEN_DIS_SHA256),
    }
    if not GRASPGEN_MANIFEST.is_file():
        raise RuntimeError("GraspGen weights are not preloaded; run preload_graspgen_weights")
    manifest = json.loads(GRASPGEN_MANIFEST.read_text())
    if manifest.get("revision") != GRASPGEN_MODELS_REVISION:
        raise RuntimeError("GraspGen model revision marker mismatch")
    for key, (path, expected_sha256) in expected_files.items():
        if not path.is_file():
            raise RuntimeError(f"missing GraspGen {key}: {path}")
        if manifest.get("files", {}).get(key, {}).get("sha256") != expected_sha256:
            raise RuntimeError(f"GraspGen {key} manifest hash mismatch")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"GraspGen {key} weight hash mismatch: {actual_sha256} != {expected_sha256}"
            )

    source_root = JOB_ROOT / source_job_id
    source_urdf = source_root / "result/sample_00.urdf"
    if not source_urdf.is_file():
        raise FileNotFoundError(f"source job has no URDF: {source_job_id}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on GraspGen worker")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise RuntimeError(f"expected NVIDIA L40S SM89, got {device_name} capability={capability}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    collision_mesh = _load_urdf_collision_mesh(source_urdf, source_root)
    sampled_points, _ = trimesh.sample.sample_surface(collision_mesh, num_points)
    if sampled_points.shape != (num_points, 3) or not np.isfinite(sampled_points).all():
        raise RuntimeError(f"invalid sampled collision point cloud: {sampled_points.shape}")

    graspgen_root = "/workspace/GraspGen"
    if graspgen_root not in sys.path:
        sys.path.insert(0, graspgen_root)
    os.chdir(graspgen_root)
    from grasp_gen.models.grasp_gen import GraspGen

    cfg = OmegaConf.load(str(GRASPGEN_CONFIG))
    cfg.discriminator.checkpoint_object_encoder_pretrained = str(GRASPGEN_GEN)
    model_started = time.perf_counter()
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    model.load_state_dict(str(GRASPGEN_GEN), str(GRASPGEN_DIS))
    model = model.cuda().eval()
    model.grasp_generator.num_grasps_per_object = num_grasps
    model_load_seconds = time.perf_counter() - model_started

    points = torch.as_tensor(sampled_points, device="cuda", dtype=torch.float32)
    center = points.mean(dim=0)
    centered_points = points - center[None, :]
    infer_started = time.perf_counter()
    with torch.inference_mode():
        output, _, _ = model.infer({"points": centered_points.unsqueeze(0)})
        grasps = output["grasps_pred"][0].clone()
        confidences = output["grasp_confidence"][0, :, 0].clone()
        grasps[:, :3, 3] += center[None, :]
        grasps[:, 3, 3] = 1.0
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - infer_started

    if grasps.ndim != 3 or tuple(grasps.shape[1:]) != (4, 4):
        raise RuntimeError(f"unexpected GraspGen pose shape: {tuple(grasps.shape)}")
    if confidences.ndim != 1 or len(confidences) != len(grasps):
        raise RuntimeError(
            f"unexpected GraspGen confidence shape: {tuple(confidences.shape)} poses={len(grasps)}"
        )
    if len(grasps) == 0:
        raise RuntimeError("GraspGen returned zero raw grasps")
    if not torch.isfinite(grasps).all() or not torch.isfinite(confidences).all():
        raise RuntimeError("GraspGen returned non-finite grasp data")

    order = torch.argsort(confidences, descending=True)[:topk]
    grasps = grasps[order]
    confidences = confidences[order]
    rotations = grasps[:, :3, :3]
    identity = torch.eye(3, device="cuda", dtype=rotations.dtype).expand(len(rotations), -1, -1)
    ortho_error = torch.max(torch.abs(rotations.transpose(1, 2) @ rotations - identity)).item()
    det = torch.linalg.det(rotations)
    max_det_error = torch.max(torch.abs(det - 1.0)).item()
    if ortho_error > 5e-3 or max_det_error > 5e-3:
        raise RuntimeError(
            f"invalid GraspGen rotations: ortho_error={ortho_error} det_error={max_det_error}"
        )

    grasps_cpu = grasps.detach().cpu().numpy()
    confidence_cpu = confidences.detach().cpu().numpy()
    source_urdf_sha256 = _sha256(source_urdf)
    collision_bounds = collision_mesh.bounds.tolist()
    payload = {
        "version": 1,
        "source_job_id": source_job_id,
        "backend": "GraspGen",
        "evidence_level": "raw",
        "gripper": "franka_panda",
        "source_frame": "urdf_link:sample_00",
        "graspgen_commit": GRASPGEN_COMMIT,
        "model_revision": GRASPGEN_MODELS_REVISION,
        "weights": {
            "generator_sha256": GRASPGEN_GEN_SHA256,
            "discriminator_sha256": GRASPGEN_DIS_SHA256,
            "config_sha256": GRASPGEN_CONFIG_SHA256,
        },
        "source_urdf_sha256": source_urdf_sha256,
        "seed": seed,
        "num_points": num_points,
        "num_grasps_requested": num_grasps,
        "topk": topk,
        "collision_bounds": collision_bounds,
        "grasps": [
            {
                "rank": rank,
                "score": float(confidence_cpu[rank]),
                "pose": grasps_cpu[rank].tolist(),
            }
            for rank in range(len(grasps_cpu))
        ],
    }

    output_dir = source_root / "affordance"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_grasps.franka.v1.json"
    report_path = output_dir / "graspgen_validation_report.json"
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    report = {
        "source_job_id": source_job_id,
        "gpu": device_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "grasp_count": int(len(grasps_cpu)),
        "score_min": float(confidence_cpu.min()),
        "score_max": float(confidence_cpu.max()),
        "rotation_orthogonality_max_error": float(ortho_error),
        "rotation_determinant_max_error": float(max_det_error),
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "result": "GRASPGEN_RAW_GRASPS_OK",
        "file": str(output_path.relative_to(source_root)),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    artifacts.commit()
    print("GRASPGEN_RAW_GRASPS_OK", json.dumps(report), flush=True)
    return report
