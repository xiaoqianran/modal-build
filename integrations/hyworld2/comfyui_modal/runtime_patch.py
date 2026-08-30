from pathlib import Path

ROOT = Path("/opt/ComfyUI/custom_nodes/ComfyUI_HYWorld2")

# V1: detect onnxruntime without importing it during ComfyUI startup.
p = ROOT / "nodes" / "world_mirror_v1.py"
s = p.read_text()
old = '''try:\n    import onnxruntime\n    SKYSEG_AVAILABLE = True\nexcept ImportError:\n    SKYSEG_AVAILABLE = False\n    print("⚠️ [VNCCS] onnxruntime not found. Sky segmentation will be disabled.")\n'''
new = '''import importlib.util as _importlib_util\nSKYSEG_AVAILABLE = _importlib_util.find_spec("onnxruntime") is not None\nif not SKYSEG_AVAILABLE:\n    print("⚠️ [VNCCS] onnxruntime not found. Sky segmentation will be disabled.")\n'''
if old not in s:
    raise RuntimeError("world_mirror_v1 onnxruntime block not found")
s = s.replace(old, new, 1)
s = s.replace(
    "skyseg_session = onnxruntime.InferenceSession(sky_model_path)",
    "skyseg_session = __import__('onnxruntime').InferenceSession(sky_model_path)",
)
p.write_text(s)

# V2: same lazy availability check.
p = ROOT / "nodes" / "world_mirror_v2.py"
s = p.read_text()
old = '''try:\n    import onnxruntime\n    ONNX_AVAILABLE = True\nexcept ImportError:\n    ONNX_AVAILABLE = False\n'''
new = '''import importlib.util as _importlib_util\nONNX_AVAILABLE = _importlib_util.find_spec("onnxruntime") is not None\n'''
if old not in s:
    raise RuntimeError("world_mirror_v2 onnxruntime block not found")
s = s.replace(old, new, 1)
s = s.replace(
    "sess   = onnxruntime.InferenceSession(sky_model_path)",
    "sess   = __import__('onnxruntime').InferenceSession(sky_model_path)",
)
p.write_text(s)

print("Applied HYWorld2 runtime lazy-import patches")

# WorldStereo: decord 0.6.0 has no supported Python 3.12 wheel. It is only
# used by get_last_video_frame(), so make it optional and fall back to OpenCV.
p = ROOT / "worldstereo" / "src" / "general_utils.py"
s = p.read_text()
if "from decord import VideoReader, cpu" in s:
    s = s.replace(
        "from decord import VideoReader, cpu",
        "try:\n    from decord import VideoReader, cpu\nexcept Exception:\n    VideoReader = None\n    cpu = None",
        1,
    )
old = '''def get_last_video_frame(video_path):
    """Use decord's random access to read the last frame directly by index."""
    vr = VideoReader(video_path, ctx=cpu(0))
    last_frame = vr[-1].asnumpy()  # Index the last frame directly with an internal seek
    return last_frame
'''
new = '''def get_last_video_frame(video_path):
    """Read the last video frame with decord when available, otherwise OpenCV."""
    if VideoReader is not None:
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr[-1].asnumpy()

    cap = cv2.VideoCapture(video_path)
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Could not read video frame count: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read final video frame: {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
'''
if old in s:
    s = s.replace(old, new, 1)
p.write_text(s)

print("Applied WorldStereo optional-decord patch")

# WorldStereo rotary embeddings can be produced on CPU while INT4 attention
# activations live on CUDA. Align all rotary cos/sin tensors to the activation
# device/dtype at the point of use.
p = ROOT / "worldstereo" / "models" / "attention.py"
s = p.read_text()
old_cos = "                cos = freqs_cos[..., 0::2]\n                sin = freqs_sin[..., 1::2]\n"
new_cos = "                cos = freqs_cos[..., 0::2].to(device=hidden_states.device, dtype=hidden_states.dtype)\n                sin = freqs_sin[..., 1::2].to(device=hidden_states.device, dtype=hidden_states.dtype)\n"
count = s.count(old_cos)
if count == 0:
    if new_cos not in s:
        raise RuntimeError("WorldStereo rotary block not found")
else:
    s = s.replace(old_cos, new_cos)
p.write_text(s)
print(f"Applied WorldStereo rotary device patch ({count} blocks)")

# diffusers 0.36.0 Wan attention has the same device mismatch: rotary cos/sin
# may stay on CPU while query/key are on CUDA. Patch the installed module so
# rotary tensors follow hidden_states device/dtype before arithmetic.
import importlib.util as _importlib_util
_diffusers_spec = _importlib_util.find_spec("diffusers")
if _diffusers_spec is None or not _diffusers_spec.submodule_search_locations:
    raise RuntimeError("diffusers package not found for Wan rotary patch")
_diffusers_root = Path(next(iter(_diffusers_spec.submodule_search_locations)))
_p = _diffusers_root / "models" / "transformers" / "transformer_wan.py"
_s = _p.read_text()
_old = "                cos = freqs_cos[..., 0::2]\n                sin = freqs_sin[..., 1::2]\n"
_new = "                cos = freqs_cos[..., 0::2].to(device=hidden_states.device, dtype=hidden_states.dtype)\n                sin = freqs_sin[..., 1::2].to(device=hidden_states.device, dtype=hidden_states.dtype)\n"
_count = _s.count(_old)
if _count == 0:
    if _new not in _s:
        raise RuntimeError("diffusers Wan rotary block not found")
else:
    _s = _s.replace(_old, _new)
    _p.write_text(_s)
print(f"Applied diffusers Wan rotary device patch ({_count} blocks)")
