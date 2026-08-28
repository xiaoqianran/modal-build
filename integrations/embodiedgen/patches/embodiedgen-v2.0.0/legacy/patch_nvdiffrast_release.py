from pathlib import Path
import nvdiffrast

p = Path(nvdiffrast.__file__).parent / "torch" / "ops.py"
s = p.read_text()
a = s.index("def _get_plugin(gl=False):")
b = s.index("#----------------------------------------------------------------------------\n# Log level.", a)
fn = '''def _get_plugin(gl=False):
    """Release-only plugin loader. This runtime must never JIT compile."""
    if _cached_plugin.get(gl, None) is not None:
        return _cached_plugin[gl]
    if gl:
        raise RuntimeError("nvdiffrast GL plugin is not shipped in the EmbodiedGen consumer release")
    import importlib.util
    import sys
    so = "/root/.cache/torch_extensions/py310_cu126/nvdiffrast_plugin/nvdiffrast_plugin.so"
    if not os.path.exists(so):
        raise ImportError(f"missing precompiled nvdiffrast plugin: {so}")
    spec = importlib.util.spec_from_file_location("nvdiffrast_plugin", so)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create loader for {so}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nvdiffrast_plugin"] = module
    spec.loader.exec_module(module)
    _cached_plugin[False] = module
    return module

'''
p.write_text(s[:a] + fn + s[b:])
check = p.read_text()[a:b]
assert "cpp_extension.load" not in check
assert "torch.utils.cpp_extension.load" not in check
print("nvdiffrast release-only loader installed:", p)
