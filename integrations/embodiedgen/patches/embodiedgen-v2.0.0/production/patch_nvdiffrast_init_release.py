from pathlib import Path
import nvdiffrast

p = Path(nvdiffrast.__file__).parent / "torch" / "__init__.py"
s = p.read_text()
marker = "# modal-build release-only loader"
if marker not in s:
    s += r'''

# modal-build release-only loader
# Every `import nvdiffrast.torch` executes this override after .ops is imported.
# There is deliberately no torch cpp_extension/JIT fallback.
from . import ops as _modal_release_ops
import importlib.util as _modal_importlib_util
import pathlib as _modal_pathlib
import sys as _modal_sys


def _modal_release_get_plugin(gl=False):
    if _modal_release_ops._cached_plugin.get(gl, None) is not None:
        return _modal_release_ops._cached_plugin[gl]
    if gl:
        raise RuntimeError("nvdiffrast GL plugin is not shipped in this release-consumer")
    _so = _modal_pathlib.Path(
        "/root/.cache/torch_extensions/py310_cu126/nvdiffrast_plugin/nvdiffrast_plugin.so"
    )
    if not _so.exists():
        raise ImportError(f"precompiled nvdiffrast plugin missing: {_so}")
    _spec = _modal_importlib_util.spec_from_file_location("nvdiffrast_plugin", _so)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot create loader for {_so}")
    _module = _modal_importlib_util.module_from_spec(_spec)
    _modal_sys.modules["nvdiffrast_plugin"] = _module
    _spec.loader.exec_module(_module)
    _modal_release_ops._cached_plugin[False] = _module
    return _module

_modal_release_ops._get_plugin = _modal_release_get_plugin
'''
    p.write_text(s)
print("patched nvdiffrast torch __init__ release loader:", p)
