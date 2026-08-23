"""Auto-loaded in the Modal release-consumer image.

Forces CUDA-extension consumers to use modal-build release .so files.
There is intentionally no JIT fallback.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys


def _load_extension(name: str, path: str):
    p = pathlib.Path(path)
    if not p.exists():
        raise ImportError(f"precompiled extension missing: {p}")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create extension loader: {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patch_nvdiffrast():
    try:
        import nvdiffrast.torch.ops as ops
    except Exception as e:
        print(f"[release-consumer] nvdiffrast import skipped: {e}", file=sys.stderr)
        return

    def release_get_plugin(gl=False):
        if ops._cached_plugin.get(gl, None) is not None:
            return ops._cached_plugin[gl]
        if gl:
            raise RuntimeError("nvdiffrast GL plugin not shipped in release-consumer")
        m = _load_extension(
            "nvdiffrast_plugin",
            "/root/.cache/torch_extensions/py310_cu126/nvdiffrast_plugin/nvdiffrast_plugin.so",
        )
        ops._cached_plugin[False] = m
        return m

    ops._get_plugin = release_get_plugin


def _patch_gsplat():
    try:
        import gsplat.cuda._backend as backend
    except Exception as e:
        print(f"[release-consumer] gsplat backend import skipped: {e}", file=sys.stderr)
        return
    class _LazyGsplatC:
        _module = None
        def _get(self):
            if self._module is None:
                self._module = _load_extension(
                    "gsplat_cuda",
                    "/root/.cache/torch_extensions/py310_cu126/gsplat_cuda/gsplat_cuda.so",
                )
            return self._module
        def __getattr__(self, name):
            return getattr(self._get(), name)

    backend._C = _LazyGsplatC()


# At Python startup CUDA may not yet be initialized, but loading the extension modules
# is safe. Actual kernels/contexts are created only when the application requests them.
_patch_nvdiffrast()
_patch_gsplat()
