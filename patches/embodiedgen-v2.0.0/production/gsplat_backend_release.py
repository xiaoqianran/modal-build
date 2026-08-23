"""Release-only gsplat CUDA backend for the no-nvcc consumer image."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_SO = pathlib.Path(
    "/root/.cache/torch_extensions/py310_cu126/gsplat_cuda/gsplat_cuda.so"
)
if not _SO.exists():
    raise ImportError(f"precompiled gsplat extension missing: {_SO}")

_spec = importlib.util.spec_from_file_location("gsplat_cuda", _SO)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot create gsplat extension loader for {_SO}")
_C = importlib.util.module_from_spec(_spec)
sys.modules["gsplat_cuda"] = _C
_spec.loader.exec_module(_C)

__all__ = ["_C"]
