import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/tmp/pytorch3d_build")
WHEEL_DIR = Path("/tmp/pytorch3d_wheels")


def run(*args, cwd=None, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.check_call([str(x) for x in args], cwd=str(cwd) if cwd else None, env=env)


shutil.rmtree(SRC, ignore_errors=True)
shutil.rmtree(WHEEL_DIR, ignore_errors=True)
WHEEL_DIR.mkdir(parents=True, exist_ok=True)
run("git", "clone", "--depth=1", "--branch", "stable", "https://github.com/facebookresearch/pytorch3d.git", SRC)

# Stable PyTorch3D always globs Pulsar C++/CUDA sources. HYWorld2 does not use
# Pulsar. Build a clean no-Pulsar extension to avoid CUDA 13 link failures.
setup = SRC / "setup.py"
text = setup.read_text()
needle = "    extension = CppExtension\n"
replacement = (
    needle
    + "    sources = [s for s in sources if f'{os.sep}pulsar{os.sep}' not in s]\n"
    + "    source_cuda = [s for s in source_cuda if f'{os.sep}pulsar{os.sep}' not in s]\n"
)
if needle not in text:
    raise RuntimeError("PyTorch3D setup.py layout changed: extension marker missing")
setup.write_text(text.replace(needle, replacement, 1))

ext = SRC / "pytorch3d" / "csrc" / "ext.cpp"
ext_text = ext.read_text()
ext_text = re.sub(
    r'#if !defined\(USE_ROCM\)\n#include "\./pulsar/global\.h".*?\n#endif\n',
    "",
    ext_text,
    flags=re.DOTALL,
)
ext_text = re.sub(
    r'#if !defined\(USE_ROCM\)\n#include "\./pulsar/pytorch/renderer\.h"\n#include "\./pulsar/pytorch/tensor_util\.h"\n#endif\n',
    "",
    ext_text,
    flags=re.DOTALL,
)
start = ext_text.find("  // Pulsar.")
if start != -1:
    end = ext_text.find("\n#endif\n}", start)
    if end == -1:
        raise RuntimeError("Could not locate PyTorch3D Pulsar binding block end")
    ext_text = ext_text[:start] + "  // Pulsar disabled for HYWorld2 no-Pulsar build.\n" + ext_text[end + len("\n#endif"):]
ext.write_text(ext_text)

points_init = SRC / "pytorch3d" / "renderer" / "points" / "__init__.py"
points_text = points_init.read_text()
points_text = re.sub(
    r"\r?\n# Pulsar not enabled on amd\.\r?\nif not torch\.version\.hip:\r?\n    from \.pulsar\.unified import PulsarPointsRenderer\r?\n",
    "\n# Pulsar disabled for HYWorld2 no-Pulsar build.\n",
    points_text,
)
points_init.write_text(points_text)

renderer_init = SRC / "pytorch3d" / "renderer" / "__init__.py"
renderer_text = renderer_init.read_text()
renderer_text = re.sub(
    r"\r?\n# Pulsar is not enabled on amd\.\r?\nif not torch\.version\.hip:\r?\n    from \.points import PulsarPointsRenderer\r?\n",
    "\n# Pulsar disabled for HYWorld2 no-Pulsar build.\n",
    renderer_text,
)
renderer_init.write_text(renderer_text)

env = os.environ.copy()
env.setdefault("FORCE_CUDA", "1")
env.setdefault("MAX_JOBS", "8")
env.setdefault("PYTORCH3D_NO_NINJA", "0")
env.setdefault("CUDA_HOME", "/usr/local/cuda")
env.setdefault("CUB_HOME", "/usr/local/cuda/include")

run(
    sys.executable,
    "-m",
    "pip",
    "wheel",
    "--no-build-isolation",
    "--no-deps",
    "-v",
    "-w",
    WHEEL_DIR,
    SRC,
    cwd=SRC,
    env=env,
)

wheels = sorted(WHEEL_DIR.glob("pytorch3d-*.whl"))
if not wheels:
    raise RuntimeError("PyTorch3D wheel was not produced")
wheel = wheels[-1]
run(sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel)

import torch
import pytorch3d
from pytorch3d import _C
from pytorch3d.renderer.cameras import look_at_rotation

assert callable(look_at_rotation)
assert hasattr(_C, "rasterize_meshes")
print(
    "PyTorch3D no-Pulsar import OK",
    pytorch3d.__version__,
    torch.__version__,
    torch.version.cuda,
    flush=True,
)
