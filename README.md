# modal-build

Public, reproducible prebuilt wheel artifacts for Modal GPU workers.

Large binary artifacts are stored as **GitHub Release assets**, not committed into Git history.
Each release is keyed by Python/CUDA/PyTorch/CUDA-architecture compatibility and ships:

- `*.wheels.zip` — prebuilt wheels
- `*.manifest.json` — exact environment and per-wheel SHA256
- `*.sha256` — archive checksum

## TRELLIS2 / L40S

Environment: `hermit-trellis2-plus-plus-py311-cu124-torch260-sm89-v1`

- Python 3.11
- Ubuntu 22.04
- CUDA 12.4.1
- PyTorch 2.6.0
- torchvision 0.21.0
- CUDA arch 8.9 (Ada / L40S)

Build and publish from Modal:

```bash
modal run -m modal_build.hermit_trellis2_plus_plus::build_and_release
```

The function is hard-limited to one L40S container and publishes using the Modal Secret
`modal-build-github`. Runtime projects should install the released wheels with `uv`, avoiding
repeated CUDA compilation.

## EmbodiedGen v2.0.0 / L40S

Environment: `embodiedgen-v2.0.0-py310-cu126-torch280-sm89-v1`

- Python 3.10 / Ubuntu 22.04
- CUDA 12.6.3
- PyTorch 2.8.0 / torchvision 0.23.0
- CUDA arch 8.9 (Ada / L40S)
- PyTorch3D pinned to `75ebeeaea0908c5527e7b1e305fbc7681382db47`
- nvdiffrast pinned to `729261d`
- gsplat 1.5.3 O3 SM89 torch-extension cache
- SAM3D model weights are intentionally kept outside release assets

The build itself is **CPU-only**: the CUDA devel image provides `nvcc`, and
`TORCH_CUDA_ARCH_LIST=8.9` targets L40S without renting a GPU.  A real L40S is used only for the
end-to-end validation run.  The validated headless runtime patches are under
`patches/embodiedgen-v2.0.0/`, with the Modal runner under `runtime/embodiedgen_v2_l40s.py`.

Build and publish:

```bash
modal run -m modal_build.embodiedgen::build_and_release
```

The release contains both normal wheels and the precompiled torch-extension cache.  Extract the
cache archive into `~/.cache/torch_extensions/` on an identical Python/Torch/CUDA/SM89 runtime to
avoid rebuilding gsplat/nvdiffrast on the paid GPU worker.

Validation completed with `VALIDATION_OK`: 95,004 PLY Gaussians, 516,271 OBJ vertices,
891,420 OBJ faces, one valid GLB geometry, resolvable URDF mesh references, and a valid MP4.

## Policy

Do not publish model weights, gated Hugging Face assets, secrets, or artifacts without clear
redistribution permission. This repository is for build tooling and redistributable wheels.

## trellis.cpp / L40S

Environment: `trellis.cpp-pynone-cu129-torchnone-sm89-v2`

- Native C++17 / GGML runtime (no Python or PyTorch at inference time)
- Ubuntu 22.04
- CUDA 12.9.1
- CUDA arch 8.9 (Ada / L40S)
- Source pinned to `pwilkin/trellis.cpp@16f3109e82f3922033bfa62b83c42899678b7b6f`

The release bundle contains the resident HTTP server, CLI, and GGML shared libraries. CUDA runtime
libraries are supplied by the pinned NVIDIA runtime image, while `libcuda.so.1` is provided by the
NVIDIA driver / Modal GPU host at container startup. Model GGUF files are intentionally stored
separately in Modal Volume.

## Pixal3D / L40S

Environment: `pixal3d-py310-cu124-torch260-sm89-v1`

- Python 3.10
- CUDA 12.4.1
- PyTorch 2.6.0 / torchvision 0.21.0 / Triton 3.2.0
- CUDA arch 8.9 (Ada / L40S)
- NATTEN 0.21.0
- Source-built `nvdiffrast`, `nvdiffrec_render`, `flex_gemm`, `cumesh`, `o_voxel`, `natten`

Runtime workers consume the Release zip with `uv`; they do not compile CUDA extensions.
