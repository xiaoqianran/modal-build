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

The release bundle contains the resident HTTP server, CLI, GGML shared libraries, and CUDA runtime
libraries. Model GGUF files are intentionally stored separately in Modal Volume.

## Pixal3D / L40S

Environment: `pixal3d-py310-cu124-torch260-sm89-v1`

- Python 3.10
- CUDA 12.4.1
- PyTorch 2.6.0 / torchvision 0.21.0 / Triton 3.2.0
- CUDA arch 8.9 (Ada / L40S)
- NATTEN 0.21.0
- Source-built `nvdiffrast`, `nvdiffrec_render`, `flex_gemm`, `cumesh`, `o_voxel`, `natten`

Runtime workers consume the Release zip with `uv`; they do not compile CUDA extensions.
