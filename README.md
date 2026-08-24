# modal-build

Reproducible CUDA/PyTorch build artifacts and production reference runtimes for Modal 3D workers.

Large binary artifacts are stored as **GitHub Release assets**, not committed into Git history.
Each release is keyed by Python/CUDA/PyTorch/CUDA-architecture compatibility and ships:

- `*.wheels.zip` — prebuilt wheels
- `*.manifest.json` — exact environment and per-wheel SHA256
- `*.sha256` — archive checksum


## Repository layers

The repository separates three different lifecycle stages. They are related, but they do not call
each other directly at runtime:

```text
modal_build/                 offline artifact builders
    embodiedgen.py           builds/publishes pinned wheels and CUDA extension archives

patches/                     source/runtime compatibility overlays
    embodiedgen-v2.0.0/
        production/          only files consumed by the current production runtime
        legacy/              historical experiments kept for reproduction only

runtime/                     deployable Modal applications
    embodiedgen_v2_l40s.py   current EmbodiedGen Image→3D / Text→3D / Retexture runtime/API
    legacy/                  historical runtime variants
```

For EmbodiedGen, the lifecycle is:

```text
modal_build/embodiedgen.py
        │
        └── build binary artifacts (CPU host + nvcc, no paid GPU)
                │
                ▼
        GitHub Release assets
                │
────────────────┼────────────────────────────────────
                │
                ▼
runtime/embodiedgen_v2_l40s.py
        │
        ├── clone exact upstream EmbodiedGen commit
        ├── verify/download the prebuilt Release artifacts
        ├── apply patches/embodiedgen-v2.0.0/production/*
        └── deploy the Modal production workers/API
```

`modal_build/embodiedgen.py` intentionally does **not** import or execute files from `patches/`:
the builder produces reusable binary artifacts, while the production runtime consumes those
artifacts and applies runtime compatibility patches in a later image-build stage.

The milestone tag `embodiedgen-v2.0.0-image-to-3d-modal-v1` marks completion of the validated
Image→3D production pipeline. It is a Git tag only, not a GitHub Release.

The production EmbodiedGen API now also exposes `POST /text-jobs`: a pinned public Kolors
Text→Image L40S stage generates the conditioning image and then reuses the exact validated
Image→3D pipeline. A full authenticated production Text→3D E2E has passed with GLB/video/validation
HTTP downloads and zero traceback/OOM/runtime-warning matches. See `docs/embodiedgen.md` for the
pinned model revision, timings and measured cold validation cost.

It also exposes `POST /jobs/{source_job_id}/retexture` for prompt-driven appearance edits of an
existing successful asset. Retexture reuses the pinned Kolors snapshot plus only ~393 MiB of pinned
RoboAssetGen ControlNet/RealESRGAN weights, preserves geometry, and has passed an authenticated
production E2E with OBJ/MTL/texture/GLB/video downloads and zero GPT-init/UV/traceback/OOM warnings.

## TRELLIS2 / L40S

Environment: `hermit-trellis2-plus-plus-py311-cu124-torch260-sm89-v2`

- Python 3.11
- Ubuntu 22.04
- CUDA 12.4.1
- PyTorch 2.6.0
- torchvision 0.21.0
- CUDA arch 8.9 (Ada / L40S)

Build and publish from Modal:

```bash
modal run -m modal_build.hermit_trellis2_plus_plus_v2::build
```

The v2 builder is hard-limited to one L40S container and writes a SHA256-manifested wheel bundle to
the `modal-build-artifacts` Volume. The published Release with the same tag contains the validated
`flash-attn`, `nvdiffrast`, `nvdiffrec`, `CuMesh`, `FlexGEMM`, and `o-voxel` wheels. Runtime
projects install the released wheels with `uv`, avoiding repeated CUDA compilation.



## FastSAM3D PyTorch3D / L40S

Environment: `fastsam3d-pytorch3d-py311-cu121-torch251-sm89-v1`

- Python 3.11 / CUDA 12.1.1 / PyTorch 2.5.1 / torchvision 0.20.1
- CUDA arch 8.9 (Ada / L40S)
- PyTorch3D pinned to `facebookresearch/pytorch3d@75ebeeaea0908c5527e7b1e305fbc7681382db47`
- SHA256-manifested wheel bundle, validated by importing the renderer on L40S

Build it with:

```bash
modal run -m modal_build.fastsam3d_pytorch3d::build
```

The production FastSAM3D worker installs this released wheel bundle instead of compiling PyTorch3D
during every image build, cutting repeated CUDA build work out of normal deployments.

## Hunyuan3D 2.1 Paint / L40S

Environment: `hunyuan3d-2.1-paint-py311-cu124-torch251-sm89-v2`

- Python 3.11 / CUDA 12.4.1 / PyTorch 2.5.1 / torchvision 0.20.1
- CUDA arch 8.9 (Ada / L40S)
- Source pinned to `Archerkattri/hunyuan2.1-plus-plus@9efd760fbec8ab490e68b330225ea1fab10de7fd`
- Bundle contains the `custom_rasterizer` CUDA wheel plus the native `mesh_inpaint_processor` extension
- Every binary and the release archive are SHA256-manifested

Build the exact runtime-native bundle:

```bash
modal run -m modal_build.hunyuan3d21_paint_v2::build
```

The resulting bundle is stored in `modal-build-artifacts` and mirrored to the GitHub Release with
the same tag. The production `modal-3D` Hunyuan worker consumes this bundle directly, so neither
CUDA rasterization nor mesh inpainting is compiled during a cold image build.

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
end-to-end validation run.  The validated production runtime patches are under
`patches/embodiedgen-v2.0.0/production/`, with the Modal runner under
`runtime/embodiedgen_v2_l40s.py`. Historical patch/runtime variants are isolated under `legacy/`.

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
