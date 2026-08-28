# modal-build

Reproducible CUDA/PyTorch build artifacts and production reference runtimes for Modal 3D workers.

Large binary artifacts are stored as **GitHub Release assets**, not committed into Git history.
Each release is keyed by Python/CUDA/PyTorch/CUDA-architecture compatibility and ships:

- `*.wheels.zip` — prebuilt wheels
- `*.manifest.json` — exact environment and per-wheel SHA256
- `*.sha256` — archive checksum


## Repository layout

The repository is organized around model integrations. Build tooling, environment manifests,
runtime code, patches, and tests for one integration stay together instead of being split across
repository-wide lifecycle folders.

```text
integrations/
    embodiedgen/
        build/                  release/build helpers
        runtime/                deployed workers + local/VPS direct control plane
        env/                    pinned environment manifest
        patches/                production and historical compatibility patches
        tests/                  EmbodiedGen-specific tests
        README.md               integration documentation

    fastsam3d/
        build/
        env/

    hermit_trellis2/
        build/
            hermit_trellis2_plus_plus.py
            hermit_trellis2_plus_plus_v2.py
        env/
        scripts/

    hunyuan3d/
    pixal3d/
    trellis_cpp/
    birefnet/

shared/                         reserved for genuinely cross-integration code
```

The Hermit/TRELLIS2 builder filenames and version relationship are intentionally preserved as-is;
this reorganization only changes their location.

For EmbodiedGen, the lifecycle is now colocated under one integration:

```text
integrations/embodiedgen/build/embodiedgen.py
        │
        └── build binary artifacts (CPU host + nvcc, no paid GPU)
                │
                ▼
        GitHub Release assets
                │
────────────────┼────────────────────────────────────
                │
                ▼
integrations/embodiedgen/runtime/embodiedgen_v2_l40s.py
        │
        ├── clone exact upstream EmbodiedGen commit
        ├── verify/download the prebuilt Release artifacts
        ├── apply integrations/embodiedgen/patches/embodiedgen-v2.0.0/production/*
        └── deploy only the Modal production compute workers
```

The builder intentionally does **not** import or execute the runtime patches: build artifacts and
runtime compatibility remain separate lifecycle stages, but they now live under the same integration.

Production request orchestration runs in the local/VPS process via
`integrations/embodiedgen/runtime/embodiedgen_direct.py`. There is no Modal ASGI gateway or
per-request orchestration Function in the request hot path.

See `integrations/embodiedgen/README.md` for the full production and benchmark history.

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
modal run integrations/hermit_trellis2/build/hermit_trellis2_plus_plus_v2.py::build
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
modal run integrations/fastsam3d/build/fastsam3d_pytorch3d.py::build
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
modal run integrations/hunyuan3d/build/hunyuan3d21_paint_v2.py::build
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
`integrations/embodiedgen/patches/embodiedgen-v2.0.0/production/`, with the Modal runner under
`integrations/embodiedgen/runtime/embodiedgen_v2_l40s.py`. Historical patch/runtime variants are isolated under `legacy/`.

Build and publish:

```bash
modal run integrations/embodiedgen/build/embodiedgen.py::build_and_release
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
