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

EmbodiedGen has moved to the dedicated `xiaoqianran/modal-embodiedgen` fork. Its complete Modal
build/runtime/control-plane stack now lives under that repository's `modal/` directory. This
repository no longer owns EmbodiedGen production code.

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

## HY-World 2.0 / RTX PRO 6000 Blackwell

HYWorld2 build recipes live under `integrations/hyworld2/`. ABI-sensitive CUDA/C++ dependencies are
prebuilt for Python 3.11 / CUDA 12.8 / PyTorch 2.7.1 / sm_120. HY-WORLD-derived native binaries
(custom gsplat and the navmesh binding) are cached in Modal Volume only because the HY-WORLD
Community License has Territory restrictions; permissively licensed third-party wheels are eligible
for SHA256-manifested GitHub Releases. Model weights are never Release assets.

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
