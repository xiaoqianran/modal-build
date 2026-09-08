# HY-World 2.0 build artifacts

This integration prepares reproducible artifacts for the `modal-provider/modal-world` HYWorld2
backend. Native bundles are built per GPU architecture while keeping one Python/CUDA/PyTorch ABI:

- Blackwell: Python 3.11 + CUDA 12.8 + PyTorch 2.7.1 on `RTX-PRO-6000` (`sm_120`).
- Hopper: Python 3.11 + CUDA 12.8 + PyTorch 2.7.1 on `H100` (`sm_90`).

## Artifact split

HY-World full worldgen has several source-built dependencies. The Tencent-provided custom
`gsplat_maskgaussian` and Recast binding are kept in a **private Modal Volume bundle** because the
HY-WORLD 2.0 Community License has Territory restrictions. We do not auto-publish those binaries to
a globally accessible GitHub Release.

Permissively licensed third-party dependencies are separate and can be published with exact source
revision, ABI, license files and SHA256 manifest.

| Bundle | Contents | Distribution |
| --- | --- | --- |
| `hyworld2-hy-native-...` | custom gsplat + HY navmesh binding | Modal Volume + private GitHub Release backup |
| `hyworld2-oss-native-...` | PyTorch3D + fused-ssim + SPZ | Volume + GitHub Release |
| `hyworld2-oss-source-...` | MoGe + pinned nerfview | Modal Volume only (nerfview pinned revision lacks LICENSE file) |
| `hyworld2-flash-attn-...` | FlashAttention, architecture-specific | Volume + GitHub Release after smoke |

FlashAttention is optional: upstream HYWorld2 falls back to PyTorch SDPA when neither FA3 nor FA2
is available. The base runtime therefore must not depend on FlashAttention succeeding.

## Build and smoke

```bash
modal run integrations/hyworld2/build/hyworld2_hy_native_sm120.py::build
modal run integrations/hyworld2/build/hyworld2_oss_native_sm120.py::build
modal run integrations/hyworld2/build/hyworld2_oss_source_wheels.py::build
modal run integrations/hyworld2/build/hyworld2_flash_attn_sm120.py::build

# H100 / Hopper
modal run integrations/hyworld2/build/hyworld2_hy_native_sm90.py::build
modal run integrations/hyworld2/build/hyworld2_oss_native_sm90.py::build
modal run integrations/hyworld2/build/hyworld2_flash_attn_sm90.py::build
```

GPU builders fail closed on the target compute capability: `(12, 0)` for Blackwell and `(9, 0)` for
Hopper. The restricted gsplat builder executes a real CUDA rasterization using the HY-only `distloss`
and `gauss_masks` arguments. The OSS builder runs a PyTorch3D CUDA KNN and fused-ssim CUDA smoke.

Each build writes to `modal-build-artifacts`:

- `<tag>.wheels.zip`
- `<tag>.manifest.json`
- `<tag>.wheels.zip.sha256`

License/notice files are carried inside the zip.

## Publishing

```bash
integrations/hyworld2/scripts/publish_from_volume.sh <tag>
```

The publisher reads the manifest and fails closed when `public_release` is false. GitHub Actions can
orchestrate the same process through `HYWorld2 Build Artifact` after `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` repository secrets are configured.

### Private restricted-artifact backup

Restricted HY-derived bundles are mirrored to the **private** repository
`xiaoqianran/modal-build-private`. The backup tool refuses to upload them unless GitHub reports that
the destination repository visibility is `PRIVATE`. It verifies the manifest ABI, source revision,
SHA256 sidecar, ZIP integrity, required HY-WORLD/NOTICE payload and the recorded CUDA smoke before
publishing or restoring.

```bash
# Back up both sm90 + sm120 bundles already present in modal-build-artifacts.
uv run --frozen python -m integrations.hyworld2.private_artifacts backup --all

# New Modal workspace: restore both bundles without compiling CUDA code.
uv run --frozen python -m integrations.hyworld2.private_artifacts restore --all

# Deployment preflight: use Volume when valid, otherwise restore from private GitHub.
# This NEVER launches a GPU compiler by default.
uv run --frozen python -m integrations.hyworld2.private_artifacts ensure --all

# Expensive source-build fallback is opt-in only.
uv run --frozen python -m integrations.hyworld2.private_artifacts ensure --all --compile-if-missing
```

`gh` authentication is intentionally kept on the deployment/CI machine; no GitHub PAT is embedded in
Modal GPU containers. Override the destinations with `HYWORLD2_PRIVATE_ARTIFACT_REPO` and
`HYWORLD2_ARTIFACT_VOLUME` when needed.

## Runtime consumption

Public bundles can be installed with `scripts/install_release.sh`. Restricted bundles are consumed
from `modal-build-artifacts`; the private Release is a durable backup used to repopulate a new Modal
workspace without compiling native CUDA code again. Model checkpoints are stored separately in Modal
Volume or a pinned Hugging Face snapshot; they are never GitHub Release assets.

## Validated ComfyUI runtime

`comfyui_modal` also contains the cost-controlled Python 3.12 + CUDA 13.0 + PyTorch 2.9.1 runtime
validated on Modal `RTX-PRO-6000`. Its ABI is recorded in
`env/hyworld2-comfyui-py312-cu130-torch291-sm120-v1.json`. Do not install the Python 3.11 / CUDA
12.8 / PyTorch 2.7.1 release wheels into that runtime; native wheel ABIs must match exactly.
