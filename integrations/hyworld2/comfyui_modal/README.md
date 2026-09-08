# ComfyUI_HYWorld2 on Modal

This runtime installs `xiaoqianran/ComfyUI_HYWorld2`, builds CUDA extensions for the selected GPU,
and persists models, inputs, outputs, and caches in `comfyui-hyworld2-data`.

## Cost policy

- Validation spend recorded on 2026-08-31 (conservative app wall-time estimate, GPU only):

  | Stage | Device and duration | Estimated GPU cost |
  | --- | --- | ---: |
  | Local checks and CPU config probe | No GPU | < $0.001 |
  | Native builds, 9/9 node imports, and ComfyUI startup | L4, ~1,691 s | ~$0.38 |
  | RTX probe and `sm_120` builds | RTX PRO 6000, ~498 s | ~$0.42 |
  | 252px real PLY smoke | RTX PRO 6000, ~69 s | ~$0.06 |
  | 518px final Blender PLY | RTX PRO 6000, ~51 s | ~$0.05 |
  | **Total** |  | **~$0.90 (conservative ceiling $0.92)** |

  Rates: L4 `$0.000222/s`, RTX PRO 6000 `$0.000842/s`. This is not an invoice and excludes CPU,
  memory, Volume, and network charges. All task-owned apps were stopped; ongoing GPU cost is `$0/h`.
- Default GPU: Modal `RTX-PRO-6000` (NVIDIA RTX PRO 6000 Blackwell, `sm_120`).
- Allowed test GPUs: `L4` and `L40S`; H100 is intentionally rejected.
- Every GPU function has `min_containers=0`, one-container concurrency, a finite timeout, and a
  short scaledown window. A deployed app therefore scales to zero when idle.
- `generate_world` runs a 252px PLY smoke first, then reuses the warm bf16 model for a 518px final
  Gaussian PLY.

## Checks and generation

```bash
uv sync --locked
uv run modal run modal_control.py::config_probe
uv run modal run modal_control.py::gpu_probe
MODAL_GPU=L4 uv run modal run modal_app.py::import_probe
uv run modal volume put comfyui-hyworld2-data input.png /input/input.png --force
uv run modal run modal_app.py::generate_world --image-name input.png
```

Download the generated Blender-compatible PLY:

```bash
uv run modal volume ls comfyui-hyworld2-data /output
uv run modal volume get comfyui-hyworld2-data \
  output/hyworld2_blender_world_00001_gaussians.ply ./hyworld2_blender_world.ply
```

Deploy the optional UI only when interactive use is needed:

```bash
uv run modal deploy modal_app.py --stream-logs
```

The deployment can remain registered while GPU containers scale to zero. To remove the deployment
itself, run `uv run modal app stop comfyui-hyworld2`; redeployment is required afterward.
