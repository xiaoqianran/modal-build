# EmbodiedGen v2.0.0 on Modal L40S

This target is a **release-consumer runtime** for `HorizonRobotics/EmbodiedGen@v2.0.0`.
The runtime does not compile CUDA code. It uses `nvidia/cuda:12.6.3-runtime-ubuntu22.04`,
and the image asserts that `nvcc` is absent.

## Validated environment

| Component | Value |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| GPU | NVIDIA L40S |
| Compute capability | SM89 / 8.9 |
| CUDA user-space runtime | 12.6.3 |
| PyTorch | 2.8.0+cu126 |
| torchvision | 0.23.0+cu126 |
| xformers | 0.0.32.post2 |
| Kaolin | 0.18.0 prebuilt wheel |
| gsplat | 1.5.3, precompiled O3 SM89 extension |
| PyTorch3D | 0.7.8 wheel, commit `75ebeeaea0908c5527e7b1e305fbc7681382db47` |
| nvdiffrast | 0.3.3 wheel, commit `729261d`, precompiled SM89 CUDA plugin |
| EmbodiedGen | v2.0.0 |

Release: `embodiedgen-v2.0.0-py310-cu126-torch280-sm89-v1`.

Release assets:

- `*.wheels.zip`: PyTorch3D and nvdiffrast wheels.
- `*.torch-extensions.zip`: `gsplat_cuda.so` and `nvdiffrast_plugin.so` plus their build cache layout.
- `*.manifest.json`: exact versions, commits, hashes and validation metadata.
- `*.sha256`: asset checksums.

SAM3D weights, DINOv2, U2Net and generated outputs are intentionally **not** stored in the
GitHub Release. They live on the persistent Modal Volume `embodiedgen-v2-data`.

## Why there are two runtime files

`runtime/embodiedgen_v2_l40s.py` is the production-style consumer. It downloads the release
artifacts and cannot compile CUDA because the image has no `nvcc`.

`runtime/embodiedgen_v2_l40s_fullbuild.py` is retained only as the fully reproducible historical
builder/validation path. It contains CUDA build steps and should not be used for normal inference.

## Headless patches

The validated patches are under `patches/embodiedgen-v2.0.0/`. They address headless/Modal issues
without changing the SAM3D model weights:

- bypass Hugging Face ZeroGPU decorators when a real Modal GPU is already allocated;
- isolate SAM3D generation from CUDA-heavy post-processing;
- use the verified gsplat path for headless rendering;
- lazy-load expensive optional backprojection modules;
- keep TRELLIS optional for a SAM3D-only runtime;
- use a cost-safe validation profile for texture baking.

The original end-to-end validation produced `VALIDATION_OK` with 95,004 PLY Gaussians,
516,271 OBJ vertices, 891,420 OBJ faces, one valid GLB geometry, valid URDF mesh references,
and a valid MP4.

## Build / cache behavior

Normal inference should never run `nvcc`. A cache miss in gsplat/nvdiffrast is treated as a hard
error rather than silently compiling on an L40S. To rebuild CUDA artifacts, use
`modal_build/embodiedgen.py` and publish a new release tag.

## Model/cache preparation

Run once; this function is CPU-only:

```bash
modal run runtime/embodiedgen_v2_l40s.py::preload_weights
```

It populates the persistent volume with SAM3D, DINOv2 and U2Net assets.

## Benchmark cold vs warm

Run:

```bash
modal run runtime/embodiedgen_v2_l40s.py
```

The local entrypoint calls the same GPU function twice. The first call measures cold dispatch +
container startup; the second call is issued immediately while `scaledown_window=300`, so Modal can
reuse the warm container. Each run records:

- client wall-clock time;
- function start → CUDA ready;
- precompiled extension load time;
- SAM3D generation time;
- post-process time;
- total function time;
- validation metadata.

Per-run results are persisted under `/data/benchmarks/<timestamp>-<label>/benchmark.json`.

Note: the current validated architecture intentionally runs SAM3D generation in a child process and
post-processing in fresh CUDA subprocesses for stability. Therefore a warm container saves container
startup/import overhead, but it does **not** keep the 13 GB SAM3D pipeline resident between requests.
A future persistent-model worker can reduce warm latency further.

## Production split runtime

The current Modal workspace uses profile `shuhuaqaq` and the following persistent storage:

- `modal-3d-embodiedgen-weights` mounted at `/weights` for SAM3D, DINOv2 and U2Net.
- `modal-3d-artifacts` mounted at `/artifacts` for per-job intermediates and final outputs.

The production path is intentionally split by resource type:

```text
CPU prepare (rembg)
    ↓
Sam3DWorker @app.cls, L40S
  - SAM3D model loaded once in @modal.enter
  - min_containers=0
  - max_containers=1
  - scaledown_window=90s
    ↓
CPU xatlas
  - mesh orientation / normalization
  - xatlas UV unwrap
  - no GPU allocated
    ↓
Lite L40S bake
  - gsplat multiview render
  - nvdiffrast/utils3d texture baking
  - no SAM3D weights loaded
  - scaledown_window=30s
    ↓
CPU finalize
  - OBJ / GLB / fallback URDF
  - structural validation
```

This avoids holding an L40S while xatlas unwraps a large mesh. The heavy SAM3D worker remains warm
for 90 seconds after a request, so bursty requests can reuse the already-loaded 13 GB model without
forcing a permanent `min_containers=1` GPU. The later texture-bake GPU worker is deliberately light:
it only loads the precompiled gsplat/nvdiffrast extensions and job artifacts.

Run the CPU-only weight pull in a fresh workspace:

```bash
modal run runtime/embodiedgen_v2_l40s.py::preload_weights
```

Run the split cold/warm benchmark:

```bash
modal run runtime/embodiedgen_v2_l40s.py
```

The benchmark executes two jobs through the same `Sam3DWorker` class handle. The first call measures
cold dispatch plus model loading; the second is issued within the 90-second keep-warm window and is
expected to reuse the resident model container.

## Measured split-runtime performance

Measured on the fresh Modal workspace `shuhuaqaq` with the sample image and the release-only
(no-`nvcc`) runtime. These are observed values, not estimates.

| Stage | Measured time | Resource |
|---|---:|---|
| First CPU rembg request | 46.242 s function / 52.314 s client wall | CPU |
| Warm CPU rembg request | 2.350 s function / 2.743 s client wall | CPU |
| SAM3D resident model load | 36.550 s | L40S cold only |
| SAM3D cold inference | 21.229 s | L40S |
| SAM3D cold method | 26.571 s | L40S |
| SAM3D cold client wall | 100.882 s | includes Modal cold dispatch/startup |
| SAM3D warm inference | 10.527 s | same resident L40S instance |
| SAM3D warm method | 15.562 s | same resident L40S instance |
| SAM3D warm client wall | **15.893 s** | same resident L40S instance |
| CPU PyVista decimation | 7.637 s | CPU |
| CPU xatlas UV unwrap | 10.353 s | CPU |
| CPU mesh/UV stage total | **22.230 s** | CPU |
| Lite-GPU 24-view gsplat render | 0.948 s | L40S, no SAM3D |
| Lite-GPU texture bake | 1.249 s | L40S, no SAM3D |
| Lite-GPU function total | **6.155 s** | L40S, no SAM3D |
| CPU finalize + validation | **2.728 s** | CPU |

The two SAM3D calls returned the exact same resident `instance_id`, proving that the 90-second
`scaledown_window` reused the loaded model. Client wall time fell from 100.882 s cold to 15.893 s
warm. The warm inference itself also improved from 21.229 s to 10.527 s after CUDA/model warm-up.

The original 884,192-face mesh made direct xatlas unwrap take more than five minutes. The production
CPU stage now uses PyVista decimation before UV unwrap. On this sample it reduced 884,192 faces to
88,418 faces in 7.637 s, followed by xatlas in 10.353 s. EmbodiedGen v2's own `backproject_v3`
defaults to `n_max_faces=50000`, so this ~88k-face validation profile remains more conservative than
the upstream default while cutting UV latency dramatically.

The Lite L40S stage does not load the 13 GB SAM3D model. Its actual function time was only 6.155 s,
with ~2.2 s spent on the measured gsplat render + texture bake. A totally cold Modal invocation has
additional scheduling/container-start latency, but that cold-start cost is independent of SAM3D
weight loading and can be amortized with the Lite worker's short keep-warm window for burst traffic.

Final split-run validation succeeded with:

- 94,852 PLY Gaussians;
- 55,355 textured OBJ vertices;
- 88,418 OBJ faces;
- one valid GLB geometry;
- valid URDF mesh reference;
- generated preview video;
- `VALIDATION_OK`.

## Resident rembg CPU worker

Production rembg uses a resident CPU Class rather than creating a new ONNX/U2Net session for every
request:

```python
@app.cls(
    cpu=1.0,
    memory=4096,
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
)
class RembgWorker:
    ...
```

A/B measurement on the `shuhuaqaq` Modal workspace:

| Config | Session load | remove() | method | client wall | Warm reuse |
|---|---:|---:|---:|---:|---|
| 1 CPU + 4 GiB cold | 1.203 s | 1.484 s | 2.091 s | 65.090 s | n/a |
| **1 CPU + 4 GiB warm** | already resident | 1.302 s | 1.864 s | **2.116 s** | same instance |
| 2 CPU + 4 GiB cold | 1.134 s | 0.874 s | 1.682 s | 59.064 s | n/a |
| 2 CPU + 4 GiB warm | already resident | 0.858 s | 1.548 s | 1.841 s | same instance |

The 2-CPU worker saves only 0.275 s of warm client latency, so production keeps the cheaper 1-CPU
worker. The long cold client-wall values are dominated by Modal container cold start; the actual
U2Net session initialization itself was only ~1.2 s once the container began executing.

At the current workspace rates returned by `modal billing rates` on 2026-08-23:

- CPU: $0.04730 per physical core-hour.
- Memory: $0.00800 per GiB-hour.

Therefore 1 CPU + 4 GiB costs $0.07930/hour while the container is allocated. Keeping it idle for the
full 120-second `scaledown_window` costs about **$0.00264 per traffic burst** (0.264 cents). A warm
1.864-second rembg method itself costs only about **$0.000041**. The 2-CPU alternative would cost
about $0.00422 for the same 120-second idle window, ~59.6% more, for only 0.275 s lower measured warm
client latency.
