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
