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
  - default scaledown_window=30s (`cost_first`)
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
  - default scaledown_window=10s (`cost_first`)
    ↓
CPU finalize
  - OBJ / GLB / fallback URDF
  - structural validation
```

This avoids holding an L40S while xatlas unwraps a large mesh. The current production default is the
`cost_first` profile: heavy SAM3D remains warm for 30 seconds, while the later texture-bake L40S stays
warm for only 10 seconds. `min_containers=0`, `buffer_containers=0`, and `max_containers=1` prevent
permanent or proactive GPU reservations. Longer `balanced`/`burst` profiles remain available when
traffic is intentionally dense.

Run the CPU-only weight pull in a fresh workspace:

```bash
modal run runtime/embodiedgen_v2_l40s.py::preload_weights
```

Run the split cold/warm benchmark with the production default:

```bash
modal run runtime/embodiedgen_v2_l40s.py::benchmark_split --profile cost_first
```

The benchmark accepts `min_cost`, `cost_first`, `balanced`, or `burst`. Calls are deliberately issued
back-to-back so the warm measurement can reuse the same resident class instance even when using the
shorter `cost_first` window.

## Measured split-runtime performance (pre-50k baseline)

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

The original 884,192-face mesh made direct xatlas unwrap take more than five minutes. This baseline
CPU stage used PyVista decimation before UV unwrap. On this sample it reduced 884,192 faces to
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
request. The A/B measurements below were originally taken with a 120-second warm window; the current
`cost_first` static default is 60 seconds (`min_containers=0`, `buffer_containers=0`,
`max_containers=1`).

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

Therefore 1 CPU + 4 GiB costs $0.07930/hour while the container is allocated. The historical
120-second warm window cost about **$0.00264 per traffic burst**; the current 60-second `cost_first`
window costs about **$0.00132**. A warm 1.864-second rembg method itself costs only about
**$0.000041**. The 2-CPU alternative was rejected because it cost ~59.6% more for only 0.275 s lower
measured warm client latency.

## Mesh/UV 50k optimization

The production MeshWorker uses `fast-simplification==0.2.0` and targets exactly 50,000 faces before
xatlas. It is CPU-only, uses 4 physical cores + 8 GiB, and now defaults to the `cost_first` 30-second
window with `min_containers=0`, `buffer_containers=0`, and `max_containers=1`. The original A/B data
below used the longer 90-second window.

A/B benchmark on the same persisted 884,192-face SAM3D mesh:

| Variant | Simplify | xatlas | Compute total | Client wall |
|---|---:|---:|---:|---:|
| **fast-simplification / 4 CPU** | **0.957 s** | 4.375 s | **5.332 s** | 14.044 s cold function |
| fast-simplification / 8 CPU | 1.339 s | 4.518 s | 5.857 s | 11.387 s |
| fast-simplification / 16 CPU | 1.590 s | 6.189 s | 7.779 s | 12.540 s |
| PyVista / 8 CPU | 7.597 s | 3.083 s | 10.680 s | 16.552 s |

More CPU did not improve xatlas on this workload. Four cores had the best measured compute time and
lowest resource cost. Compared with the previous 88,418-face PyVista path, the production mesh stage
changed from about 22.23 seconds to a resident-worker warm client wall of **6.46 seconds**:

- MeshWorker cold method: 7.636 s; cold client wall: 13.368 s.
- MeshWorker warm method: **6.024 s**; warm client wall: **6.460 s**.
- Warm simplify: 0.850 s.
- Warm xatlas: 4.369 s.
- Cold and warm calls returned the same worker instance id.

The critical path no longer exports the 884k-face raw OBJ and stores the small `bake_mesh.npz`
uncompressed; the lossless high-poly state remains available in `sample_00_state.pkl`.

The resulting 50k mesh was sent through the real Lite L40S texture stage and CPU finalizer:

- Lite L40S 24-view render: 0.254 s.
- Lite L40S texture bake: 1.201 s.
- Lite L40S function total: 4.158 s.
- CPU finalize + structural validation: 1.508 s.
- Final OBJ: 32,112 vertices / **50,000 faces**.
- GLB geometry, URDF mesh reference and preview video all validated.
- Final result: `VALIDATION_OK`.

At the current workspace rates (CPU $0.04730/core-hour, memory $0.00800/GiB-hour), a 4 CPU + 8 GiB
MeshWorker costs $0.2532/hour while allocated. The historical 90-second idle tail was about
$0.00633; the current 30-second `cost_first` tail is about **$0.00211 per traffic burst**.
`min_containers=0` means it scales to zero when idle.

## Final production benchmark (50k fast-simplification)

A final end-to-end cold + warm run was executed after all production cleanup, using the release-only
no-`nvcc` image and the 50,000-face `MeshWorker`. The Modal run exited with code 0 and both jobs
finished with `VALIDATION_OK`.

Warm request path (client-observed wall time):

| Stage | Warm client wall | Warm function/method |
|---|---:|---:|
| RembgWorker (1 CPU / 4 GiB) | **2.225 s** | 1.861 s |
| Sam3DWorker (resident L40S) | **15.988 s** | 15.307 s |
| MeshWorker (4 CPU / 8 GiB) | **9.738 s** | 9.269 s |
| Lite L40S texture stage | **7.730 s** | 7.258 s |
| CPU finalize + validation | **5.281 s** | 3.421 s |
| **Sequential warm end-to-end** | **40.962 s** | — |

The warm end-to-end value is the sum of the five sequential client-wall measurements from the final
production run. It improves the prior ~49-second warm baseline by about **8 seconds (~16%)** while
also reducing the textured mesh from ~88k to exactly 50k faces.

Important final-run details:

- Rembg cold client wall: 50.058 s; warm: 2.225 s; same resident instance.
- SAM3D model load on this cold run: 48.039 s.
- SAM3D cold client wall: 131.253 s; warm: 15.988 s; same resident L40S instance.
- SAM3D warm inference: 12.172 s.
- Mesh cold downstream call: 12.338 s client wall; warm: 9.738 s.
- Mesh warm simplify: 1.188 s; xatlas: 6.736 s.
- Lite L40S warm render24: 0.031 s; texture bake: 0.915 s.
- Warm downstream (MeshWorker + Lite L40S + finalize): **22.749 s**.

Final warm job validation:

- 94,852 PLY Gaussians;
- 32,628 textured OBJ vertices;
- exactly **50,000 OBJ faces**;
- one valid GLB geometry;
- valid URDF mesh reference;
- generated preview video;
- `VALIDATION_OK`.

Cold client-wall values include Modal scheduling/container startup and are expected to vary much more
than warm values. That benchmark used the former latency-oriented windows: Rembg 120 s, heavy SAM3D
90 s, MeshWorker 90 s, Lite L40S 30 s. Those values are now preserved as the `balanced` profile rather
than the production default.

## SAM3D sampling optimization

A resident-L40S profiler showed that, after the one-time CUDA/decoder warm-up, the dominant SAM3D
cost is the two sampling stages rather than mesh/GS decoding or GLB post-processing. On a genuinely
warm pass:

| SAM3D stage | Time |
|---|---:|
| Stage 1 sparse-structure sampling | 3.546 s |
| Stage 2 sparse-latent sampling | 3.105 s |
| Mesh decoder | 0.216 s |
| Gaussian decoder | 0.049 s |
| Pointmap | 0.064 s |
| Preprocess | 0.035 s |
| GLB/postprocess wrapper | 0.021 s |
| Full warm pipeline | 9.782 s |

The first pass after model load had a one-time ~9.7-second mesh-decoder/kernel warm-up; subsequent
passes decoded the mesh in ~0.22 seconds.

Sampling A/B on the same resident L40S, same image and seed:

| Schedule | Distillation | Warm SAM3D pipeline | Relative to 25/25 |
|---|---|---:|---:|
| 25 / 25 | off | 9.878 s | baseline |
| 16 / 25 | off | 8.577 s | -13% |
| 25 / 16 | off | 8.405 s | -15% |
| **16 / 16** | **off** | **7.289 s** | **-26%** |
| 8 / 8 | shortcut on | 4.300 s | -56% |

All three full end-to-end candidates (25/25, 16/16, shortcut 8/8) completed the 50k mesh, texture,
GLB, URDF and video pipeline with `VALIDATION_OK`. The production default is nevertheless **16/16**,
because its output remains much closer to the 25/25 reference.

Quality comparison against the 25/25 final output:

| Metric | 16/16 | Shortcut 8/8 |
|---|---:|---:|
| 60-frame preview PSNR | **31.54 dB** | 21.73 dB |
| 60-frame preview SSIM | **0.9718** | 0.9119 |
| Surface Chamfer / baseline bbox diagonal | **0.4595%** | 0.8242% |
| Surface RMS / baseline bbox diagonal | **0.5101%** | 1.0543% |
| P95 surface distance / baseline bbox diagonal | **0.8522%** | 2.2219% |
| P99 surface distance / baseline bbox diagonal | **1.0825%** | 3.6451% |

The 16/16 final mesh remained extremely close in size to baseline (50,000 faces after the common
MeshWorker step), while shortcut 8/8 changed Gaussian count and overall extent more noticeably.
Therefore `SAM3D_STAGE1_STEPS=16` and `SAM3D_STAGE2_STEPS=16` are now the production defaults.
Shortcut 8/8 is documented only as a possible future `turbo` quality/speed mode, not as the default.

## Heavy L40S state handoff optimization

The heavy SAM3D L40S no longer writes PLY files or commits the shared Volume. Profiling showed that
these operations were expensive and highly variable while holding a $1.95/hour GPU.

State-only Heavy L40S focused warm measurement before replacing Volume commit:

| Operation | Time |
|---|---:|
| SAM3D 16/16 inference | 8.319 s |
| `pack_state` | 0.013 s |
| local pickle serialization | 0.006 s |
| copy 20.414 MiB state into Volume mount | 0.027 s |
| explicit Volume commit | 1.375 s |
| Heavy method total | 10.006 s |
| Heavy client wall | 10.247 s |

A later full production run observed the same explicit Volume commit taking **6.339 seconds**, proving
that commit latency can vary by several seconds. That run still completed with `VALIDATION_OK`, but
its warm Heavy SAM3D client wall rose to 15.915 s solely because the expensive L40S was waiting for
storage persistence.

The serialized state was also reduced losslessly from roughly 30.6 MiB to **20.414 MiB** by storing
mesh face indices as `int32` rather than `int64`; ~450k mesh vertices are far below the int32 index
limit.

Gaussian PLY generation and alignment were moved to the CPU MeshWorker. Reconstructing both raw and
aligned PLY files from state took only **0.065 s CPU**. Compared field-by-field with the previous GPU
path, xyz/features were identical or ~1e-7-level different, scales/quaternions stayed below ~5e-7,
and opacity differed only at ~2e-5 mean from floating-point inverse-activation arithmetic. The CPU
PLYs passed the real Lite L40S texture stage and final CPU validation with `VALIDATION_OK`.

### 20.4 MiB cross-stage transport benchmark

The target is to let Heavy L40S stop immediately after inference/packing rather than wait on Volume
commit. A CPU-only Modal benchmark compared the available handoff mechanisms using a 20.4 MiB
payload (Modal function arguments/results above 2 MiB automatically use blob storage):

| Transport | Producer/parent blocking time | Consumer/read time |
|---|---:|---:|
| direct `spawn(payload)` | 1.60-2.53 s | child became available 0.66-2.67 s later |
| Function return blob | 2.12-5.67 s client wall | — |
| **Modal Dict** | **0.43-0.61 s put** | **0.40-0.55 s get** |

`Queue` is not suitable because each Queue item is limited to 1 MiB. `NetworkFileSystem` has immediate
sharing semantics but is deprecated in the installed Modal SDK and is not used for the production
baseline.

The billing report for the Dict/transport benchmark contained only ordinary CPU and Memory charges
(no separate Dict resource line); the complete CPU-only transport experiment cost about $0.00029 CPU
plus $0.000047 Memory. Dict items expire after seven days of inactivity, but production removes each
transient state immediately after CPU persistence succeeds.

Production handoff is therefore:

```text
Heavy L40S
  SAM3D 16/16
  -> pack_state
  -> pickle (~20.4 MiB)
  -> Dict.put(job_id, state)
  -> return / release expensive GPU

CPU MeshWorker
  -> Dict.get(job_id)
  -> persist state.pkl to Volume
  -> rebuild raw/aligned PLY (~0.065 s CPU)
  -> fast-simplification to 50k
  -> xatlas
  -> Volume commit (CPU pays the wait)
  -> Dict.pop(job_id)
```

At $1.95/hour, removing 1.375-6.339 seconds of Volume-commit wait saves about **$0.00074-$0.00343 of
L40S time per request**, in addition to improving latency and freeing scarce GPU capacity earlier.


### Validated production Dict handoff

The real SAM3D state was then run through the Dict handoff on one resident L40S. The second (warm)
call measured:

| Heavy L40S operation | Warm time |
|---|---:|
| SAM3D 16/16 inference | **9.675 s** |
| `pack_state` | 0.007 s |
| pickle serialization | 0.007 s |
| Dict put (20.414 MiB) | **0.403 s** |
| Heavy method total | **10.367 s** |
| Heavy client wall | **10.716 s** |

Cold and warm calls returned the same SAM3D `instance_id`. The Heavy worker therefore spends almost
all of its warm lifetime doing useful inference, plus roughly 0.4 seconds handing the state to CPU.
It no longer performs a Volume commit or writes PLY files.

A clean CPU-only Dict-to-MeshWorker run measured:

| CPU MeshWorker operation | Time |
|---|---:|
| Dict get | **0.597 s** |
| persist `state.pkl` + deserialize | 0.041 s |
| rebuild raw + aligned PLY | 0.056 s |
| fast-simplification to 50k | 0.437 s |
| xatlas | 5.501 s |
| Dict delete | 0.437 s |
| MeshWorker method total | **8.651 s** |

The CPU worker also has a `volume-fallback` retry path. If a retry occurs after state was already
persisted but after the transient Dict item was removed, it reuses `sample_00_state.pkl` rather than
forcing another paid SAM3D run.

The resulting CPU-generated aligned PLY then completed the real Lite L40S stage in **4.215 s**
(function time; 24-view render 0.184 s, texture bake 1.200 s), followed by CPU finalization in
**2.200 s**, ending with `VALIDATION_OK` (95,560 PLY Gaussians, 32,517 OBJ vertices, 50,000 OBJ
faces, valid GLB, URDF mesh reference and preview video).

The previous full warm sequential benchmark was 35.866 s with a 15.915 s Heavy-SAM3D client stage
that included a 6.339 s Volume commit. Replacing only that measured Heavy stage with the validated
10.716 s Dict handoff gives a conservative same-shape estimate of roughly **30.7 s warm end-to-end**.
A new full cold+warm run was intentionally not purchased just to make this number prettier; the
individual production stages and final artifacts have already been validated, and the next natural
production request can provide an end-to-end observation at no extra benchmark-only GPU cost.

At $1.95/hour, the measured warm Heavy L40S method (10.367 s) costs about **$0.00562** of active GPU
time. The measured Lite L40S function (4.215 s) costs about **$0.00228**, so useful active L40S work
is about **$0.00790 per warm request** before idle keep-warm tails. This makes keep-warm policy, not
active computation, the next largest GPU-cost lever.

Billing snapshot for the final optimization tests on 2026-08-23:

- Heavy Dict benchmark (one cold + one warm): $0.07150005 L40S + $0.01040601 CPU + $0.00938667 memory = **$0.09129273**.
- Lite L40S final validation: **$0.00629796** total, including $0.00541682 L40S.
- The current workspace's cumulative `modal-3d-embodiedgen` R&D/test usage for the day was **$1.80120448**. This includes all earlier failed experiments, profiling, A/B tests and cold starts; it is not a per-image production cost.

## Autoscale cost profiles

Idle keep-warm tails became the largest avoidable GPU cost after state handoff removed Heavy-L40S
Volume waits. Production therefore uses explicit traffic profiles. All stages retain
`min_containers=0`; static workers also set `buffer_containers=0` and `max_containers=1`, so idle
capacity eventually reaches zero and no second GPU container is created accidentally.

Current profiles:

| Profile | Rembg CPU | Heavy L40S | Mesh CPU | Lite L40S | Maximum idle-tail cost / isolated burst |
|---|---:|---:|---:|---:|---:|
| `min_cost` | 2 s | 2 s | 2 s | 2 s | **$0.00283** |
| `cost_first` (static fallback) | 60 s | 30 s | 30 s | 10 s | **$0.03048** |
| `balanced` | 120 s | 90 s | 90 s | 30 s | $0.09011 |
| `burst` | 300 s | 180 s | 120 s | 60 s | $0.17733 |

These costs include the full allocated resource set, not only the GPU line item. At the current
2026-08-23 workspace rates used by the runtime calculator:

- CPU: $0.04730 / physical core-hour;
- memory: $0.00800 / GiB-hour;
- L40S: $1.95 / GPU-hour.

Thus the allocated hourly rates of the four warm containers are approximately $0.0793/h for Rembg,
$2.4898/h for Heavy SAM3D (L40S + 6 CPU + 32 GiB), $0.2532/h for MeshWorker, and $2.2672/h for Lite
L40S (L40S + 4 CPU + 16 GiB).

`cost_first` cuts the theoretical isolated-burst idle tail from $0.09011 to $0.03048, a reduction of
about **66.2%** versus the former `balanced` policy. `min_cost` cuts it to about $0.00283, but usually
forces a cold start on the next non-overlapping request and is intended for extremely sparse traffic.
The Heavy SAM3D cold model-load penalty has measured roughly 36-48 seconds, so keeping its $2.4898/h
container warm for 30 seconds costs about $0.02075. In a genuinely sparse workload, `min_cost` wins;
in a bursty workload, 30 seconds can be cheaper overall if it avoids enough repeated model loads.

The request-level default is now `auto`. AUTO is intentionally conservative and cost-first rather than
latency-first:

- one request seen in the last 60 seconds -> `min_cost`;
- two or more requests seen in the last 60 seconds -> `cost_first`;
- `balanced` and `burst` are never selected automatically; they remain explicit operator choices.

Each traffic observation is stored as an independent event in `modal-3d-embodiedgen-traffic`, so
concurrent request writers do not race on a shared JSON counter/list. Events older than 60 seconds are
pruned opportunistically on the next request. The classifier itself is a pure function and is covered
by unit tests, so the event store can later be replaced by Redis/SQL without changing the policy.

Most importantly, AUTO does **not** use `with_options()`. Dynamic options would create independent
container pools and could turn a warm SAM3D into a needless cold start when a profile changes. The
runtime now calls `update_autoscaler()` on the same resident Class/function pool and changes only
`scaledown_window` while keeping `min_containers=0`, `buffer_containers=0`, and `max_containers=1`.

Zero-GPU control-plane validation on 2026-08-23:

```text
traffic state cleared
AUTO request #1 -> min_cost   (1 request / 60s, idle-tail ceiling $0.00282750)
AUTO request #2 -> cost_first (2 requests / 60s, idle-tail ceiling $0.03047778)
```

The test only hydrated the Modal app and updated autoscaler settings; it did not invoke any model
worker or allocate L40S compute.

Control/benchmark examples:

```bash
# Zero-compute policy check for this app run; it records one synthetic traffic event.
modal run runtime/embodiedgen_v2_l40s.py::autoscale_policy_check --profile auto

# End-to-end benchmark under automatic policy.
modal run runtime/embodiedgen_v2_l40s.py::benchmark_split --profile auto

# Explicit latency-oriented override when an operator knows a dense period is coming.
modal run runtime/embodiedgen_v2_l40s.py::benchmark_split --profile balanced
```

The runtime prints both the requested and selected profile plus its computed idle-tail cost before
remote model work starts. `autoscale_policy_check` is diagnostic only: a standalone `modal run` ends
that ephemeral app after the check. Production orchestration must call `select_request_profile()` and
`apply_autoscale_profile()` in the same app run before dispatching the workers, as `benchmark_split`
already does. AUTO deliberately caps itself at `cost_first`: the 90/180-second GPU tails buy latency,
but are not the economical default when the primary requirement is minimum spend.
