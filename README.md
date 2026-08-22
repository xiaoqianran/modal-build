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
