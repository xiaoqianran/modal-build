# GLM-5.3-Flash / B300 compile cache

This integration owns the reusable compile/autotune cache for the single-B300
`GLM-5.3-Flash NVFP4 + DFlash2` runtime.

The bundle contains **no model weights**. It is limited to generated runtime
artifacts such as vLLM metadata, FlashInfer autotune/JIT output, TileLang
kernels and other cache roots under `/root/.cache`.

## Compatibility key

`glm53-flash-dflash2-vllm0281rc1-fi0618-sm103a-b300-cache-v1`

- GPU: NVIDIA B300 / `sm_103a`
- vLLM image: `vllm/vllm-openai:glm53-flash`
- vLLM: `0.28.1rc1`
- FlashInfer: `0.6.18`
- GLM target revision: `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49`
- DFlash2 revision: `bf582e4eacc1810f76656d1811693ff6c6737d2a`

## One-time migration of the first successful cache

The original runtime used two Volumes. Seed them into the unified compile-cache
Volume entirely on CPU:

```bash
modal run integrations/glm53_flash/build/cache_bundle.py --action seed-legacy
```

## Package the warmed unified cache

After the B300 runtime has completed its warmup and explicitly committed the
unified cache Volume:

```bash
modal run integrations/glm53_flash/build/cache_bundle.py --action inspect
modal run integrations/glm53_flash/build/cache_bundle.py --action package
```

The package step writes these files to `modal-build-artifacts`:

- `<tag>.cache.tar.gz`
- `<tag>.cache.tar.gz.sha256`
- `<tag>.manifest.json`

## Publish

```bash
integrations/glm53_flash/scripts/publish_cache_release.sh
```

The public GitHub Release in `xiaoqianran/modal-build` is the source of truth.
Runtime workspaces restore it with a CPU-only job before any B300 is started.
