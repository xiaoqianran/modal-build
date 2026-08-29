# EmbodiedGen v2.0.0 patches

`production/` contains the only patches/loaders used by the current production runtime in
`runtime/embodiedgen_v2_l40s.py`, including the headless/no-JIT loaders and the Retexture lazy-Delight
patch (`retexture-lazy-delight.patch`) that prevents GPT/segmentation imports when `delight=False`.

`legacy/` contains earlier CLI/subprocess/JIT-loader experiments retained only for historical
reproduction. Production code must not import or mount files from `legacy/`.
