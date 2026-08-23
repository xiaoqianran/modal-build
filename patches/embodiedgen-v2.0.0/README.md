# EmbodiedGen v2.0.0 patches

`production/` contains the only patches/loaders used by the current production runtime in
`runtime/embodiedgen_v2_l40s.py`.

`legacy/` contains earlier CLI/subprocess/JIT-loader experiments retained only for historical
reproduction. Production code must not import or mount files from `legacy/`.
