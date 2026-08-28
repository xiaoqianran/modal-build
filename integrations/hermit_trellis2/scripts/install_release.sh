#!/usr/bin/env bash
set -euo pipefail
TAG="${1:?release tag required}"
REPO="${REPO:-xiaoqianran/modal-build}"
DIR="${2:-/tmp/modal-wheels}"
mkdir -p "$DIR"
gh release download "$TAG" --repo "$REPO" --pattern '*.wheels.zip' --dir "$DIR"
unzip -oq "$DIR/$TAG.wheels.zip" -d "$DIR/wheels"
uv pip install --system --no-index --find-links "$DIR/wheels" flash-attn cumesh flex-gemm o-voxel
