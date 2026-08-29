#!/usr/bin/env bash
set -euo pipefail
TAG="${1:?release tag required}"
REPO="${REPO:-xiaoqianran/modal-build}"
DIR="${2:-/tmp/modal-hyworld2-wheels/$TAG}"
mkdir -p "$DIR"
gh release download "$TAG" --repo "$REPO" \
  --pattern "$TAG.wheels.zip" \
  --pattern "$TAG.wheels.zip.sha256" \
  --pattern "$TAG.manifest.json" \
  --dir "$DIR"
(
  cd "$DIR"
  sha256sum -c "$TAG.wheels.zip.sha256"
)
unzip -oq "$DIR/$TAG.wheels.zip" -d "$DIR/bundle"
mapfile -t wheels < <(find "$DIR/bundle/wheels" -maxdepth 1 -type f -name '*.whl' | sort)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "no wheels found in release bundle" >&2
  exit 1
fi
uv pip install --system --no-deps "${wheels[@]}"
