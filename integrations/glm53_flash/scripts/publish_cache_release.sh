#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-glm53-flash-dflash2-vllm0281rc1-fi0618-sm103a-b300-cache-v1}"
REPO="${REPO:-xiaoqianran/modal-build}"
VOLUME="${VOLUME:-modal-build-artifacts}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT

for suffix in cache.tar.gz cache.tar.gz.sha256 manifest.json; do
  modal volume get "$VOLUME" "$TAG.$suffix" "$DIR/$TAG.$suffix" --force
done

(
  cd "$DIR"
  sha256sum -c "$TAG.cache.tar.gz.sha256"
)

PUBLIC="$(python - "$DIR/$TAG.manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as fh:
    payload = json.load(fh)
if payload.get("contains_model_weights") is not False:
    raise SystemExit("refusing release: contains_model_weights must be false")
print(str(bool(payload.get("public_release"))).lower())
PY
)"

if [[ "$PUBLIC" != "true" ]]; then
  echo "refusing public GitHub Release: manifest public_release=false" >&2
  exit 3
fi

ASSETS=(
  "$DIR/$TAG.cache.tar.gz"
  "$DIR/$TAG.cache.tar.gz.sha256"
  "$DIR/$TAG.manifest.json"
)

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "${ASSETS[@]}" --repo "$REPO" --clobber
else
  gh release create "$TAG" "${ASSETS[@]}" \
    --repo "$REPO" \
    --title "$TAG" \
    --notes "Precompiled B300 runtime/JIT/autotune cache for GLM-5.3-Flash NVFP4 + DFlash2. No model weights are included; see manifest for exact compatibility pins and SHA256." 
fi
