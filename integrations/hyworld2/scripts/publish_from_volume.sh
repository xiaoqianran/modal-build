#!/usr/bin/env bash
set -euo pipefail
TAG="${1:?artifact tag required}"
REPO="${REPO:-xiaoqianran/modal-build}"
VOLUME="${VOLUME:-modal-build-artifacts}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT
for suffix in wheels.zip wheels.zip.sha256 manifest.json; do
  modal volume get "$VOLUME" "$TAG.$suffix" "$DIR/$TAG.$suffix" --force
done
(
  cd "$DIR"
  sha256sum -c "$TAG.wheels.zip.sha256"
)
PUBLIC="$(python - "$DIR/$TAG.manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as fh:
    print(str(bool(json.load(fh)["public_release"])).lower())
PY
)"
if [[ "$PUBLIC" != "true" ]]; then
  echo "refusing public GitHub Release: manifest public_release=false" >&2
  exit 3
fi
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "$DIR/$TAG.wheels.zip" "$DIR/$TAG.wheels.zip.sha256" "$DIR/$TAG.manifest.json" --repo "$REPO" --clobber
else
  gh release create "$TAG" "$DIR/$TAG.wheels.zip" "$DIR/$TAG.wheels.zip.sha256" "$DIR/$TAG.manifest.json" \
    --repo "$REPO" --title "$TAG" \
    --notes "Reproducible modal-build artifact. See manifest for source revisions, ABI, licenses and checksums."
fi
