#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv sync --locked
exec uv run modal deploy modal_app.py --stream-logs
