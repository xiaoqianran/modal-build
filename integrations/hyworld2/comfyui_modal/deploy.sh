#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec modal deploy modal_app.py --stream-logs
