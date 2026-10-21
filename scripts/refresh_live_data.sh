#!/usr/bin/env bash
# Scheduled refresh for a local machine or server (the web app does not need
# to be running). Example crontab line, every 12 hours:
#   0 */12 * * * cd /path/to/affordable-housing-estimator && ./scripts/refresh_live_data.sh >> refresh.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
"$PY" data_fetcher.py "$@"
