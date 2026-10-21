#!/usr/bin/env bash
# Build the image, start a container and check that /health and the home
# page respond. Used by CI and handy before deploying.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-affordable-housing-estimator:smoke}"
NAME="housing-smoke-$$"
PORT="${SMOKE_PORT:-8080}"

docker build -t "$IMAGE" .
docker run -d --rm --name "$NAME" -p "${PORT}:8000" "$IMAGE" >/dev/null
trap 'docker stop "$NAME" >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/health"; echo
curl -fsS -o /dev/null -w "Home page HTTP %{http_code}\n" "http://127.0.0.1:${PORT}/"
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/estimate" -H 'Content-Type: application/json' \
  -d '{"locality":"Whitefield","area_sqft":1200,"bedrooms":2,"bathrooms":2,"balconies":1,"area_type":"Super built-up Area","ready_to_move":"1"}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("Estimate:", d["estimate"]["price"]["lakh"])'
echo "Smoke test passed."
