#!/usr/bin/env bash
# Lab 06 demo: call the team service from ANOTHER computer.
#   bash scripts/demo_requests.sh <public-ip>
# Sends one valid photo (reaches the model) and one text file (rejected before the model).
set -u
HOST="${1:?usage: bash scripts/demo_requests.sh <public-ip>}"
URL="http://$HOST:8000"
cd "$(dirname "$0")/.."

echo "== GET /health"
curl -s "$URL/health"; echo; echo

echo "== POST /predict  valid tray photo (expect HTTP 200, prediction_called true)"
curl -s -w "\n[HTTP %{http_code}]\n" -F "file=@examples/glass373.jpg;type=image/jpeg" \
     -F "request_id=req-cloud-001" "$URL/predict"; echo

echo "== POST /predict  text file (expect HTTP 400, reject_input, prediction_called false)"
curl -s -w "\n[HTTP %{http_code}]\n" -F "file=@examples/not_an_image.txt;type=text/plain" \
     -F "request_id=req-cloud-002" "$URL/predict"; echo
