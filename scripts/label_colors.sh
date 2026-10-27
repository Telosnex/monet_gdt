#!/bin/sh
# Step 4 (colors): app-built requests (telosnex
# dev/monet_gdt/color_requests_test.dart: vivid merge, hex + HCT + share
# labels) for every manifest image, then Jev -> jev/colors.jsonl
# (per-role probabilities; the triad is app code, rule_v1). Resumable.
# Requests and agent JPEGs: out/colors_requests/ (rebuildable). The
# requests are built by the Telosnex app: set TELOSNEX=<telosnex checkout>
# to rebuild them; without it, the existing out/colors_requests/ are sent.
# Log: out/logs/label_colors.log
cd "$(dirname "$0")/.." || exit 1
R=$(pwd)
: "${USER_API_KEY_JEV:?set USER_API_KEY_JEV (Jev API key, api.typesafe.ai)}"
if [ -n "$TELOSNEX" ]; then
  echo "[$(date +%H:%M:%S)] telosnex $(git -C "$TELOSNEX" rev-parse --short HEAD) (+ uncommitted color changes)"
  (cd "$TELOSNEX" && MONET_IMAGES=$R/out/colors_images.tsv MONET_OUT=$R/out/colors_requests \
    flutter test dev/monet_gdt/color_requests_test.dart --run-skipped 2>&1 | grep --line-buffered -E "^\[|All tests|Some tests|rror")
else
  [ -d out/colors_requests ] || { echo "no out/colors_requests: set TELOSNEX to build them" >&2; exit 1; }
  echo "[$(date +%H:%M:%S)] TELOSNEX unset: using the existing out/colors_requests"
fi
PYTHONWARNINGS=ignore .venv/bin/python -u scripts/label.py colors data/all_ids.json jev/colors.jsonl --parallel 24 --colors-dir out/colors_requests
echo "[$(date +%H:%M:%S)] all done"
