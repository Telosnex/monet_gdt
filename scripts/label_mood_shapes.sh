#!/bin/sh
# Step 4 (part): full Jev mood then shapes labels for data/all_ids.json.
# Resumable (label.py skips answered IDs). Log: out/logs/label_mood_shapes.log
cd "$(dirname "$0")/.." || exit 1
: "${USER_API_KEY_JEV:?set USER_API_KEY_JEV (Jev API key, api.typesafe.ai)}"
export PYTHONWARNINGS=ignore
.venv/bin/python -u scripts/label.py mood data/all_ids.json jev/mood.jsonl --parallel 24
.venv/bin/python -u scripts/label.py shapes data/all_ids.json jev/shapes.jsonl --parallel 24
echo "[$(date +%H:%M:%S)] all done"
