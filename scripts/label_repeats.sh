#!/bin/sh
# ADR D4: a second run of each call on the test split (data/test_ids.json)
# -> jev/*_repeat.jsonl, the Jev ceiling. Shapes use sheet o1 (new
# question order). Colors resend the same app-built requests.
# Log: out/logs/label_repeats.log
cd "$(dirname "$0")/.." || exit 1
: "${USER_API_KEY_JEV:?set USER_API_KEY_JEV (Jev API key, api.typesafe.ai)}"
export PYTHONWARNINGS=ignore
.venv/bin/python -u scripts/label.py mood data/test_ids.json jev/mood_repeat.jsonl --parallel 24
.venv/bin/python -u scripts/label.py shapes data/test_ids.json jev/shapes_repeat.jsonl --parallel 24 --sheet o1
.venv/bin/python -u scripts/label.py colors data/test_ids.json jev/colors_repeat.jsonl --parallel 24 --colors-dir out/colors_requests
echo "[$(date +%H:%M:%S)] all done"
