# Parity: graph_rule.pick (the graph's rule_v1) vs the app's Dart
# libmonet monetGdtRuleV1Triad on all Jev color labels, same options, same
# probabilities (float64: Jev rounds them, so exact cutoff ties occur), Dart's rounded HCT (out/teacher_colors/rule_triads.jsonl).
#   .venv/bin/python scripts/check_graph_rule.py
import json
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import graph_rule as GR

dart = {r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'out', 'teacher_colors', 'rule_triads.jsonl')))}
same = n = 0
bad = []
for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'colors.jsonl'))):
    d = dart[r['id']]
    opts = [o['hex'] for o in r['options']]
    h = torch.tensor(d['hct'], dtype=torch.float64)
    rgb = torch.tensor([[int(x[i : i + 2], 16) for i in (1, 3, 5)] for x in opts], dtype=torch.float32)
    p = torch.tensor(
        [[r['probabilities'][role].get(x, 0.0) for x in opts] for role in ('primary', 'secondary', 'tertiary')],
        dtype=torch.float64,
    )
    with torch.no_grad():
        i = GR.pick(
            h[:, 0], h[:, 1], h[:, 2], GR.rgb2lab(rgb / 255).double(), torch.ones(len(opts), dtype=torch.bool), p
        )
    got = [opts[k] for k in i.tolist()]
    n += 1
    if got == d['triad']:
        same += 1
    elif len(bad) < 5:
        bad.append((r['id'], got, d['triad']))
print(f'graph rule vs app rule on Jev options: same triad {same}/{n}')
[print(' ', b) for b in bad]
