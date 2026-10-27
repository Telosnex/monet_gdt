# What the graph's color step can reach at best: Jev's own role
# probabilities, moved from the app's options onto the graph's color list
# (each option's probability to its nearest list color), through the
# graph's rule (graph_rule.Rule), vs the app's rule_v1 triad. Also the
# shipped model's distance for reference is not comparable (different
# targets). Run on the test split.
#   .venv/bin/python scripts/colors_ceiling.py
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
import color_space as D
import graph_rule as GR

man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
jev = {r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'colors.jsonl')))}
dart = {
    r['id']: r['triad'] for r in map(json.loads, open(os.path.join(ROOT, 'out', 'teacher_colors', 'rule_triads.jsonl')))
}
z = np.load(os.path.join(ROOT, 'out', 'colors', 'lists_fit224.npz'))
ids = list(z['ids'])
RGB, SH, K = z['rgb'], z['sh'], z['k']
row = {i: k for k, i in enumerate(ids)}
rule = GR.Rule(os.path.join(ROOT, 'out', 'graph', 'hct_lut.f32'))
test = [m['id'] for m in man if m['split'] == 'test']
de = []
jopt = []
for i in test:
    k = row[i]
    n = int(K[k])
    m = max(n, 64)
    codes = torch.zeros(m, dtype=torch.int64)
    codes[:n] = torch.tensor((RGB[k, :n].astype(np.int64) // 8) @ np.array([1024, 32, 1]))
    sh = torch.zeros(m)
    sh[:n] = torch.tensor(SH[k, :n])
    lab = D.rgb2lab(RGB[k, :n] / 255)
    opts = [o['hex'] for o in jev[i]['options']]
    ol = D.rgb2lab([D.hex2rgb(h) for h in opts])
    near = np.linalg.norm(lab[None] - ol[:, None], axis=-1).argmin(1)
    p = torch.zeros(3, m, dtype=torch.float64)
    for r, role in enumerate(('primary', 'secondary', 'tertiary')):
        for o, h in enumerate(opts):
            p[r, near[o]] += jev[i]['probabilities'][role].get(h, 0.0)
    with torch.no_grad():
        rgb = rule(codes, sh, p.float()).numpy()
    de.append(np.linalg.norm(D.rgb2lab(rgb / 255) - D.rgb2lab([D.hex2rgb(h) for h in dart[i]]), axis=-1))
de = np.array(de)
print(
    f'ceiling (Jev answers through the graph lists + groups + rule), test {len(test)}: ΔE to app triad {de.mean():.2f} '
    f'(P/S/T {"/".join(f"{x:.1f}" for x in de.mean(0))}), all 3 within ΔE 10: {np.mean((de < 10).all(1)):.0%}'
)
