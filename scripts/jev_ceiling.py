# ADR D4: Jev vs itself on the test split (jev/<call>.jsonl vs
# jev/<call>_repeat.jsonl). Writes out/jev_ceiling.json; the report
# compares the student against these numbers.
#   .venv/bin/python scripts/jev_ceiling.py
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = lambda f: {r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'jev', f))) if 'error' not in r}
out = {}
a, b = L('mood.jsonl'), L('mood_repeat.jsonl')
ids = [i for i in b if i in a]
W = list(a[ids[0]]['raw'])
x = np.array([[a[i]['raw'][w] for w in W] for i in ids])
y = np.array([[b[i]['raw'][w] for w in W] for i in ids])
out['mood'] = {
    'n': len(ids),
    'per_word_r': {w: float(np.corrcoef(x[:, j], y[:, j])[0, 1]) for j, w in enumerate(W)},
    'mean_abs_diff': float(np.abs(x - y).mean()),
}
a, b = L('shapes.jsonl'), L('shapes_repeat.jsonl')
ids = [i for i in b if i in a]
S = list(a[ids[0]]['scores'])
x = np.array([[a[i]['scores'][s] for s in S] for i in ids])
y = np.array([[b[i]['scores'][s] for s in S] for i in ids])
rank = lambda r: sorted(r['scores'], key=lambda k: -r['scores'][k])
out['shapes'] = {
    'n': len(ids),
    'score_r': float(np.corrcoef(x.ravel(), y.ravel())[0, 1]),
    'same_top1': float(np.mean([rank(a[i])[0] == rank(b[i])[0] for i in ids])),
    'top3_overlap': float(np.mean([len(set(rank(a[i])[:3]) & set(rank(b[i])[:3])) / 3 for i in ids])),
}
a, b = L('colors.jsonl'), L('colors_repeat.jsonl')
ids = [i for i in b if i in a]
top = lambda r, role: max(r['probabilities'][role], key=r['probabilities'][role].get)
out['colors'] = {
    'n': len(ids),
    'same_top': {
        r: float(np.mean([top(a[i], r) == top(b[i], r) for i in ids])) for r in ('primary', 'secondary', 'tertiary')
    },
}
json.dump(out, open(os.path.join(ROOT, 'out', 'jev_ceiling.json'), 'w'), indent=1)
m = out['mood']
print(
    f"mood ({m['n']}): per-word r mean {np.mean(list(m['per_word_r'].values())):.3f} (min {min(m['per_word_r'].values()):.3f}), mean |diff| {m['mean_abs_diff']:.3f}"
)
s = out['shapes']
print(
    f"shapes ({s['n']}, o0 vs o1): score r {s['score_r']:.3f}, same top-1 {s['same_top1']:.0%}, top-3 overlap {s['top3_overlap']:.0%}"
)
c = out['colors']
print(f"colors ({c['n']}): same top pick P/S/T " + ' / '.join(f'{v:.0%}' for v in c['same_top'].values()))
