# CONCLUDED 2026-10-25. Museum CC0 art hurt Wallhaven and Unsplash in every
# mix; dropped (ADR D1). Results: results/.
#
# Museum A/B: does CC0 museum art in training help mood heads, alone or with
# MONET synthetic? Same head and metric as synthetic_ab/ab.py (ridge on the
# shipped graph's `feature`, mean Pearson r vs Jev over 10 words).
#   <venv>/bin/python -u experiments/museum_ab/ab.py
# Train pools (2,000 each, labeled in synthetic_ab and here): real
# (megalith), synthetic, museum. Arms of 4,000 hold the size fixed where it
# matters. Test sets: megalith 500, synthetic 300, museum ~480, Unsplash 500,
# Wallhaven ~494 (the last two test-only). CIs: 1000 bootstrap draws.
import hashlib
import json
import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, '..', 'synthetic_ab')
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, SYN)
from progress import Progress  # noqa: E402
import ab as A  # noqa: E402

GRAPH = A.GRAPH
W = A.WORDS


def load(path, thumbs):
    R = list({r['id']: r for r in map(json.loads, open(path)) if 'error' not in r}.values())
    for r in R:
        r['thumb'] = os.path.join(thumbs, r['source'], f"{r['id']}.webp")
    return R


def features(R):
    sha = hashlib.sha256(open(GRAPH, 'rb').read()).hexdigest()[:12]
    cache = os.path.join(HERE, 'data', f'feature_{sha}.npz')
    have = dict(np.load(cache, allow_pickle=True)['f'].item()) if os.path.exists(cache) else {}
    old = os.path.join(SYN, 'data', f'feature_{sha}.npz')
    if os.path.exists(old):
        have.update(np.load(old, allow_pickle=True)['f'].item())
    key = lambda r: r['id']
    todo = [r for r in R if key(r) not in have]
    s = ort.InferenceSession(GRAPH, providers=['CPUExecutionProvider'])
    p = Progress('computing the shipped graph feature per image (~30 ms each)', len(R), start_done=len(R) - len(todo))
    for r in todo:
        have[key(r)] = s.run(['feature'], {'image': np.asarray(Image.open(r['thumb']).convert('RGBA'))})[0][0]
        p.add()
    p.done()
    np.savez(cache, f=np.array(have, dtype=object))
    return np.array([have[key(r)] for r in R], np.float32)


def main():
    sd = os.path.join(SYN, 'data')
    R = (
        load(os.path.join(sd, 'jev_mood.jsonl'), os.path.join(sd, 'thumbs'))
        + load(os.path.join(HERE, 'data', 'jev_mood.jsonl'), os.path.join(HERE, 'data', 'thumbs'))
        + load(os.path.join(sd, 'jev_mood_unsplash.jsonl'), os.path.join(sd, 'thumbs'))
        + load(os.path.join(sd, 'jev_mood_wallhaven.jsonl'), os.path.join(sd, 'thumbs'))
    )
    X = features(R)
    Y = np.array([[r['raw'][w] for w in W] for r in R], np.float32)
    g = lambda arm, split: np.array([i for i, r in enumerate(R) if r['arm'] == arm and r['split'] == split])
    tr = {'real': g('real', 'train'), 'syn': g('synthetic', 'train'), 'museum': g('museum', 'train')}
    te = {
        'megalith': g('real', 'test'),
        'synthetic': g('synthetic', 'test'),
        'museum': g('museum', 'test'),
        'unsplash': g('unsplash', 'test'),
        'wallhaven': g('wallhaven', 'test'),
    }
    print({k: len(v) for k, v in {**tr, **te}.items()}, flush=True)
    rng = np.random.default_rng(0)
    half = lambda a: rng.choice(tr[a], 1000, False)
    arms = {
        'real 2000': [tr['real']],
        'real + syn (4000)': [np.r_[tr['real'], tr['syn']]],
        'real + museum (4000)': [np.r_[tr['real'], tr['museum']]],
        'museum + syn (4000)': [np.r_[tr['museum'], tr['syn']]],
        'real + syn + museum (6000)': [np.r_[tr['real'], tr['syn'], tr['museum']]],
        'real 1000 + syn 2000 + museum 1000 (4000)': [np.r_[half('real'), tr['syn'], half('museum')] for _ in range(5)],
        'museum 2000': [tr['museum']],
    }
    P = {}
    res = {}
    p = Progress('fitting ridge heads (arm x test set)', len(arms))
    for name, sets in arms.items():
        fits = [A.ridge_fit(X[t], Y[t]) for t in sets]
        P[name] = {k: np.mean([f(X[i]) for f in fits], 0) for k, i in te.items()}
        p.add()
    p.done()
    boot = {k: [rng.integers(0, len(i), len(i)) for _ in range(1000)] for k, i in te.items()}
    head = f"{'arm':44s}" + ''.join(f'{k:>11s}' for k in te)
    print('\nmean r vs Jev (test sets)\n' + head)
    for name in arms:
        res[name] = {k: round(A.mean_r(P[name][k], Y[i]), 4) for k, i in te.items()}
        print(f'{name:44s}' + ''.join(f'{res[name][k]:11.3f}' for k in te))
    print('\n95% CI of the difference vs "real + syn (4000)"')
    base = 'real + syn (4000)'
    for name in arms:
        if name == base:
            continue
        cells = []
        for k, i in te.items():
            d = [A.mean_r(P[name][k][b], Y[i][b]) - A.mean_r(P[base][k][b], Y[i][b]) for b in boot[k]]
            lo, hi = np.percentile(d, [2.5, 97.5])
            cells.append(f'{lo:+.3f}..{hi:+.3f}')
            res[name][f'{k}_ci_vs_base'] = [round(float(lo), 4), round(float(hi), 4)]
        print(f'{name:44s}' + ''.join(f'{c:>16s}' for c in cells))
    mm = {
        a: Y[i].mean(0)
        for a, i in (
            ('megalith', np.r_[tr['real'], te['megalith']]),
            ('museum', np.r_[tr['museum'], te['museum']]),
            ('synthetic', np.r_[tr['syn'], te['synthetic']]),
            ('wallhaven', te['wallhaven']),
        )
    }
    print('\nJev mean   ' + ' / '.join(mm))
    for j, w in enumerate(W):
        print(f'  {w:20s} ' + ' / '.join(f'{mm[a][j]:.2f}' for a in mm))
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    json.dump(
        {'r': res, 'jev_mean': {a: dict(zip(W, np.round(v, 3).tolist())) for a, v in mm.items()}},
        open(os.path.join(HERE, 'out', 'ab_museum.json'), 'w'),
        indent=1,
    )


if __name__ == '__main__':
    main()
