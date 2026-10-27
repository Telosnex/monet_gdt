# CONCLUDED 2026-10-25. Real + MONET synthetic beats real only on wallpaper
# test sets (mood); became ADR D1. Results: results/.
#
# Synthetic A/B: mood heads on the shipped graph's `feature`, trained on
# different real/synthetic mixes, scored on held-out real and synthetic.
#   <venv>/bin/python -u experiments/synthetic_ab/ab.py
# Head: ridge regression 192 -> 10 raw Jev mood scores (alpha by 5-fold CV
# on the training set). Metric: mean over words of Pearson r vs Jev.
# Arms (train sets): real 2000; mix 1000 + 1000 (5 draws); real + synthetic
# 4000; synthetic 2000. CIs: 1000 bootstrap draws of the test images.
import hashlib
import json
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')
GRAPH = os.path.join(
    os.environ['TELOSNEX'], 'assets', 'models', 'libmonet_gdt.onnx'
)  # TELOSNEX: Telosnex app checkout (maintainer-only)
WORDS = json.load(open(os.path.join(HERE, '..', '..', 'prompts', 'mood.json')))['words']


def features(rows):
    sha = hashlib.sha256(open(GRAPH, 'rb').read()).hexdigest()[:12]
    cache = os.path.join(DATA, f'feature_{sha}.npz')
    have = dict(np.load(cache, allow_pickle=True)['f'].item()) if os.path.exists(cache) else {}
    s = ort.InferenceSession(GRAPH, providers=['CPUExecutionProvider'])
    for r in rows:
        if r['id'] in have:
            continue
        im = Image.open(os.path.join(DATA, 'thumbs', r['source'], f"{r['id']}.webp")).convert('RGBA')
        have[r['id']] = s.run(['feature'], {'image': np.asarray(im)})[0][0]
    np.savez(cache, f=np.array(have, dtype=object))
    return np.array([have[r['id']] for r in rows], np.float32), sha


def ridge_fit(X, Y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = Y.mean(0)

    def solve(Z, Y, a):
        return np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))

    folds = np.arange(len(Z)) % 5
    best = None
    for a in (1, 10, 100, 1000, 10000):
        err = 0
        for k in range(5):
            tr, va = folds != k, folds == k
            W = solve(Z[tr], Y[tr], a)
            err += ((Z[va] @ W + Y[tr].mean(0) - Y[va]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, a)
    W = solve(Z, Y, best[1])
    return lambda X: ((X - mu) / sd) @ W + ym


def mean_r(P, Y):
    return float(np.mean([np.corrcoef(P[:, j], Y[:, j])[0, 1] for j in range(Y.shape[1])]))


def main():
    R = [json.loads(l) for l in open(os.path.join(DATA, 'jev_mood.jsonl'))]
    R = list({r['id']: r for r in R if 'error' not in r}.values())
    X, sha = features(R)
    Y = np.array([[r['raw'][w] for w in WORDS] for r in R], np.float32)
    g = lambda arm, split: np.array([i for i, r in enumerate(R) if r['arm'] == arm and r['split'] == split])
    rtr, rte, str_, ste = g('real', 'train'), g('real', 'test'), g('synthetic', 'train'), g('synthetic', 'test')
    res = {
        'graph_sha12': sha,
        'n': {'real_train': len(rtr), 'real_test': len(rte), 'syn_train': len(str_), 'syn_test': len(ste)},
    }
    # Label distributions.
    res['jev_mean'] = {
        a: dict(zip(WORDS, np.round(Y[i].mean(0), 3).tolist()))
        for a, i in (('real', np.r_[rtr, rte]), ('synthetic', np.r_[str_, ste]))
    }
    rng = np.random.default_rng(0)
    arms = {
        'real 2000': [rtr],
        'real 2000 + syn 2000': [np.r_[rtr, str_]],
        'syn 2000': [str_],
        'mix 1000 + 1000': [np.r_[rng.choice(rtr, 1000, False), rng.choice(str_, 1000, False)] for _ in range(5)],
        'real 1000': [rng.choice(rtr, 1000, False) for _ in range(5)],
    }
    boot = [rng.integers(0, len(rte), len(rte)) for _ in range(1000)]
    preds = {}
    for name, sets in arms.items():
        Pr = np.mean([ridge_fit(X[t], Y[t])(X[rte]) for t in sets], 0)
        Ps = np.mean([ridge_fit(X[t], Y[t])(X[ste]) for t in sets], 0)
        preds[name] = Pr
        res[name] = {
            'real_test_r': round(mean_r(Pr, Y[rte]), 4),
            'syn_test_r': round(mean_r(Ps, Y[ste]), 4),
            'real_test_per_word': dict(
                zip(WORDS, [round(float(np.corrcoef(Pr[:, j], Y[rte][:, j])[0, 1]), 3) for j in range(10)])
            ),
        }
        print(
            f"{name:22s} real test r {res[name]['real_test_r']:.3f}   synthetic test r {res[name]['syn_test_r']:.3f}",
            flush=True,
        )
    base = preds['real 2000']
    for name in ('real 2000 + syn 2000', 'mix 1000 + 1000', 'syn 2000', 'real 1000'):
        d = [mean_r(preds[name][b], Y[rte][b]) - mean_r(base[b], Y[rte][b]) for b in boot]
        res[name]['delta_vs_real2000_ci95'] = [
            round(float(np.percentile(d, 2.5)), 4),
            round(float(np.percentile(d, 97.5)), 4),
        ]
        print(f"  {name} - real 2000 on real test: 95% CI {res[name]['delta_vs_real2000_ci95']}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'ab_mood.json'), 'w'), indent=1)
    print(
        'Jev mean, real vs synthetic:',
        {w: (res['jev_mean']['real'][w], res['jev_mean']['synthetic'][w]) for w in WORDS},
    )


if __name__ == '__main__':
    main()
