# CONCLUDED 2026-10-25. Real + MONET synthetic beats real only on wallpaper
# test sets (mood); became ADR D1. Results: results/.
#
# A test-only set for the synthetic A/B (never trained on in monet_gdt).
#   <venv>/bin/python -u experiments/synthetic_ab/unsplash_eval.py [unsplash|wallhaven]
# 500 images from libmonet_gdt wallpaper_colors/<unsplash|wallhaven_train>
# (lowest sha256(name)),
# resized to long side 384 (as MONET thumbnails), Jev mood with prompts/mood.json.
# Caveat: the shipped graph's backbone trained on these the test set photos, so
# absolute r is inflated; arm-vs-arm differences are still fair (the ridge
# heads never see the test set).
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import ab
import label_mood as L

import sys

NAME = sys.argv[1] if len(sys.argv) > 1 else 'unsplash'
# LIBMONET_GDT: libmonet_gdt checkout (maintainer-only)
SRC = os.path.join(
    os.environ['LIBMONET_GDT'], 'wallpaper_colors', {'unsplash': 'unsplash', 'wallhaven': 'wallhaven_train'}[NAME]
)
DIR = os.path.join(ab.DATA, 'thumbs', NAME)


def rows():
    os.makedirs(DIR, exist_ok=True)
    names = sorted(
        (f for f in os.listdir(SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))),
        key=lambda f: hashlib.sha256(f.encode()).hexdigest(),
    )[:500]
    out = []
    for f in names:
        i = os.path.splitext(f)[0]
        p = os.path.join(DIR, f'{i}.webp')
        if not os.path.exists(p):
            im = Image.open(os.path.join(SRC, f)).convert('RGB')
            im.thumbnail((384, 384), Image.LANCZOS)
            im.save(p, 'WEBP', quality=90)
        out.append({'id': i, 'source': NAME, 'arm': NAME, 'split': 'test'})
    return out


def main():
    U = rows()
    path = os.path.join(ab.DATA, f'jev_mood_{NAME}.jsonl')
    done = {json.loads(l)['id'] for l in open(path) if 'error' not in json.loads(l)} if os.path.exists(path) else set()
    with open(path, 'a') as f, ThreadPoolExecutor(16) as ex:
        for r in ex.map(L.ask, [u for u in U if u['id'] not in done]):
            f.write(json.dumps(r) + '\n')
    Ru = list({r['id']: r for r in map(json.loads, open(path)) if 'error' not in r}.values())
    R = list(
        {
            r['id']: r for r in map(json.loads, open(os.path.join(ab.DATA, 'jev_mood.jsonl'))) if 'error' not in r
        }.values()
    )
    X, _ = ab.features(R + Ru)
    Xr, Xu = X[: len(R)], X[len(R) :]
    W = ab.WORDS
    Y = np.array([[r['raw'][w] for w in W] for r in R])
    Yu = np.array([[r['raw'][w] for w in W] for r in Ru])
    g = lambda a, s: np.array([i for i, r in enumerate(R) if r['arm'] == a and r['split'] == s])
    rtr, rte, st, ste = g('real', 'train'), g('real', 'test'), g('synthetic', 'train'), g('synthetic', 'test')
    print(f'{len(Ru)} the test set labeled, {sum(r.get("tokens", 0) for r in Ru)} tokens')
    # 1. Jev's view: label means, and which group the test set is closer to.
    mr, ms, mu = Y[np.r_[rtr, rte]].mean(0), Y[np.r_[st, ste]].mean(0), Yu.mean(0)
    print(f'Jev mean   megalith / {NAME} / synthetic')
    for j, w in enumerate(W):
        print(f'  {w:20s} {mr[j]:.2f} / {mu[j]:.2f} / {ms[j]:.2f}')
    pos = float(((mu - mr) @ (ms - mr)) / ((ms - mr) @ (ms - mr)))
    print(f'the test set on the megalith(0) -> synthetic(1) line, Jev labels: {pos:.2f}')
    # 2. The feature's view: a real-vs-synthetic classifier applied to the test set.
    tr = np.r_[rtr, st]
    mu_, sd_ = Xr[tr].mean(0), Xr[tr].std(0) + 1e-6
    Z = (Xr[tr] - mu_) / sd_
    t = np.r_[np.zeros(len(rtr)), np.ones(len(st))]
    w = np.zeros(192)
    b = 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= 0.1 * (Z.T @ (p - t) / len(t) + 1e-2 * w)
        b -= 0.1 * (p - t).mean()
    pr = lambda Xx: 1 / (1 + np.exp(-(((Xx - mu_) / sd_) @ w + b)))
    print(
        f'P(synthetic) by feature: megalith test {pr(Xr[rte]).mean():.2f}, synthetic test {pr(Xr[ste]).mean():.2f}, '
        f'{NAME} {pr(Xu).mean():.2f} (share > 0.5: {(pr(Xu) > 0.5).mean():.0%})'
    )
    # 3. The A/B on the test set.
    rng = np.random.default_rng(0)
    res = {'unsplash_on_line': pos, 'unsplash_p_synthetic': float(pr(Xu).mean())}
    arms = {'real 2000': rtr, 'real 2000 + syn 2000': np.r_[rtr, st], 'syn 2000': st}
    P = {n: ab.ridge_fit(Xr[t], Y[t])(Xu) for n, t in arms.items()}
    boot = [rng.integers(0, len(Yu), len(Yu)) for _ in range(1000)]
    for n in arms:
        res[n] = ab.mean_r(P[n], Yu)
        line = f'{n:22s} the test set r {res[n]:.3f}'
        if n != 'real 2000':
            d = [ab.mean_r(P[n][k], Yu[k]) - ab.mean_r(P['real 2000'][k], Yu[k]) for k in boot]
            line += f'  vs real 2000: 95% CI {np.percentile(d, 2.5):+.3f}..{np.percentile(d, 97.5):+.3f}'
        print(line)
    json.dump(res, open(os.path.join(ab.OUT, f'ab_mood_{NAME}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
