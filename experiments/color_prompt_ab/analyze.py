# CONCLUDED 2026-10-25. Winners: arm a2 (hex + HCT + share labels), the
# vivid merge, and the rule_v1 triad (beat the on-device rule 27-11). They
# now live in telosnex lib/features/system_agents/wallpaper_color_decisions.dart
# (parity with rule_v1.py: 60/60); these scripts are the record, not
# production. Votes and Jev answers: results/.
#
# Automatic comparison of the color prompt arms (run_arms.py output).
#   python3 analyze.py   -> prints a table; writes out/summary.json
import json
import math
import os
import statistics as st
import collections

TX = os.path.join(
    os.environ['TELOSNEX'], 'dev', 'wallpaper_colors', 'out', 'monet_color_ab'
)  # TELOSNEX: Telosnex app checkout (maintainer-only)
C = json.load(open(f'{TX}/candidates.json'))


def lab(h):
    c = [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c]
    X = (0.4124 * c[0] + 0.3576 * c[1] + 0.1805 * c[2]) / 0.95047
    Y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    Z = (0.0193 * c[0] + 0.1192 * c[1] + 0.9505 * c[2]) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return 116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))


de = lambda a, b: math.dist(lab(a), lab(b))
R = collections.defaultdict(dict)
for r in map(json.loads, open('out/raw_prompt_ab.jsonl')):
    if r.get('triad'):
        R[r['arm']][r['wallpaper']] = r
names = sorted(C)
arms = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']
LABEL = {
    'a0': 'hex (control)',
    'a1': 'hex+HCT',
    'a2': 'hex+HCT+share',
    'a3': 'swatch strip',
    'a4': 'swatch/option',
    'a5': 'swatch+hex',
    'a6': 'swatch+hex+HCT',
    'a7': 'a2, role-only brief',
    'a8': 'narrow Nouls',
    'a9': 'a2 in 3 calls',
    'a10': 'a9, no hue rules',
}
opt = lambda n, h: next(o for o in C[n]['merged'] if o['hex'] == h)
hue_d = lambda a, b: abs((a - b + 180) % 360 - 180)
summary = {}
print(
    f"{'arm':22s}{'same3':>6s}{'sameP':>6s}{'rep ΔE P/S/T':>15s}{'=ctrl P':>8s}{'topP':>6s}{'Pshare':>7s}{'chroma P/S/T':>14s}{'hueΔ S/T':>10s}{'tokens':>7s}"
)
for a in arms:
    A, B = R[f'prompt_{a}'], R[f'prompt_{a}_rep']
    ns = [n for n in names if n in A and n in B]
    t1 = {n: A[n]['triad'] for n in ns}
    t2 = {n: B[n]['triad'] for n in ns}
    ctrl = R['prompt_a0']
    a2 = R['prompt_a2']
    s = {
        'n': len(ns),
        'same_triad': st.mean(t1[n] == t2[n] for n in ns),
        'same_primary': st.mean(t1[n][0] == t2[n][0] for n in ns),
        'rep_de': [st.mean(de(t1[n][i], t2[n][i]) for n in ns) for i in range(3)],
        'primary_eq_control': st.mean(t1[n][0] == ctrl[n]['triad'][0] for n in ns),
        'top_p': st.median(A[n]['probabilities']['primary'][0]['probability'] for n in ns),
        'p_share': st.median(opt(n, t1[n][0])['share'] for n in ns) * 100,
        'chroma': [st.median(opt(n, t1[n][i])['hct'][1] for n in ns) for i in range(3)],
        'hue_step': [
            st.median(
                hue_d(opt(n, t1[n][0])['hct'][0], opt(n, t1[n][i])['hct'][0])
                for n in ns
                if opt(n, t1[n][0])['hct'][1] > 8
            )
            for i in (1, 2)
        ],
        'tokens': st.median(A[n]['usage']['input_tokens'] for n in ns),
    }
    summary[a] = s
    print(
        f"{a + ' ' + LABEL[a]:22s}{s['same_triad']:6.0%}{s['same_primary']:6.0%}{'/'.join(f'{x:.1f}' for x in s['rep_de']):>15s}"
        f"{s['primary_eq_control']:8.0%}{s['top_p']:6.2f}{s['p_share']:6.1f}%{'/'.join(f'{x:.0f}' for x in s['chroma']):>14s}"
        f"{'/'.join(f'{x:.0f}' for x in s['hue_step']):>10s}{s['tokens']:7.0f}"
    )
print(
    '\n(=ctrl P: same primary as the control run 1; a0 vs itself is its own run 1, so 100%. hueΔ: median hue step of S and T from P, when P is not gray.'
)
print(' a8 topP is P(true) normalized over options, not comparable.)')
# Arm 9: does Jev's confidence predict stability? Top-to-second ratio of primary in run 1 vs same primary in run 2.
print('\nArm 9, confidence vs stability (a0-a7 pooled): primary top-to-second ratio in run 1 -> same primary in run 2')
pairs = []
for a in [x for x in arms if x != 'a8']:
    A, B = R[f'prompt_{a}'], R[f'prompt_{a}_rep']
    for n in names:
        if n in A and n in B:
            p = A[n]['probabilities']['primary']
            ratio = p[0]['probability'] / max(p[1]['probability'], 1e-9) if len(p) > 1 else 99
            pairs.append((ratio, A[n]['triad'][0] == B[n]['triad'][0]))
for lo, hi in ((0, 1.25), (1.25, 2), (2, 4), (4, 1e9)):
    g = [s for r, s in pairs if lo <= r < hi]
    print(
        f'  ratio {lo:>4}-{hi if hi < 1e9 else "inf":<4}: {len(g):4d} images, same primary {st.mean(g):.0%}'
        if g
        else ''
    )
json.dump(summary, open('out/summary.json', 'w'), indent=1)
