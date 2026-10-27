# CONCLUDED 2026-10-25. Winners: arm a2 (hex + HCT + share labels), the
# vivid merge, and the rule_v1 triad (beat the on-device rule 27-11). They
# now live in telosnex lib/features/system_agents/wallpaper_color_decisions.dart
# (parity with rule_v1.py: 60/60); these scripts are the record, not
# production. Votes and Jev answers: results/.
#
# Reorder each triad so S is whichever of S, T is closer in hue to P
# (the demo UI shows S as a message color, which rewards a big
# P-S hue gap). Hue = libmonet HCT hue from candidates.json. If P, S or T
# is near gray (chroma < 8), hue is unreliable: use CIELAB ΔE76 instead.
# Writes arms <arm>_st into the telosnex raw.jsonl for that corpus.
#   python3 reorder_st.py <corpus> <raw_prompt_ab*.jsonl> <arm>...
import json
import math
import os
import sys

corpus, raw, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
TX = os.path.join(
    os.environ['TELOSNEX'], 'dev', 'wallpaper_colors', 'out', corpus
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


out, swaps = [], {a: [0, 0] for a in arms}
for r in map(json.loads, open(raw)):
    a = r['arm']
    if a not in arms or not r.get('triad'):
        continue
    hct = {o['hex']: o['hct'] for o in C[r['wallpaper']]['merged']}
    p, s, t = r['triad']
    if min(hct[x][1] for x in (p, s, t)) >= 8:
        d = lambda x: abs((hct[x][0] - hct[p][0] + 180) % 360 - 180)
    else:
        d = lambda x: math.dist(lab(x), lab(p))
    swap = d(t) < d(s)
    swaps[a][0] += swap
    swaps[a][1] += 1
    out.append({**r, 'arm': a + '_st', 'triad': [p, t, s] if swap else [p, s, t], 'st_swapped': swap})
with open(f'{TX}/raw.jsonl', 'a') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
for a, (k, n) in swaps.items():
    print(f'{a}: S and T swapped on {k} of {n} images')
