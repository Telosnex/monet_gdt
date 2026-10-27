# CONCLUDED 2026-10-25. Winners: arm a2 (hex + HCT + share labels), the
# vivid merge, and the rule_v1 triad (beat the on-device rule 27-11). They
# now live in telosnex lib/features/system_agents/wallpaper_color_decisions.dart
# (parity with rule_v1.py: 60/60); these scripts are the record, not
# production. Votes and Jev answers: results/.
#
# Color rule (2026-10-25), Jev-led. Options: merged_chroma (each
# merge group keeps its most chromatic member with >= 1% of the group).
# Jev (arm 2) probabilities bound each role: only options with p >= K x the
# role's best remaining p are "plausible"; the rule picks among them.
#   P: the most chromatic plausible option.
#   T: plausible, not within ΔE 10 of P; colorful (HCT chroma >= CMIN)
#      options nearest P's complement hue in widening bands (30, 60, 90,
#      120, 180°), most chromatic in the first band that has any; if none
#      colorful, most chromatic.
#   S: plausible, not within ΔE 10 of P or T; colorful and >= HMIN° from
#      both P and T -> most chromatic; else max W x chroma/max + (1-W) x
#      hue gap from P / 180 (tone gap / 100 when P is gray).
# Baseline arm ondevice_a2: arm 2 on the old merge, each pick moved to its
# group's most chromatic member with >= 1% of the group, S/T >= ΔE 10 from
# earlier picks (the on-device chroma1_x10 rule, on Jev's groups).
#   python3 rule_v1.py   -> appends arms to telosnex out/monet_color_ab3/raw.jsonl
import json
import math
import statistics as st
import itertools
import os

K, CMIN, HMIN, W = float(os.environ.get('RULE_K', '0.25')), 16, 30, 0.5
ARM = os.environ.get('RULE_ARM', 'rule_v1')
TX = os.path.join(
    os.environ['TELOSNEX'], 'dev', 'wallpaper_colors', 'out', 'monet_color_ab3'
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
hd = lambda a, b: abs((a - b + 180) % 360 - 180)


def load(f):
    return {r['wallpaper']: r for r in map(json.loads, open(f)) if r.get('triad')}


def probs(r, role):
    return {x['value']: x['probability'] for x in r['probabilities'][role]}


def rule(r, opts):
    H = {o['hex']: o['hct'] for o in opts}
    ch = lambda x: H[x][1]

    def plaus(role, pool):
        p = probs(r, role)
        pool = [x for x in pool if x in p] or pool
        top = max(p.get(x, 0) for x in pool)
        return [x for x in pool if p.get(x, 0) >= K * top]

    P = max(plaus('primary', list(H)), key=ch)
    pool = [x for x in H if x != P and de(x, P) >= 10] or [x for x in H if x != P]
    pt = plaus('tertiary', pool)
    col = [x for x in pt if ch(x) >= CMIN]
    T = None
    if col and ch(P) >= CMIN:
        comp = (H[P][0] + 180) % 360
        for band in (30, 60, 90, 120, 180):
            b = [x for x in col if hd(H[x][0], comp) <= band]
            if b:
                T = max(b, key=ch)
                break
    T = T or max(pt, key=ch)
    pool = [x for x in H if x not in (P, T) and de(x, P) >= 10 and de(x, T) >= 10] or [x for x in H if x not in (P, T)]
    ps = plaus('secondary', pool)
    good = [x for x in ps if ch(x) >= CMIN and hd(H[x][0], H[P][0]) >= HMIN and hd(H[x][0], H[T][0]) >= HMIN]
    if good:
        S = max(good, key=ch)
    else:
        mc = max(ch(x) for x in ps) or 1
        gap = (lambda x: hd(H[x][0], H[P][0]) / 180) if ch(P) >= CMIN else (lambda x: abs(H[x][2] - H[P][2]) / 100)
        S = max(ps, key=lambda x: W * ch(x) / mc + (1 - W) * gap(x))
    return [P, S, T]


def ondevice(r, opts, cand):
    sh = {c['hex']: c['share'] for c in cand}
    out = []
    hct = {o['hex']: o for o in opts}
    for h in r['triad']:
        m = hct[h]['members']
        tot = sum(sh.get(x, 0) for x in m)
        ok = [x for x in m if sh.get(x, 0) >= 0.01 * tot and all(de(x, y) >= 10 for y in out)] or [h]
        out.append(max(ok, key=lambda x: math.hypot(*lab(x)[1:])))
    return out


old, new = load('out/raw_ab3_merged.jsonl'), load('out/raw_ab3_chroma.jsonl')
rows = []
for n in sorted(C):
    if n in old and ARM == 'rule_v1':
        rows.append(
            {'arm': 'ondevice_a2', 'wallpaper': n, 'triad': ondevice(old[n], C[n]['merged'], C[n]['candidates'])}
        )
    if n in new:
        rows.append(
            {'arm': ARM, 'wallpaper': n, 'triad': rule(new[n], C[n]['merged_chroma']), 'jev_triad': new[n]['triad']}
        )
with open(f'{TX}/raw.jsonl', 'a') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')


def hue_c(h):
    L, a, b = lab(h)
    return math.degrees(math.atan2(b, a)) % 360, math.hypot(a, b)


def good3(t):
    return all(hue_c(x)[1] >= 16 for x in t) and all(
        hd(hue_c(a)[0], hue_c(b)[0]) >= 30 for a, b in itertools.combinations(t, 2)
    )


for arm in sorted({r['arm'] for r in rows}):
    T = [r['triad'] for r in rows if r['arm'] == arm]
    gap = lambda i: st.median(
        hd(hue_c(t[0])[0], hue_c(t[i])[0]) for t in T if hue_c(t[0])[1] >= 16 and hue_c(t[i])[1] >= 16
    )
    print(
        f"{arm:12s} n {len(T)}: 3 distinct colorful {sum(map(good3, T))}; chroma P/S/T "
        f"{'/'.join(f'{st.median(hue_c(t[i])[1] for t in T):.0f}' for i in range(3))}; hue gap P-S {gap(1):.0f}°, P-T {gap(2):.0f}°"
    )
R = [r for r in rows if r['arm'] == ARM]
print(
    f'{ARM} vs Jev top pick (new merge): P same',
    sum(r['triad'][0] == r['jev_triad'][0] for r in R),
    '/ S same',
    sum(r['triad'][1] == r['jev_triad'][1] for r in R),
    '/ T same',
    sum(r['triad'][2] == r['jev_triad'][2] for r in R),
    'of',
    len(R),
)
