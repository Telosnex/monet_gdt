# CONCLUDED 2026-10-25: mood reproduced;
# shapes and colors differed from old about as much as Jev from itself.
# Ran against the hex-only color request, before arm a2; rerunning now
# compares the new request with old answers.
#
# Pilot check (ADR step 3): do the monet_gdt Jev clients reproduce the
# libmonet_gdt answers? 20 old images per call (Unsplash, test-only, D8).
#   <venv>/bin/python -u scripts/repro_check.py
# mood: old out/mood_label/scores.jsonl (same 1024 px q88 thumbs).
# shapes: old echoes_full labels (arm echoes_split_noul, options_o0).
# colors: old labels_unsplash_colors.jsonl vs requests from the app code
#   (out/pilot/colors, ids repro_<name>).
# Writes out/pilot/repro.jsonl and prints agreement next to Jev's own
# run-to-run agreement where known.
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from progress import Progress  # noqa: E402
import jev  # noqa: E402
import private_paths as PP

LG = PP.libmonet_gdt()
h = lambda s: hashlib.sha256(s.encode()).hexdigest()


def main():
    old_m = sorted(
        (
            r
            for r in map(json.loads, open(f'{LG}/wallpaper_student/out/mood_label/scores.jsonl'))
            if 'raw' in r and r['src'] == 'unsplash'
        ),
        key=lambda r: h(r['name']),
    )[:20]
    lab = [
        r
        for r in map(json.loads, open(f'{LG}/wallpaper_shapes/out/echoes_full/labels.jsonl'))
        if r.get('arm') == 'echoes_split_noul' and r.get('sheet') == 'options_o0' and not r.get('replicate')
    ]
    old_s = sorted(lab, key=lambda r: h(r['image_id']))[:20]
    reqs = [json.loads(l) for f in glob.glob(f'{ROOT}/out/pilot/colors/colors_requests_*.jsonl') for l in open(f)]
    reqs = [r for r in reqs if r['id'].startswith('repro_')]
    old_c = {
        r['wallpaper']: r
        for r in map(json.loads, open(f'{LG}/wallpaper_student/out/labels_unsplash_colors.jsonl'))
        if 'colors' in r
    }
    jobs = (
        [
            ('mood', r, lambda r=r: jev.mood(r['name'], f"{LG}/wallpaper_student/out/mood_label/thumbs/{r['name']}"))
            for r in old_m
        ]
        + [
            ('shapes', r, lambda r=r: jev.shapes(r['image_id'], f"{LG}/wallpaper_shapes/out/echoes_full/{r['image']}"))
            for r in old_s
        ]
        + [('colors', old_c[q['id'][6:]], lambda q=q: jev.colors(q)) for q in reqs]
    )
    p = Progress('repro check: 60 Jev calls (20 mood, 20 shapes, 20 colors), 8 parallel', len(jobs))
    res = []
    with ThreadPoolExecutor(8) as ex:
        for (call, old, _), new in zip(jobs, ex.map(lambda j: j[2](), jobs)):
            res.append((call, old, new))
            p.add()
    p.done()
    with open(f'{ROOT}/out/pilot/repro.jsonl', 'w') as f:
        for call, old, new in res:
            f.write(json.dumps({'call': call, 'new': new}) + '\n')
    errs = [n for _, _, n in res if 'error' in n]
    print(f'{len(errs)} errors', [e['error'][:120] for e in errs[:3]])
    ok = [(c, o, n) for c, o, n in res if 'error' not in n]
    M = [(o, n) for c, o, n in ok if c == 'mood']
    if M:
        W = jev.MOOD['words']
        a = np.array([[o['raw'][w] for w in W] for o, _ in M])
        b = np.array([[n['raw'][w] for w in W] for _, n in M])
        print(
            f'mood: {len(M)} images, mean |diff| {np.abs(a - b).mean():.3f}, per-word r {np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(10)]):.3f}'
        )
    S = [(o, n) for c, o, n in ok if c == 'shapes']
    if S:
        top = lambda d: sorted(d, key=lambda k: -d[k])[:3]
        same1 = np.mean(
            [
                top(
                    o['scores']
                    if False
                    else {k: jev.shape_score(o['nouls'][k], o['extra']['d'][k], o['extra']['t'][k]) for k in o['nouls']}
                )[0]
                == top(n['scores'])[0]
                for o, n in S
            ]
        )
        e = np.array([[o['nouls'][k] for k in o['nouls']] for o, _ in S])
        f = np.array([[n['parts'][k]['echoes'] for k in o['nouls']] for o, n in S])
        print(
            f'shapes: {len(S)} images, echoes mean |diff| {np.abs(e - f).mean():.3f}, r {np.corrcoef(e.ravel(), f.ravel())[0, 1]:.3f}, same top-1 {same1:.0%}'
        )
    C = [(o, n) for c, o, n in ok if c == 'colors']

    def first_free(p):  # the old wallpaperColorTriad, as the old labels used
        used, t = set(), []
        for r in ('primary', 'secondary', 'tertiary'):
            h = next(h for h, _ in sorted(p[r].items(), key=lambda kv: -kv[1]) if h not in used)
            used.add(h)
            t.append(h)
        return t

    for _, n in C:
        n['colors'] = first_free(n['probabilities'])
    if C:
        same = np.mean([o['colors'] == n['colors'] for o, n in C])
        prim = np.mean([o['colors'][0] == n['colors'][0] for o, n in C])
        opts = np.mean([n['colors'][0] in [x['hex'] for x in n['options']] for _, n in C])
        print(f'colors: {len(C)} images, same triad {same:.0%}, same primary {prim:.0%} (primary offered: {opts:.0%})')
    tok = {c: [n['tokens'] for cc, _, n in ok if cc == c] for c in ('mood', 'shapes', 'colors')}
    print('tokens per call (median):', {c: int(np.median(v)) for c, v in tok.items() if v})


if __name__ == '__main__':
    main()
