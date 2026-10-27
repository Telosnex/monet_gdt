# The wallpaper population for mood percentiles (ADR D6; docs/FONTS.md step 1).
# The app's mood map (export_app.py) turns a raw score into "percentile
# among these images"; fonts are matched on those percentiles.
# Population: Jev's mood labels (raw, 0..1) for 1,500 wallpaper-matched
# Megalith photos (ADR D11; data/population_ids.json, a fixed sample of
# data/matched.jsonl), CC0 / public domain (R1). Replaces the 272 Pixel /
# Wallhaven / Unsplash Lite wallpapers libmonet_gdt calibrated on.
# The matched images are stored at 448 px; the corpus was labeled at 1024.
# Jev at 448 vs 1024 on 100 corpus test photos (jev/mood_res448.jsonl; ids data/res_check_ids.json): r 0.96-0.995
# per word, mean shift -0.027..+0.013; each word's mean shift is removed.
#   <venv>/bin/python -u scripts/wall_population.py  -> fonts/wall_population.json
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = [
    'artistic',
    'attention-grabbing',
    'calm',
    'complex',
    'disorderly',
    'dramatic',
    'formal',
    'playful',
    'strong',
    'wide',
]


def raw(p):
    return {r['id']: r['raw'] for r in map(json.loads, open(os.path.join(ROOT, p))) if 'error' not in r}


def main():
    ids = json.load(open(os.path.join(ROOT, 'data', 'population_ids.json')))
    lab = raw('jev/mood_population.jsonl')
    ids = [i for i in ids if i in lab]
    hi, lo = raw('jev/mood.jsonl'), raw('jev/mood_res448.jsonl')
    chk = [i for i in lo if i in hi]
    shift = {w: float(np.mean([lo[i][w] - hi[i][w] for i in chk])) for w in WORDS}
    pop = {w: sorted(round(float(np.clip(lab[i][w] - shift[w], 0, 1)), 6) for i in ids) for w in WORDS}
    out = {
        'version': 'matched_v1',
        'words': WORDS,
        'source': f'Jev mood labels of {len(ids)} wallpaper-matched Megalith photos (448 px), minus the per-word 448-vs-1024 mean shift ({len(chk)} photos)',
        'shift_448_minus_1024': {w: round(v, 4) for w, v in shift.items()},
        'population': pop,
    }
    p = os.path.join(ROOT, 'fonts', 'wall_population.json')
    json.dump(out, open(p, 'w'), indent=0)
    print(p, len(ids), {w: round(float(np.median(v)), 3) for w, v in pop.items()})


if __name__ == '__main__':
    main()
