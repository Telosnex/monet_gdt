# CONCLUDED 2026-10-25. Museum CC0 art hurt Wallhaven and Unsplash in every
# mix; dropped (ADR D1). Results: results/.
#
# Jev mood labels for the museum images (same request as synthetic_ab).
#   <venv>/bin/python -u experiments/museum_ab/label_mood.py
# Split by sha256(id) over data/manifest.jsonl (both museums together):
# 2,000 train, 500 test, rest unused. Appends data/jev_mood.jsonl; resumable.
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'synthetic_ab'))
from progress import Progress  # noqa: E402
import label_mood as L  # noqa: E402  (synthetic_ab/label_mood.py: ask, prompt)

L.DATA = DATA  # read thumbnails from museum_ab/data/thumbs/<museum>/


def plan():
    M = sorted(
        (json.loads(l) for l in open(os.path.join(DATA, 'manifest.jsonl'))),
        key=lambda m: hashlib.sha256(m['id'].encode()).hexdigest(),
    )
    return [{**m, 'source': m['museum'], 'arm': 'museum', 'split': 'train'} for m in M[:2000]] + [
        {**m, 'source': m['museum'], 'arm': 'museum', 'split': 'test'} for m in M[2000:2500]
    ]


def main():
    path = os.path.join(DATA, 'jev_mood.jsonl')
    done = {json.loads(l)['id'] for l in open(path) if 'error' not in json.loads(l)} if os.path.exists(path) else set()
    P = plan()
    todo = [m for m in P if m['id'] not in done]
    p = Progress(
        f'Jev mood labels for museum images ({len(done)} already labeled)', len(P), start_done=len(P) - len(todo)
    )
    bad = tok = 0
    with open(path, 'a') as f, ThreadPoolExecutor(16) as ex:
        for r in ex.map(L.ask, todo):
            f.write(json.dumps(r) + '\n')
            f.flush()
            p.add()
            if 'error' in r:
                bad += 1
            else:
                tok += r['tokens']
    p.done()
    print(f'{bad} failed, {tok} tokens, ${tok * 0.042 / 1e6:.2f}', flush=True)


if __name__ == '__main__':
    main()
