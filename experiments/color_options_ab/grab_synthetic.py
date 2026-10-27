# CONCLUDED 2026-10-25. Only fetched synthetic images early for the color
# options A/B (<=12 merged vs 32 vs 255; run with telosnex
# dev/wallpaper_colors/run.dart). <=12 won 53-36.
#
# Synthetic images for the color-options A/B before the main download gets
# to them: stream synthetic tars from the start and keep selected rows
# (data/selection.jsonl) until N generic and N genre are saved. Writes the
# same files the main download writes (data/images/<part>/<id>.jpg + .json).
#   ../../.venv/bin/python -u grab_synthetic.py
import json
import os
import sys
import tarfile
from huggingface_hub import HfFileSystem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from progress import Progress  # noqa: E402
import select_corpus as S  # noqa: E402

N = 25
sel = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'selection.jsonl'))]
by = {}
for r in sel:
    if r['part'] != 'megalith':
        by.setdefault(r['shard'], {})[r['id']] = r
got = {'generic': set(), 'genre': set()}
p = Progress(f'grabbing {N} generic + {N} genre synthetic JPGs from the start of MONET tars', 2 * N)
fs = HfFileSystem()
for shard in sorted(by):
    if all(len(v) >= N for v in got.values()):
        break
    want = by[shard]
    with fs.open(f'{S.MONET}/{shard}.tar', block_size=2**22) as raw:
        tf = tarfile.open(fileobj=raw, mode='r|')
        for m in tf:
            key, _, ext = m.name.partition('.')
            r = want.get(key)
            if r is None or ext not in ('jpg', 'json'):
                continue
            part = r['part']
            if len(got[part]) >= N and key not in got[part]:
                continue
            d = os.path.join(ROOT, 'data', 'images', part)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f'{key}.{ext}'), 'wb').write(tf.extractfile(m).read())
            if ext == 'jpg' and key not in got[part]:
                got[part].add(key)
                p.add()
            if all(len(v) >= N for v in got.values()):
                break
p.done()
json.dump(
    {k: sorted(v) for k, v in got.items()},
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'synthetic_ids.json'), 'w'),
)
print({k: len(v) for k, v in got.items()})
