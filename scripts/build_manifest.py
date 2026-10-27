# Step 2c of docs/ADR_train_monet_gdt.md: data/manifest.jsonl from the selection
# and the downloaded files. One line per usable image:
#   id, part, split, path, sha256 (our hash of the stored JPG), width,
#   height, monet_source, monet_revision, license, and for Megalith (D9):
#   flickr_photo_id, flickr_page (Megalith url_source), image_url,
#   megalith_revision.
#   .venv/bin/python -u scripts/build_manifest.py
# Drops (counted in data/manifest_stats.json): not downloaded, unreadable,
# short side < 512, aspect > 3:1, Megalith photo absent from Megalith.
# Megalith url_source comes from its 10 parquet files (url_source column
# only, ~790 MB total), cached as data/megalith_pages.json.
import json
import os
import sys
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, HERE)
from progress import Progress  # noqa: E402
import hashlib

MEGALITH_REV = 'dcdd5dc4381ee21da0cd378dfd50fb827d80e960'
MEGALITH = f'datasets/madebyollin/megalith-10m@{MEGALITH_REV}/data'


def megalith_pages(want):
    cache = os.path.join(DATA, 'megalith_pages.json')
    pages = json.load(open(cache)) if os.path.exists(cache) else {}
    if want <= set(pages):
        return pages
    fs = HfFileSystem()
    files = sorted(fs.ls(MEGALITH, detail=False))
    p = Progress('reading Megalith url_source (10 parquet files, ~80 MB each) for Flickr page URLs', len(files))
    for f in files:
        for u in pq.read_table(fs.open(f), columns=['url_source']).column(0).to_pylist():
            pid = u.rstrip('/').rsplit('/', 1)[-1]
            if pid in want:
                pages[pid] = u
        p.add()
    p.done()
    json.dump(pages, open(cache, 'w'))
    return pages


def main():
    sel = [json.loads(l) for l in open(os.path.join(DATA, 'selection.jsonl'))]
    pages = megalith_pages({r['flickr_photo_id'] for r in sel if r['part'] == 'megalith'})
    drop = {'not_downloaded': 0, 'unreadable': 0, 'too_small': 0, 'aspect': 0, 'no_megalith_page': 0}
    out = []
    p = Progress('hashing and checking downloaded images', len(sel))
    for r in sel:
        path = os.path.join('data', 'images', r['part'], f"{r['id']}.jpg")
        full = os.path.join(ROOT, path)
        p.add()
        if not os.path.exists(full):
            drop['not_downloaded'] += 1
            continue
        try:
            b = open(full, 'rb').read()
            w, h = Image.open(full).size
        except Exception:
            drop['unreadable'] += 1
            continue
        if min(w, h) < 512:
            drop['too_small'] += 1
            continue
        if max(w, h) > 3 * min(w, h):
            drop['aspect'] += 1
            continue
        m = {
            'id': r['id'],
            'part': r['part'],
            'split': r['split'],
            'path': path,
            'sha256': hashlib.sha256(b).hexdigest(),
            'width': w,
            'height': h,
            'monet_source': r['monet_source'],
            'monet_revision': r['monet_revision'],
            'license': r['license'],
        }
        if r['part'] == 'megalith':
            page = pages.get(r['flickr_photo_id'])
            if not page:
                drop['no_megalith_page'] += 1
                continue
            m.update(
                {
                    'flickr_photo_id': r['flickr_photo_id'],
                    'flickr_page': page,
                    'image_url': r['url'],
                    'megalith_revision': MEGALITH_REV,
                }
            )
        out.append(m)
    p.done()
    with open(os.path.join(DATA, 'manifest.jsonl'), 'w') as f:
        for m in out:
            f.write(json.dumps(m) + '\n')
    import collections

    stats = {
        'selected': len(sel),
        'kept': len(out),
        'dropped': drop,
        'kept_by_part': dict(collections.Counter(m['part'] for m in out)),
        'kept_by_split': dict(collections.Counter(m['split'] for m in out)),
    }
    json.dump(stats, open(os.path.join(DATA, 'manifest_stats.json'), 'w'), indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
