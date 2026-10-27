# CONCLUDED 2026-10-25. Museum CC0 art hurt Wallhaven and Unsplash in every
# mix; dropped (ADR D1). Results: results/.
#
# Select wallpaper-like CC0 museum images (Cleveland Museum of Art, National
# Gallery of Art) for the museum A/B.
#   <venv>/bin/python -u experiments/museum_ab/select.py
# Inputs (data/raw/): cma_data.csv (github ClevelandMuseumArt/openaccess,
# LFS), nga_objects.csv + nga_published_images.csv (github
# NationalGalleryOfArt/opendata). License per item: CMA share_license_status
# == "CC0"; NGA published_images.openaccess == "1" (primary view).
# Candidates: every Painting and Photograph, plus 3,000 Prints per museum
# (lowest sha256 of id). Thumbnails: long side 384 (as MONET). Score: CLIP
# ViT-B/32 (openai) image embedding vs the MONET select.py text prompts.
# Writes data/thumbs/<museum>/<id>.webp, data/candidates.jsonl (all, with
# score) and data/manifest.jsonl (top TOP per museum).
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from progress import Progress  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RAW = os.path.join(DATA, 'raw')
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location('syn_select', os.path.join(HERE, '..', 'synthetic_ab', 'select.py'))
_syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syn)
POS, NEG = _syn.POS, _syn.NEG  # the MONET selection prompts
TOP = 1500
PRINTS = 3000
UA = {'User-Agent': 'monet_gdt-research/0.1 (training-data selection; CC0 only)'}
csv.field_size_limit(10**9)
h = lambda s: hashlib.sha256(s.encode()).hexdigest()


def candidates():
    out = []
    cma = [
        r
        for r in csv.DictReader(open(f'{RAW}/cma_data.csv', encoding='utf-8-sig'))
        if r['share_license_status'] == 'CC0' and r['image_web']
    ]
    for kind in ('Painting', 'Photograph', 'Print'):
        rows = sorted((r for r in cma if r['type'] == kind), key=lambda r: h(r['id']))
        for r in rows[:PRINTS] if kind == 'Print' else rows:
            out.append(
                {
                    'id': f"cma-{r['id']}",
                    'museum': 'cma',
                    'type': kind,
                    'title': r['title'],
                    'license': 'CC0 (share_license_status)',
                    'page': r['url'],
                    'image': r['image_web'],
                }
            )
    O = {o['objectid']: o for o in csv.DictReader(open(f'{RAW}/nga_objects.csv', encoding='utf-8'))}
    nga = [
        i
        for i in csv.DictReader(open(f'{RAW}/nga_published_images.csv', encoding='utf-8'))
        if i['openaccess'] == '1' and i['viewtype'] == 'primary' and i['depictstmsobjectid'] in O
    ]
    for kind in ('Painting', 'Photograph', 'Print'):
        rows = sorted(
            (i for i in nga if O[i['depictstmsobjectid']]['classification'] == kind), key=lambda i: h(i['uuid'])
        )
        for i in rows[:PRINTS] if kind == 'Print' else rows:
            o = O[i['depictstmsobjectid']]
            out.append(
                {
                    'id': f"nga-{o['objectid']}",
                    'museum': 'nga',
                    'type': kind,
                    'title': o['title'],
                    'license': 'CC0 (openaccess=1)',
                    'page': f"https://www.nga.gov/collection/art-object-page.{o['objectid']}.html",
                    'image': f"{i['iiifurl']}/full/!384,384/0/default.jpg",
                }
            )
    return list({c['id']: c for c in out}.values())


def fetch(c):
    p = os.path.join(DATA, 'thumbs', c['museum'], f"{c['id']}.webp")
    if os.path.exists(p):
        return c, True
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(c['image'], headers=UA), timeout=60) as r:
                b = r.read()
            im = Image.open(io.BytesIO(b)).convert('RGB')
            im.thumbnail((384, 384), Image.LANCZOS)
            im.save(p, 'WEBP', quality=90)
            return c, True
        except Exception:
            time.sleep(2 * 2**attempt)
    return c, False


def main():
    C = candidates()
    for m in ('cma', 'nga'):
        os.makedirs(os.path.join(DATA, 'thumbs', m), exist_ok=True)
    print(f'{len(C)} candidates', {m: sum(c["museum"] == m for c in C) for m in ('cma', 'nga')}, flush=True)
    on_disk = lambda c: os.path.exists(os.path.join(DATA, 'thumbs', c['museum'], f"{c['id']}.webp"))
    ok = [c for c in C if on_disk(c)]
    todo = [c for c in C if not on_disk(c)]
    # Items already on disk are counted as done before the clock starts, so
    # the remaining-time estimate uses the download rate only.
    p = Progress(f'step 1/3: downloading CC0 museum thumbnails ({len(ok)} already on disk)', len(C), start_done=len(ok))
    with ThreadPoolExecutor(12) as ex:
        for c, good in ex.map(fetch, todo):
            p.add()
            if good:
                ok.append(c)
    p.done()
    print(f'{len(ok)} of {len(C)} thumbnails ok', flush=True)
    import open_clip
    import torch

    model, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval()
    tok = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        T = model.encode_text(tok(POS + NEG)).float()
        T /= T.norm(dim=-1, keepdim=True)
        p = Progress('step 2/3: CLIP-scoring thumbnails for wallpaper-likeness', len(ok))
        for k in range(0, len(ok), 256):
            batch = ok[k : k + 256]
            x = torch.stack(
                [
                    pre(Image.open(os.path.join(DATA, 'thumbs', c['museum'], f"{c['id']}.webp")).convert('RGB'))
                    for c in batch
                ]
            )
            E = model.encode_image(x).float()
            E /= E.norm(dim=-1, keepdim=True)
            sc = (100 * E @ T.T).softmax(-1)[:, : len(POS)].sum(-1).numpy()
            for c, s in zip(batch, sc):
                c['clip_wallpaper'] = round(float(s), 4)
            p.add(len(batch))
        p.done()
    print('step 3/3: writing candidates and manifest (seconds)', flush=True)
    with open(os.path.join(DATA, 'candidates.jsonl'), 'w') as f:
        for c in ok:
            f.write(json.dumps(c) + '\n')
    with open(os.path.join(DATA, 'manifest.jsonl'), 'w') as f:
        for m in ('cma', 'nga'):
            top = sorted((c for c in ok if c['museum'] == m), key=lambda c: -c['clip_wallpaper'])[:TOP]
            for c in top:
                f.write(json.dumps(c) + '\n')
            import collections

            print(
                f"{m}: {len(top)} kept, score {top[-1]['clip_wallpaper']:.2f}..{top[0]['clip_wallpaper']:.2f}, types {dict(collections.Counter(c['type'] for c in top))}",
                flush=True,
            )


if __name__ == '__main__':
    main()
