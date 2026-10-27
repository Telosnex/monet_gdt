# CONCLUDED 2026-10-25. Genre-targeted synthetic closes most of the
# Wallhaven gap; the 1/2 real, 1/4 generic, 1/4 genre mix is ADR D1.
# Results: results/.
#
# Genre-targeted MONET synthetic images: the dark / busy / game / anime look
# that Wallhaven has and megalith + generic synthetic lack.
#   <venv>/bin/python -u experiments/genre_ab/select.py
# Chosen by CLIP B/32 TEXT prompts only (no Wallhaven image is used as a
# query, so the Wallhaven test set stays fair). New shards (folders 2-0..4-0
# of each generator; synthetic_ab used 0-0 and 1-0). Same filters as
# synthetic_ab/select.py. Top PER_SHARD per shard by genre score.
# v2: dropped 'a stylized 3D render' (pulled interior renders); added
# negatives for interiors, product shots, floor plans, white backgrounds.
# Writes data/thumbs/<source>/<id>.webp and data/manifest.jsonl.
import hashlib
import json
import os
import sys
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
sys.path.insert(0, os.path.join(HERE, '..'))
from progress import Progress  # noqa: E402

ROOT = 'datasets/jasperai/monet/v1.2.0'
GENS = {'synthetic-flux-schnell': 'flux-schnell', 'synthetic-z-image': 'z-image', 'synthetic-flux-klein': 'flux-klein'}
SHARDS = [(src, f'synthetic/{g}/{k}-0/000000') for src, g in GENS.items() for k in (2, 3, 4)]
PER_SHARD = 350
POS = [
    'dark fantasy digital art',
    'video game concept art',
    'an anime illustration',
    'a cyberpunk neon city at night',
    'sci-fi space art',
    'a dark moody digital painting',
    'an epic fantasy battle scene',
    'a glowing neon synthwave scene',
]
NEG = [
    'a realistic landscape photo',
    'a photo of a flower',
    'a portrait photo of a person',
    'a snapshot of people',
    'a photo of an animal',
    'a photo of food',
    'a photo of a document or text',
    'a classical oil painting',
    'an interior design render of a room',
    'a product photo on a white background',
    'a floor plan',
    'a furniture catalog image',
    'an object isolated on a white background',
]
META = [
    'id',
    'source',
    'license',
    'width',
    'height',
    'aesthetic_laion-improved',
    'wk_jasperai',
    'wk_laion',
    'nsfw_falcon',
    'nsfw_jasperai',
    'embedding_clip-vit-base-patch32',
]


def main():
    import open_clip
    import torch

    m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tok = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        T = m.encode_text(tok(POS + NEG)).float()
        T = (T / T.norm(dim=-1, keepdim=True)).numpy()
    fs = HfFileSystem()
    os.makedirs(DATA, exist_ok=True)
    out = open(os.path.join(DATA, 'manifest.jsonl'), 'w')
    p = Progress('reading 9 MONET synthetic shards (CLIP+metadata ~25 MB, then thumbnails ~157 MB each)', len(SHARDS))
    for src, shard in SHARDS:
        os.makedirs(os.path.join(DATA, 'thumbs', src), exist_ok=True)
        path = f'{ROOT}/{shard}.parquet'
        t = pq.read_table(fs.open(path), columns=META).to_pydict()
        E = np.array(t['embedding_clip-vit-base-patch32'], np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        L = 100 * E @ T.T
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        score = P[:, : len(POS)].sum(1)
        ok = np.array(
            [
                (a or 0) >= 5.0
                and (w1 or 0) < 0.5
                and (w2 or 0) < 0.5
                and (n1 or 0) < 0.2
                and (n2 or 0) < 0.2
                and min(wd or 0, ht or 0) >= 512
                and max(wd, ht) / min(wd, ht) <= 3
                for a, w1, w2, n1, n2, wd, ht in zip(
                    t['aesthetic_laion-improved'],
                    t['wk_jasperai'],
                    t['wk_laion'],
                    t['nsfw_falcon'],
                    t['nsfw_jasperai'],
                    t['width'],
                    t['height'],
                )
            ]
        )
        idx = [i for i in np.argsort(-score) if ok[i]][:PER_SHARD]
        th = pq.read_table(fs.open(path), columns=['id', 'thumbnail']).to_pydict()
        pos = {k: i for i, k in enumerate(th['id'])}
        for i in idx:
            b = th['thumbnail'][pos[t['id'][i]]]
            if not b:
                continue
            open(os.path.join(DATA, 'thumbs', src, f"{t['id'][i]}.webp"), 'wb').write(b)
            out.write(
                json.dumps(
                    {
                        'id': t['id'][i],
                        'source': src,
                        'license': t['license'][i],
                        'shard': shard,
                        'clip_genre': round(float(score[i]), 4),
                        'sha256': hashlib.sha256(b).hexdigest(),
                    }
                )
                + '\n'
            )
        out.flush()
        p.add()
        print(
            f'  {shard}: {int(ok.sum())} pass filters, kept {len(idx)}, genre score {score[idx].min():.2f}..{score[idx].max():.2f}',
            flush=True,
        )
    p.done()
    out.close()


if __name__ == '__main__':
    main()
