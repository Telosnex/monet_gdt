# CONCLUDED 2026-10-25. Real + MONET synthetic beats real only on wallpaper
# test sets (mood); became ADR D1. Results: results/.
#
# Select wallpaper-like MONET images for the synthetic A/B.
#   <venv>/bin/python -u experiments/synthetic_ab/select.py
# For each source, reads metadata + CLIP columns of the first shards, keeps
# rows that pass the filters, ranks them by a CLIP text score, takes the top
# PER_SHARD of each shard, then reads thumbnails for those rows only.
# Writes data/thumbs/<source>/<id>.webp and data/manifest.jsonl.
import json
import os
import hashlib
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
ROOT = 'datasets/jasperai/monet/v1.2.0'
# source -> shards (first files of <n>-0 folders). Real: megalith (photos).
SOURCES = {
    'megalith10m': [
        'megalith10m/0-0/000000',
        'megalith10m/0-0/000001',
        'megalith10m/0-0/000002',
        'megalith10m/1-0/000000',
        'megalith10m/2-0/000000',
        'megalith10m/3-0/000000',
    ],
    'synthetic-flux-schnell': ['synthetic/flux-schnell/0-0/000000', 'synthetic/flux-schnell/1-0/000000'],
    'synthetic-z-image': ['synthetic/z-image/0-0/000000', 'synthetic/z-image/1-0/000000'],
    'synthetic-flux-klein': ['synthetic/flux-klein/0-0/000000', 'synthetic/flux-klein/1-0/000000'],
}
PER_SHARD = 500
POS = [
    'a desktop wallpaper',
    'a phone wallpaper',
    'a beautiful background image',
    'a scenic wallpaper',
    'an abstract wallpaper',
    'digital art wallpaper',
]
NEG = [
    'a snapshot of people',
    'a photo of a document or text',
    'a screenshot',
    'a product photo',
    'a close-up of food',
    'a photo of a crowd at an event',
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


def text_embeddings():
    import open_clip
    import torch

    m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tok = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        t = m.encode_text(tok(POS + NEG)).float()
    t = t / t.norm(dim=-1, keepdim=True)
    return t.numpy(), 100.0  # CLIP logit scale


def main():
    os.makedirs(DATA, exist_ok=True)
    T, scale = text_embeddings()
    fs = HfFileSystem()
    out = open(os.path.join(DATA, 'manifest.jsonl'), 'w')
    for source, shards in SOURCES.items():
        os.makedirs(os.path.join(DATA, 'thumbs', source), exist_ok=True)
        kept_total = 0
        for shard in shards:
            path = f'{ROOT}/{shard}.parquet'
            t = pq.read_table(fs.open(path), columns=META).to_pydict()
            E = np.array(t['embedding_clip-vit-base-patch32'], np.float32)
            E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
            logits = scale * E @ T.T
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            score = p[:, : len(POS)].sum(1)
            ok = np.array(
                [
                    (a or 0) >= 5.0
                    and (w1 or 0) < 0.5
                    and (w2 or 0) < 0.5
                    and (n1 or 0) < 0.2
                    and (n2 or 0) < 0.2
                    and min(wd or 0, ht or 0) >= 512
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
                f = os.path.join(DATA, 'thumbs', source, f"{t['id'][i]}.webp")
                open(f, 'wb').write(b)
                out.write(
                    json.dumps(
                        {
                            'id': t['id'][i],
                            'source': source,
                            'license': t['license'][i],
                            'shard': shard,
                            'clip_wallpaper': round(float(score[i]), 4),
                            'aesthetic': t['aesthetic_laion-improved'][i],
                            'width': t['width'][i],
                            'height': t['height'][i],
                            'sha256': hashlib.sha256(b).hexdigest(),
                        }
                    )
                    + '\n'
                )
                kept_total += 1
            print(
                f'{shard}: {len(t["id"])} rows, {int(ok.sum())} pass filters, kept {len(idx)}; '
                f'score of kept {score[idx].min():.2f}..{score[idx].max():.2f}',
                flush=True,
            )
        print(f'{source}: {kept_total}', flush=True)
    out.close()


if __name__ == '__main__':
    main()
