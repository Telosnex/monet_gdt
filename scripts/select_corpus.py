# Step 2a of docs/ADR_train_monet_gdt.md: choose the corpus from MONET parquet
# columns only (no image bytes). Writes data/selection.jsonl.
#   <venv>/bin/python -u scripts/select_corpus.py
# D1 mix: MEGALITH_N real, GENERIC_N generic synthetic, GENRE_N genre
# synthetic. Per shard: D1 filters, then top-k by CLIP B/32 text score on
# MONET's stored CLIP embedding. Megalith also: caption keyword filter.
# Near-duplicates (same sscd_cluster_id) keep the best-scoring row.
# Split (D2): sha256(id) -> 90% train, 5% val, 5% test.
import hashlib
import json
import os
import re
import sys
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, HERE)
from progress import Progress  # noqa: E402

MONET_REV = 'a718d526ddcd9bd13681784d1fd02a2082d4e615'
MONET = f'datasets/jasperai/monet@{MONET_REV}/v1.2.0'
MEGALITH_N, GENERIC_N, GENRE_N = 12500, 6250, 6250
MEGALITH_SHARDS = [f'megalith10m/{k}-0/000000' for k in range(50)]
SYN_SHARDS = [
    (g, f'synthetic/{g}/{k}-0/000000') for g in ('flux-schnell', 'flux-klein', 'z-image') for k in range(5, 13)
]


_SEL = json.load(open(os.path.join(ROOT, 'prompts', 'selection.json')))
GENERIC = (_SEL['generic']['pos'], _SEL['generic']['neg'])
GENRE = (_SEL['genre']['pos'], _SEL['genre']['neg'])
CAPTIONS = [
    'caption_florence-2-large',
    'caption_gemini-2.5-flash-lite',
    'caption_internvl-3-8b',
    'caption_sharegpt4v-7b',
]
TEXT_WORDS = re.compile(
    r'\b(watermark\w*|logo\w*|signature\w*|signed|copyright\w*|©|text|texts|written|writing|lettering|'
    r'inscription\w*|caption\w*|words?|letters)\b',
    re.I,
)
META = [
    'id',
    'url',
    'width',
    'height',
    'aesthetic_laion-improved',
    'wk_jasperai',
    'wk_laion',
    'nsfw_falcon',
    'nsfw_jasperai',
    'nsfw_bumble',
    'sscd_cluster_id',
    'hash_sha256',
    'embedding_clip-vit-base-patch32',
]


def text_matrix(pos, neg):
    import open_clip
    import torch

    m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tok = open_clip.get_tokenizer('ViT-B-32')
    with torch.no_grad():
        T = m.encode_text(tok(pos + neg)).float()
    return (T / T.norm(dim=-1, keepdim=True)).numpy(), len(pos)


def score(E, TM):
    T, npos = TM
    L = 100 * E @ T.T
    P = np.exp(L - L.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return P[:, :npos].sum(1)


def passes(t, i):
    g = lambda k: t[k][i] or 0
    w, h = t['width'][i] or 0, t['height'][i] or 0
    return (
        g('aesthetic_laion-improved') >= 5
        and g('wk_jasperai') < 0.5
        and g('wk_laion') < 0.5
        and g('nsfw_falcon') < 0.2
        and g('nsfw_jasperai') < 0.2
        and g('nsfw_bumble') < 0.2
        and min(w, h) >= 512
        and max(w, h) <= 3 * min(w, h)
    )


def split(i):
    v = int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if v < 0.90 else 'val' if v < 0.95 else 'test'


def cached(shard, fn):
    # Per-shard results in data/selection_cache/ so a crash does not redo work.
    path = os.path.join(DATA, 'selection_cache', shard.replace('/', '_') + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    r = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(r, open(path, 'w'))
    return r


def main():
    fs = HfFileSystem()
    os.makedirs(DATA, exist_ok=True)
    TG, TS = text_matrix(*GENERIC), text_matrix(*GENRE)
    pick, stats = [], {'megalith_rows': 0, 'megalith_pass': 0, 'megalith_caption_drop': 0, 'syn_rows': 0, 'syn_pass': 0}
    per_mega = int(np.ceil(MEGALITH_N * 1.15 / len(MEGALITH_SHARDS)))  # 15% slack for dedupe + failed downloads
    p = Progress('step 2a: scoring MONET megalith shards (metadata, CLIP, captions; ~30 MB each)', len(MEGALITH_SHARDS))

    def mega(shard):
        rows, st = [], {'megalith_rows': 0, 'megalith_pass': 0, 'megalith_caption_drop': 0}
        f = fs.open(f'{MONET}/{shard}.parquet')
        t = pq.read_table(f, columns=META + CAPTIONS).to_pydict()
        E = np.array(t['embedding_clip-vit-base-patch32'], np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        s = score(E, TG)
        n = len(t['id'])
        st['megalith_rows'] += n
        ok = [i for i in range(n) if passes(t, i)]
        st['megalith_pass'] += len(ok)
        clean = [i for i in ok if not any(TEXT_WORDS.search(t[c][i] or '') for c in CAPTIONS)]
        st['megalith_caption_drop'] += len(ok) - len(clean)
        for i in sorted(clean, key=lambda i: -s[i])[:per_mega]:
            rows.append(
                {
                    'id': t['id'][i],
                    'part': 'megalith',
                    'monet_source': 'megalith10m',
                    'shard': shard,
                    'url': t['url'][i],
                    'flickr_photo_id': re.search(r'/(\d+)_', t['url'][i]).group(1),
                    'monet_sha256': t['hash_sha256'][i],
                    'sscd': t['sscd_cluster_id'][i],
                    'score': round(float(s[i]), 4),
                    'license': 'Megalith-10m Flickr license at crawl: CC0 / PDM / no known restrictions / US Gov (D9)',
                }
            )
        return {'rows': rows, 'stats': st}

    for shard in MEGALITH_SHARDS:
        r = cached(shard, lambda: mega(shard))
        pick += r['rows']
        for k, v in r['stats'].items():
            stats[k] += v
        p.add()
    p.done()
    per_syn = int(np.ceil((GENERIC_N + GENRE_N) * 1.1 / len(SYN_SHARDS) / 2))
    p = Progress('step 2a: scoring MONET synthetic shards (metadata, CLIP; ~25 MB each)', len(SYN_SHARDS))

    def syn(gen, shard):
        rows, st = [], {'syn_rows': 0, 'syn_pass': 0}
        t = pq.read_table(fs.open(f'{MONET}/{shard}.parquet'), columns=[c for c in META if c != 'url']).to_pydict()
        E = np.array(t['embedding_clip-vit-base-patch32'], np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        sg, ss = score(E, TG), score(E, TS)
        n = len(t['id'])
        st['syn_rows'] += n
        ok = [i for i in range(n) if passes(t, i)]
        st['syn_pass'] += len(ok)
        genre = sorted(ok, key=lambda i: -ss[i])[:per_syn]
        taken = set(genre)
        generic = [i for i in sorted(ok, key=lambda i: -sg[i]) if i not in taken][:per_syn]
        for part, idx, sc in (('genre', genre, ss), ('generic', generic, sg)):
            for i in idx:
                rows.append(
                    {
                        'id': t['id'][i],
                        'part': part,
                        'monet_source': f'synthetic-{gen}',
                        'shard': shard,
                        'monet_sha256': t['hash_sha256'][i],
                        'sscd': t['sscd_cluster_id'][i],
                        'score': round(float(sc[i]), 4),
                        'license': f'synthetic-{gen}: Apache-2.0 generator output (MONET card)',
                    }
                )
        return {'rows': rows, 'stats': st}

    for gen, shard in SYN_SHARDS:
        r = cached(shard, lambda: syn(gen, shard))
        pick += r['rows']
        for k, v in r['stats'].items():
            stats[k] += v
        p.add()
    p.done()
    best = {}
    for r in pick:  # near-duplicates: one per SSCD cluster
        k = r['sscd'] if r['sscd'] not in (None, -1) else r['id']
        if k not in best or r['score'] > best[k]['score']:
            best[k] = r
    out = list(best.values())
    stats['dedupe_dropped'] = len(pick) - len(out)
    with open(os.path.join(DATA, 'selection.jsonl'), 'w') as f:
        for r in out:
            f.write(json.dumps({**r, 'split': split(r['id']), 'monet_revision': MONET_REV}) + '\n')
    import collections

    stats['kept'] = dict(collections.Counter(r['part'] for r in out))
    stats['split'] = dict(collections.Counter(split(r['id']) for r in out))
    json.dump(stats, open(os.path.join(DATA, 'selection_stats.json'), 'w'), indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
