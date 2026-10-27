# Wallpaper-matched ("Unsplash-like") Megalith images for the student (ADR D11).
# The student copies the teacher and needs no Jev labels, so it can train
# on more images. Pick CC0 Megalith photos that look like everyday
# wallpapers: for each reference image (Unsplash Lite), its nearest
# Megalith images by CLIP B/32 cosine, using the embedding MONET stores per
# row, so no image is downloaded until it is chosen. The reference images
# only choose; they are not trained on, and under the Unsplash Dataset
# Terms (internal use only) they and their embeddings are not shared.
#   <venv>/bin/python -u scripts/match_unsplash.py parity | ref | scan | select | download
# parity: local open_clip ViT-B-32 (openai) on our downloaded Megalith
#   images vs MONET's stored embedding (must agree, or matching is wrong).
# ref:    Unsplash Lite photos -> out/match/ref_clip.npy (see ref()).
# scan:   Megalith shards (D1 filters + caption filter, not already in the
#   corpus) -> data/match_cache/<shard>.npz (ids, fp16 embeddings, meta).
# select: per reference image, nearest pool images, merged by rank until
#   N (50,000); one per SSCD cluster; if out/match/d8_clip.npy exists,
#   drops any image within cosine 0.95 of a D8 test image ->
#   data/selection_matched.jsonl (part 'matched', split 'train').
# download: matched images from Flickr (see download()).
import json
import os
import re
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'out', 'match')
sys.path.insert(0, HERE)
import select_corpus as SC  # noqa: E402  (MONET path, filters, caption filter)

N = int(os.environ.get('N', 50000))
LEAK = 0.95


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def shard_list():
    """Every Megalith parquet in MONET (90 folders, 811 files), listed once
    and cached in data/match_shards.json."""
    p = os.path.join(DATA, 'match_shards.json')
    if os.path.exists(p):
        return json.load(open(p))
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f'{SC.MONET}/megalith10m'
    out = sorted(
        f[len(SC.MONET) + 1 : -len('.parquet')]
        for d in fs.ls(base, detail=False)
        for f in fs.ls(d, detail=False)
        if f.endswith('.parquet')
    )
    json.dump(out, open(p, 'w'))
    return out


def clip_model():
    import open_clip

    m, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    return m.eval().to('mps'), pre


class Files:  # module level: macOS DataLoader workers pickle it
    def __init__(s, paths, pre):
        s.paths, s.pre = paths, pre

    def __len__(s):
        return len(s.paths)

    def __getitem__(s, i):
        from PIL import Image

        return s.pre(Image.open(s.paths[i]).convert('RGB'))


def embed(paths, label):
    import torch

    m, pre = clip_model()
    dl = torch.utils.data.DataLoader(Files(paths, pre), batch_size=64, num_workers=8)
    out, n, t0, last = [], 0, time.time(), time.time()
    with torch.no_grad():
        for b in dl:
            e = m.encode_image(b.to('mps')).float()
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            n += len(b)
            if time.time() - last >= 10:
                last = time.time()
                log(f'.. embedding {label}: {n}/{len(paths)}, remaining ~{(last - t0) / n * (len(paths) - n):.0f}s')
    log(f'embedded {len(paths)} {label} in {time.time() - t0:.0f}s')
    return np.concatenate(out)


def parity():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    sel = [json.loads(l) for l in open(os.path.join(DATA, 'selection.jsonl'))]
    shard = 'megalith10m/0-0/000000'
    rows = [r for r in sel if r.get('shard') == shard and r['part'] == 'megalith']
    have = {r['id']: r for r in map(json.loads, open(os.path.join(DATA, 'manifest.jsonl'))) if r['part'] == 'megalith'}
    t = pq.read_table(
        HfFileSystem().open(f'{SC.MONET}/{shard}.parquet'), columns=['id', 'embedding_clip-vit-base-patch32']
    ).to_pydict()
    S = {i: np.array(e, np.float32) for i, e in zip(t['id'], t['embedding_clip-vit-base-patch32'])}
    ids = [
        r['id']
        for r in rows
        if r['id'] in S and any(m.get('monet_id', m['id']) == r['id'] for m in [have.get(r['id'], {})])
    ]
    if not ids:  # manifest ids may differ from MONET ids: match through the sidecar JSON
        side = {}
        for f in os.listdir(os.path.join(DATA, 'images', 'megalith')):
            if f.endswith('.json'):
                side[json.load(open(os.path.join(DATA, 'images', 'megalith', f))).get('id')] = f[:-5]
        pairs = [(r['id'], side[r['id']]) for r in rows if r['id'] in S and r['id'] in side]
    else:
        pairs = [(i, i) for i in ids]
    log(f'{len(pairs)} local images from {shard}')
    E = embed([os.path.join(DATA, 'images', 'megalith', f'{loc}.jpg') for _, loc in pairs], 'parity')
    St = np.stack([S[i] for i, _ in pairs])
    St /= np.linalg.norm(St, axis=1, keepdims=True)
    cos = (E * St).sum(1)
    M = E @ St.T
    top1 = float(np.mean(M.argmax(1) == np.arange(len(pairs))))
    log(
        f'local vs MONET stored CLIP: cosine median {np.median(cos):.4f}, min {cos.min():.4f}; nearest stored = own row {top1:.0%}'
    )


def ref():
    """Unsplash Lite reference (ADR D11). Metadata: Hugging Face
    1aurent/unsplash-lite (`photo` column). Images: images.unsplash.com,
    long side 1024 (their CDN, not the rate-limited API), cached in
    data/unsplash_lite/ (not redistributed: Unsplash Dataset Terms 2A, 3A-B).
    Writes out/match/ref_clip.npy + ref_names.json. N_REF=<k> limits the
    photos (fixed shuffle, seed 0). Resumable; backs off on 429/5xx."""
    import random
    import urllib.request
    import urllib.error
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    d = os.path.join(DATA, 'unsplash_lite')
    os.makedirs(d, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    mp = os.path.join(d, 'photos.json')
    if os.path.exists(mp):
        photos = json.load(open(mp))
    else:
        fs = HfFileSystem()
        photos = []
        for i in range(3):
            t = (
                pq.read_table(
                    fs.open(f'datasets/1aurent/unsplash-lite/data/train-0000{i}-of-00003.parquet'), columns=['photo']
                )
                .column('photo')
                .to_pylist()
            )
            photos += [{'id': x['id'], 'image_url': x['image_url']} for x in t]
        photos.sort(key=lambda x: x['id'])
        random.Random(0).shuffle(photos)
        json.dump(photos, open(mp, 'w'))
    photos = photos[: int(os.environ.get('N_REF', len(photos)))]

    def one(x):
        path = os.path.join(d, x['id'] + '.jpg')
        if os.path.exists(path):
            return True
        for k in range(5):
            try:
                req = urllib.request.Request(
                    x['image_url'] + '?w=1024&h=1024&fit=max&fm=jpg&q=85',
                    headers={'User-Agent': 'monet_gdt/1.0 (dataset: unsplash-lite)'},
                )
                b = urllib.request.urlopen(req, timeout=30).read()
                open(path + '.part', 'wb').write(b)
                os.replace(path + '.part', path)
                return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                time.sleep(5 * (k + 1))
            except Exception:
                time.sleep(2 * (k + 1))
        return False

    t0 = last = time.time()
    ok = []
    with ThreadPoolExecutor(16) as ex:
        for n, (x, got) in enumerate(zip(photos, ex.map(one, photos))):
            if got:
                ok.append(x['id'])
            if time.time() - last >= 10:
                last = time.time()
                log(
                    f'.. unsplash lite {n + 1}/{len(photos)}, remaining ~{(last - t0) / (n + 1) * (len(photos) - n - 1):.0f}s'
                )
    log(f'unsplash lite: {len(ok)}/{len(photos)} images')
    np.save(
        os.path.join(OUT, 'ref_clip.npy'),
        embed([os.path.join(d, i + '.jpg') for i in ok], 'unsplash lite').astype(np.float16),
    )
    json.dump(ok, open(os.path.join(OUT, 'ref_names.json'), 'w'))


def scan():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from concurrent.futures import ThreadPoolExecutor

    fs = HfFileSystem()
    os.makedirs(os.path.join(DATA, 'match_cache'), exist_ok=True)
    have = {json.loads(l)['id'] for l in open(os.path.join(DATA, 'selection.jsonl'))}
    shards = shard_list()

    def one(shard):
        p = os.path.join(DATA, 'match_cache', shard.replace('/', '_') + '.npz')
        if os.path.exists(p):
            return p, 0
        t = pq.read_table(fs.open(f'{SC.MONET}/{shard}.parquet'), columns=SC.META + SC.CAPTIONS).to_pydict()
        ok = [
            i
            for i in range(len(t['id']))
            if SC.passes(t, i)
            and t['id'][i] not in have
            and not any(SC.TEXT_WORDS.search(t[c][i] or '') for c in SC.CAPTIONS)
        ]
        E = np.array([t['embedding_clip-vit-base-patch32'][i] for i in ok], np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        meta = [
            {'id': t['id'][i], 'url': t['url'][i], 'monet_sha256': t['hash_sha256'][i], 'sscd': t['sscd_cluster_id'][i]}
            for i in ok
        ]
        np.savez(p, emb=E.astype(np.float16), meta=json.dumps(meta))
        return p, len(t['id'])

    t0 = last = time.time()
    done = rows = 0

    def safe(shard):
        for k in range(3):
            try:
                return one(shard)
            except Exception as e:
                if k == 2:
                    log(f'!! {shard}: {type(e).__name__}: {e}; skipped')
                    return None, 0
                time.sleep(5)

    with ThreadPoolExecutor(8) as ex:
        for p, n in ex.map(safe, shards):
            done += 1
            rows += n
            if time.time() - last >= 10 or done == len(shards):
                last = time.time()
                log(
                    f'.. scanned {done}/{len(shards)} shards ({rows} new rows read), elapsed {last - t0:.0f}s, remaining ~{(last - t0) / done * (len(shards) - done):.0f}s'
                )
    ok = [s for s in shards if os.path.exists(os.path.join(DATA, 'match_cache', s.replace('/', '_') + '.npz'))]
    log(f'scan done: {len(ok)}/{len(shards)} shards cached')


def select():
    import torch

    shards = [
        s for s in shard_list() if os.path.exists(os.path.join(DATA, 'match_cache', s.replace('/', '_') + '.npz'))
    ]
    E, meta = [], []
    for s in shards:
        z = np.load(os.path.join(DATA, 'match_cache', s.replace('/', '_') + '.npz'))
        E.append(z['emb'])
        meta += [{**m, 'shard': s} for m in json.loads(str(z['meta']))]
    P = torch.tensor(np.concatenate(E), dtype=torch.float16, device='mps')
    Rf = torch.tensor(np.load(os.path.join(OUT, 'ref_clip.npy')), device='mps')
    d8p = os.path.join(OUT, 'd8_clip.npy')  # optional (legacy D8 test images): exclude near-copies
    D8 = torch.tensor(np.load(d8p), device='mps') if os.path.exists(d8p) else None
    log(f'pool {len(meta)} images, reference {len(Rf)}')
    K = 32
    rank = torch.empty((len(Rf), K), dtype=torch.long)
    sims = torch.empty((len(Rf), K))
    for i in range(0, len(Rf), 512):
        v, ix = (Rf[i : i + 512] @ P.T).float().topk(K, dim=1)
        rank[i : i + 512] = ix.cpu()
        sims[i : i + 512] = v.cpu()
    leak = torch.zeros(len(meta), dtype=torch.bool)
    for i in range(0, len(meta), 65536):
        if D8 is not None:
            leak[i : i + 65536] = ((P[i : i + 65536] @ D8.T).max(1).values.float() > LEAK).cpu()
    log(f'{int(leak.sum())} pool images with cosine > {LEAK} to a D8 image: excluded')
    chosen, seen, clusters = [], set(), set()
    for r in range(K):  # round-robin: every reference's 1st match, then 2nd, ...
        for q in range(len(Rf)):
            j = int(rank[q, r])
            if j in seen or leak[j]:
                continue
            seen.add(j)
            c = meta[j]['sscd']
            if c not in (None, -1) and c in clusters:
                continue
            if c not in (None, -1):
                clusters.add(c)
            chosen.append((j, float(sims[q, r]), r))
            if len(chosen) >= N:
                break
        if len(chosen) >= N:
            break
    log(
        f'chose {len(chosen)} (deepest rank used {chosen[-1][2] + 1}); cosine to its reference median {np.median([c[1] for c in chosen]):.3f}'
    )
    # how wallpaper-like: nearest-reference cosine, chosen vs the current Megalith corpus
    near = lambda X: (X @ Rf.T).float().max(1).values.cpu().numpy()
    with open(os.path.join(DATA, 'selection_matched.jsonl'), 'w') as f:
        for j, s, r in chosen:
            m = meta[j]
            f.write(
                json.dumps(
                    {
                        'id': m['id'],
                        'part': 'matched',
                        'monet_source': 'megalith10m',
                        'shard': m['shard'],
                        'url': m['url'],
                        'flickr_photo_id': (re.search(r'/(\d+)_', m['url']) or [None, None])[1],
                        'monet_sha256': m['monet_sha256'],
                        'sscd': m['sscd'],
                        'match_cos': round(s, 4),
                        'match_rank': r,
                        'split': 'train',
                        'monet_revision': SC.MONET_REV,
                        'license': 'Megalith-10m Flickr license at crawl: CC0 / PDM / no known restrictions / US Gov (D9)',
                    }
                )
                + '\n'
            )
    nc = near(P[torch.tensor([c[0] for c in chosen], device='mps')])
    pool_rand = near(P[torch.randperm(len(meta), device='mps')[:20000]])
    log(
        f'nearest-reference cosine: chosen median {np.median(nc):.3f}; random pool {np.median(pool_rand):.3f}; wrote data/selection_matched.jsonl'
    )


def download():
    """Matched images from their Flickr URLs (MONET tars are 4.1 GB each and
    the matches spread over all 811; Flickr serves the same photo, 99% still
    up). Stored at 448 px short side, JPEG q92 (the student reads 384;
    ~60 KB an image), with the source row as a sidecar. Resumable; failures
    (404 etc.) recorded in data/matched_failed.jsonl. Writes
    data/matched.jsonl (rows with 'path') for train_student.py EXTRA=."""
    import hashlib
    import io
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    rows = [json.loads(l) for l in open(os.path.join(DATA, 'selection_matched.jsonl'))]
    d = os.path.join(DATA, 'images', 'matched')
    os.makedirs(d, exist_ok=True)
    failed_p = os.path.join(DATA, 'matched_failed.jsonl')
    failed = {json.loads(l)['id'] for l in open(failed_p)} if os.path.exists(failed_p) else set()
    todo = [r for r in rows if r['id'] not in failed and not os.path.exists(os.path.join(d, f"{r['id']}.json"))]
    log(f'{len(rows)} matched rows; {len(rows) - len(todo)} done or failed before; {len(todo)} to fetch')

    def one(r):
        for k in range(3):
            try:
                b = urllib.request.urlopen(
                    urllib.request.Request(r['url'], headers={'User-Agent': 'curl/8.7.1'}), timeout=30
                ).read()
                im = Image.open(io.BytesIO(b)).convert('RGB')
                w, h = im.size
                f = 448 / min(w, h)
                if f < 1:
                    im = im.resize((round(w * f), round(h * f)), Image.LANCZOS)
                im.save(os.path.join(d, f"{r['id']}.jpg"), quality=92)
                json.dump(
                    {
                        **r,
                        'source_sha256': hashlib.sha256(b).hexdigest(),
                        'source_size': [w, h],
                        'stored': 'short side 448, JPEG q92',
                    },
                    open(os.path.join(d, f"{r['id']}.json"), 'w'),
                )
                return None
            except urllib.error.HTTPError as e:
                if e.code in (404, 410):
                    return {'id': r['id'], 'error': f'HTTP {e.code}'}
                err = f'HTTP {e.code}'
            except Exception as e:
                err = f'{type(e).__name__}: {str(e)[:60]}'
            time.sleep(2 * (k + 1))
        return {'id': r['id'], 'error': err}

    t0 = last = time.time()
    nf = 0
    with ThreadPoolExecutor(16) as ex, open(failed_p, 'a') as ff:
        for n, res in enumerate(ex.map(one, todo)):
            if res:
                ff.write(json.dumps(res) + '\n')
                nf += 1
            if time.time() - last >= 10:
                last = time.time()
                log(
                    f'.. {n + 1}/{len(todo)} ({nf} failed), {(n + 1) / (last - t0):.0f}/s, remaining ~{(last - t0) / (n + 1) * (len(todo) - n - 1) / 60:.0f}m'
                )
    have = [r for r in rows if os.path.exists(os.path.join(d, f"{r['id']}.json"))]
    with open(os.path.join(DATA, 'matched.jsonl'), 'w') as f:
        for r in have:
            f.write(json.dumps({'id': r['id'], 'path': f"data/images/matched/{r['id']}.jpg"}) + '\n')
    log(f'downloaded {len(have)}/{len(rows)}; wrote data/matched.jsonl')


if __name__ == '__main__':
    {'parity': parity, 'ref': ref, 'scan': scan, 'select': select, 'download': download}[sys.argv[1]]()
