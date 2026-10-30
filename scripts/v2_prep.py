# Data for the color head (ADR D12): for every corpus image (manifest, all
# splits) and every matched image, in one pass with the base graph
# (out/graph/monet_gdt_w8_shapes.onnx):
#   feature  (192)       the graph's own `feature` output
#   ship     (3 RGB)     the base graph's `colors` output, a baseline for reports
#   list     codes/share the graph's color list (graph_front.front(), SINGLE TWO OPSET=20)
#   thumb96  RGBA        telosnex's thumbnail: integer box fold to 384, then 96
#                        (lib/extensions/rgba_box_average.dart), for the Dart v2 labeler
# The app decodes, then box-folds to 2048; no image here is larger, so the
# graph sees the original pixels. Matched images are stored at 448 px, so
# their thumb96 is 448 -> 384 -> 96 (the app would fold from the original).
#   .venv/bin/python -u scripts/v2_prep.py            (12 processes; resumable by shard)
# Out: out/v2/prep_<shard>.npz + out/v2/thumb96_<shard>.bin, out/v2/ids.json
import json
import os
import sys
import time

for k, v in (('SINGLE', '1'), ('TWO', '1'), ('OPSET', '20')):
    os.environ.setdefault(k, v)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
OUT = os.path.join(ROOT, 'out', 'v2')
GRAPH = os.path.join(ROOT, 'out', 'graph', 'monet_gdt_w8_shapes.onnx')
SHARD = 1000
PROCS = int(os.environ.get('PROCS', 12))


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def fit(w, h, m):
    if w <= m and h <= m:
        return w, h
    return (m, h * m // w) if w >= h else (w * m // h, m)


def fold(a, m):
    """telosnex rgba_box_average: integer box fold to fit m (opaque pixels only, round half up)."""
    import numpy as np

    h, w, _ = a.shape
    dw, dh = fit(w, h, m)
    if (dw, dh) == (w, h):
        return a
    col = np.minimum(np.arange(w) * dw // w, dw - 1)
    row = np.minimum(np.arange(h) * dh // h, dh - 1)
    cell = (row[:, None] * dw + col[None, :]).ravel()
    px = a.reshape(-1, 4).astype(np.int64)
    ok = px[:, 3] == 255
    cell, px = cell[ok], px[ok]
    n = np.bincount(cell, minlength=dw * dh)
    out = np.zeros((dw * dh, 4), np.uint8)
    nz = n > 0
    for c in range(4):
        s = np.bincount(cell, weights=px[:, c], minlength=dw * dh).astype(np.int64)
        out[nz, c] = ((s[nz] + (n[nz] >> 1)) // n[nz]).astype(np.uint8)
    return out.reshape(dh, dw, 4)


_S = {}


def sessions():
    if not _S:
        import onnxruntime as ort
        import graph_front as G

        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        _S['g'] = ort.InferenceSession(GRAPH, o, providers=['CPUExecutionProvider'])
        _S['f'] = ort.InferenceSession(G.front().SerializeToString(), o, providers=['CPUExecutionProvider'])
        _S['gout'] = [x.name for x in _S['g'].get_outputs()]
    return _S


def one(path):
    import numpy as np
    from PIL import Image

    s = sessions()
    try:
        a = np.asarray(Image.open(path).convert('RGBA'))
    except Exception:
        return None
    a = fold(a, 2048)
    r = dict(zip(s['gout'], s['g'].run(None, {'image': a})))
    codes, share = s['f'].run(['codes', 'share'], {'image': np.ascontiguousarray(a[..., :3])})
    keep = share > 0
    t = fold(fold(a, 384), 96)
    return (
        r['feature'].reshape(-1).astype(np.float32),
        r['colors'].astype(np.uint8),
        codes[keep].astype(np.int16),
        share[keep].astype(np.float32),
        t,
    )


def shard(args):
    import numpy as np

    k, items = args
    feats, ships, lens, codes, shares, ok = [], [], [], [], [], []
    with open(os.path.join(OUT, f'thumb96_{k:03d}.bin.tmp'), 'wb') as tb:
        for it in items:
            r = one(os.path.join(ROOT, it['path']))
            if r is None:
                r = (
                    np.zeros(192, np.float32),
                    np.zeros((3, 3), np.uint8),
                    np.zeros(1, np.int16),
                    np.ones(1, np.float32),
                    np.full((1, 1, 4), 255, np.uint8),
                )
                ok.append(False)
            else:
                ok.append(True)
            f, c, cd, sh, t = r
            feats.append(f)
            ships.append(c)
            lens.append(len(cd))
            codes.append(cd)
            shares.append(sh)
            tb.write(np.array([t.shape[1], t.shape[0]], '<u4').tobytes() + t.tobytes())
    np.savez(
        os.path.join(OUT, f'prep_{k:03d}.npz'),
        ids=np.array([it['id'] for it in items]),
        feature=np.stack(feats),
        ship=np.stack(ships),
        lens=np.array(lens, np.int32),
        codes=np.concatenate(codes),
        share=np.concatenate(shares).astype(np.float16),
        ok=np.array(ok),
    )
    os.replace(os.path.join(OUT, f'thumb96_{k:03d}.bin.tmp'), os.path.join(OUT, f'thumb96_{k:03d}.bin'))
    return k, len(items), sum(ok)


def items():
    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    mat = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'matched.jsonl'))]
    return [
        {'id': m['id'], 'path': m['path'], 'src': 'corpus', 'split': m['split'], 'part': m['part']} for m in man
    ] + [{'id': m['id'], 'path': m['path'], 'src': 'matched'} for m in mat]


def main():
    from multiprocessing import get_context

    os.makedirs(OUT, exist_ok=True)
    its = items()
    json.dump(its, open(os.path.join(OUT, 'ids.json'), 'w'))
    shards = [(k, its[i : i + SHARD]) for k, i in enumerate(range(0, len(its), SHARD))]
    todo = [s for s in shards if not os.path.exists(os.path.join(OUT, f'prep_{s[0]:03d}.npz'))]
    if len(sys.argv) > 1:
        todo = todo[: int(sys.argv[1])]
    log(
        f'{len(its)} images, {len(shards)} shards, {len(todo)} to do, {PROCS} processes; graph {os.path.basename(GRAPH)}'
    )
    t0 = time.time()
    done = 0
    with get_context('spawn').Pool(PROCS) as pool:
        for k, n, good in pool.imap_unordered(shard, todo):
            done += n
            el = time.time() - t0
            log(
                f'shard {k}: {good}/{n} ok; {done} images in {el:.0f}s, ~{el / done * (sum(len(s[1]) for s in todo) - done) / 60:.1f} min left'
            )
    log('all done')


if __name__ == '__main__':
    main()
