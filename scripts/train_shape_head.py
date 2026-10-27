# ADR D5 step 5: the shape head. Recipe = libmonet_gdt train_shapes.py (lib/shape_head.py) as
# shipped (TARGET=split: 87 outputs = echoes, domain clash, tone clash per
# shape; BCE + 0.1 x listwise rank loss on the combined score, tau 0.05;
# samples weighted 1/sqrt(frequency of Jev's winner); hidden 256, dropout
# 0.3, wd 1e-4; up to 200 epochs, best by val correlation; final fit on
# train + val at that epoch count). Input: the student's 192-value feature
# (out/student/student_features.npy, computed here from the cached 224 px
# images if missing). Labels: jev/shapes.jsonl (sheet o0). Split: manifest.
#   .venv/bin/python -u scripts/train_shape_head.py
# Out: out/shapes/shape_head.pt (NEW), report.json (val, test; Jev ceiling).
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import numpy as np
import torch
import timm  # noqa: E402
import shape_head as TS  # noqa: E402

OUT = os.path.join(ROOT, 'out', 'shapes')


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def graph_features(ids, base):
    """`feature` of the graph that will carry the head (as libmonet
    embed_ship.py: Pillow RGBA, long side <= 2048, Lanczos), cached per
    graph sha. onnx_shape_head.py requires features from that exact graph."""
    import hashlib
    import onnxruntime as ort
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    sha = hashlib.sha256(open(base, 'rb').read()).hexdigest()
    fp = os.path.join(ROOT, 'out', 'shapes', f'features_{sha[:12]}.npy')
    if os.path.exists(fp):
        return np.load(fp), sha
    man = {m['id']: m for m in map(json.loads, open(os.path.join(ROOT, 'data', 'manifest.jsonl')))}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = [ort.InferenceSession(base, so, providers=['CPUExecutionProvider']) for _ in range(10)]
    names = [o.name for o in sess[0].get_outputs()]
    fi = names.index('feature')

    def one(a):
        k, i = a
        im = Image.open(os.path.join(ROOT, man[i]['path'])).convert('RGBA')
        im.thumbnail((2048, 2048), Image.LANCZOS)
        return sess[k % 10].run(None, {'image': np.asarray(im)})[fi][0]

    F = np.zeros((len(ids), 192), np.float32)
    t0 = last = time.time()
    with ThreadPoolExecutor(10) as ex:
        for n, f in enumerate(ex.map(one, enumerate(ids))):
            F[n] = f
            if time.time() - last >= 10:
                last = time.time()
                log(
                    f'.. graph features {n + 1}/{len(ids)}, remaining ~{(last - t0) / (n + 1) * (len(ids) - n - 1):.0f}s'
                )
    np.save(fp, F)
    log(f'saved {fp}')
    return F, sha


def student_features(ids):
    fp = os.path.join(ROOT, 'out', 'student', 'student_features.npy')
    if os.path.exists(fp):
        return np.load(fp)
    d = torch.load(os.path.join(ROOT, 'out', 'student', 'images224.pt'))
    assert d['ids'] == ids
    U = d['u']
    sk = torch.load(os.path.join(ROOT, 'out', 'student', 'student.pt'), weights_only=False)
    m = timm.create_model(sk['model_name'], pretrained=False, num_classes=0, img_size=sk['res'], depth=sk['depth'])
    m.load_state_dict(sk['backbone'])
    m = m.to('mps').eval()
    out, t0, last = [], time.time(), time.time()
    with torch.no_grad():
        for i in range(0, len(U), 256):
            out.append(m(U[i : i + 256].float().div(127.5).sub(1).to('mps')).float().cpu().numpy())
            if time.time() - last >= 10:
                last = time.time()
                log(
                    f'.. student features {i + 256}/{len(U)}, remaining ~{(last - t0) / (i + 256) * (len(U) - i - 256):.0f}s'
                )
    F = np.concatenate(out)
    np.save(fp, F)
    log(f'saved {fp}')
    return F


def main():
    os.makedirs(OUT, exist_ok=True)
    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    lab = {r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'mood.jsonl'))) if 'error' not in r}
    ids_all = [m['id'] for m in man if m['id'] in lab]  # the student's row order (mood-labeled = all)
    BASE = os.environ.get('BASE')  # the graph that will carry the head; default: PyTorch student features
    if BASE:
        F, base_sha = graph_features(ids_all, BASE)
    else:
        F, base_sha = student_features(ids_all), None
    row = {i: k for k, i in enumerate(ids_all)}
    sh = {r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'shapes.jsonl'))) if 'error' not in r}
    shapes = sorted(next(iter(sh.values()))['parts'])
    M = [m for m in man if m['id'] in sh and m['id'] in row]
    X = np.stack([F[row[m['id']]] for m in M]).astype(np.float32)
    Y = np.array(
        [[sh[m['id']]['parts'][s][p] for p in ('echoes', 'domain', 'tone') for s in shapes] for m in M], np.float32
    )
    sp = np.array([m['split'] for m in M])
    tr, va, te = (np.where(sp == s)[0] for s in ('train', 'val', 'test'))
    log(f'{len(shapes)} shapes; train {len(tr)}, val {len(va)}, test {len(te)}')
    res = {'baseline_mean_val': TS.metrics(np.tile(Y[tr].mean(0), (len(va), 1)), Y[va])}
    log(f"baseline (training mean) val {res['baseline_mean_val']}")
    t0 = time.time()
    (m, mu, sd), ep = TS.fit_rank(X[tr], Y[tr], 256, 0.3, 1e-4, epochs=TS.MAX_EPOCHS, Xva=X[va], Yva=Y[va])
    res['val'] = {**TS.metrics(TS.predict((m, mu, sd), X[va]), Y[va]), 'epoch': ep}
    log(f"val {res['val']} ({time.time() - t0:.0f}s)")
    final, _ = TS.fit_rank(np.concatenate([X[tr], X[va]]), np.concatenate([Y[tr], Y[va]]), 256, 0.3, 1e-4, epochs=ep)
    res['test'] = TS.metrics(TS.predict(final, X[te]), Y[te])
    log(f"test vs Jev {res['test']}")
    rp = {
        r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'shapes_repeat.jsonl'))) if 'error' not in r
    }
    ti = [k for k in te if M[k]['id'] in rp]
    if ti:
        Yr = np.array(
            [[rp[M[k]['id']]['parts'][s][p] for p in ('echoes', 'domain', 'tone') for s in shapes] for k in ti],
            np.float32,
        )
        res['jev_repeat_o1_test'] = TS.metrics(Yr, Y[ti])
        log(f"Jev repeat (sheet o1) vs Jev, test {res['jev_repeat_o1_test']}")
    fm, fmu, fsd = final
    torch.save(
        {
            'state_dict': fm.state_dict(),
            'hidden': 256,
            'dropout': 0.3,
            'weight_decay': 1e-4,
            'epochs': ep,
            'feat_mean': fmu,
            'feat_std': fsd,
            'shapes': shapes,
            'embedding': f'ship_{base_sha[:12]}'
            if base_sha
            else 'monet_gdt student (out/student/student.pt), PyTorch, 224 center crop',
            'teacher': 'echoes_split_noul',
            'target': 'split',
            'tone_floor': TS.TONE_FLOOR,
            'rank': TS.RANK,
            'tau': TS.TAU,
            'winner_weighting': True,
        },
        os.path.join(OUT, 'shape_head.pt'),
    )
    json.dump(res, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)
    log('saved out/shapes/shape_head.pt; all done')


if __name__ == '__main__':
    main()
