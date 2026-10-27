# ADR D5 step 1: the mood teacher. Recipe = libmonet_gdt
# wallpaper_student/mood_unfreeze.py as used for final_k4.pt (K 4,
# backbone lr 3e-5, head lr 1e-3, batch 64, up to 20 epochs, patience 5),
# with the fixed manifest split (D2): train, early stop on val, report test.
#   .venv/bin/python -u scripts/train_teacher.py
# Inputs: data/manifest.jsonl, jev/mood.jsonl, fonts/mood_font_model.json
#   (font vectors + wallpaper population, for the font-pick score only).
# Outputs (out/teacher/): cache_cut8.pt (tokens after the frozen 8 blocks,
#   fp16, rebuildable), teacher.pt (backbone + head + feature norm +
#   quantile-map data), report.json (test r per word and by part, top 5%,
#   median rank; frozen baseline alongside). Logs every 10 s.
import json
import os
import time
import numpy as np
import torch
import timm
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, 'out', 'teacher')
# N=<k>: learning curve. Train on the first k train images
# in sha256(id) order (nested: smaller sets are inside larger ones); same
# val, test and recipe. Writes out/curves/teacher_<k>/ (refuses to overwrite).
NSUB = int(os.environ.get('N', 0))
OUT = os.path.join(ROOT, 'out', 'curves', f'teacher_{NSUB}') if NSUB else CACHE_DIR
NAME = 'vit_pe_core_tiny_patch16_384.fb'
K = 4
BLR = 3e-5
HLR = 1e-3
BS = 64
EPOCHS = int(os.environ.get('EPOCHS', 20))
PAT = int(os.environ.get('PATIENCE', 5))
DEV = 'mps'
# Stop rule: an epoch counts as a gain only if val MSE beats the best so far
# (starting from the frozen baseline) by MIN_GAIN (relative); PAT epochs in a
# row without a gain stop training. The saved model is the best epoch.
MIN_GAIN = float(os.environ.get('MIN_GAIN', 0.005))
WORDS = [
    'artistic',
    'attention-grabbing',
    'calm',
    'complex',
    'disorderly',
    'dramatic',
    'formal',
    'playful',
    'strong',
    'wide',
]


class Clock:
    def __init__(s):
        s.t = s.last = time.time()

    def due(s):
        if time.time() - s.last >= 10:
            s.last = time.time()
            return True
        return False

    def el(s):
        return time.time() - s.t


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def qmap_fit(Ptr, Ytr):  # mood_pe_tiny.qmap_fit
    S = np.sort(Ptr, 0)

    def f(P):
        Q = np.empty_like(P)
        for w in range(P.shape[1]):
            Q[:, w] = np.quantile(Ytr[:, w], np.clip(np.searchsorted(S[:, w], P[:, w]) / len(S), 0, 1))
        return Q

    return f


FM = json.load(open(os.path.join(ROOT, 'fonts', 'mood_font_model.json')))
FV = np.array(FM['font_vectors'])


def pct(y):
    return np.array(
        [
            (
                np.sum(np.array(FM['wall_population'][w]) < y[k])
                + 0.5 * np.sum(np.array(FM['wall_population'][w]) == y[k])
                + 0.5
            )
            / (len(FM['wall_population'][w]) + 1)
            for k, w in enumerate(WORDS)
        ]
    )


def font_scores(y):
    return -np.mean((FV - pct(y)) ** 2, axis=1)  # vl_mood.scores


class Images(torch.utils.data.Dataset):  # module level: macOS workers pickle it
    def __init__(s, paths, tf):
        s.paths, s.tf = paths, tf

    def __len__(s):
        return len(s.paths)

    def __getitem__(s, i):
        return s.tf(Image.open(s.paths[i]).convert('RGB'))


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    lab = {
        r['id']: r['raw'] for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'mood.jsonl'))) if 'error' not in r
    }
    R = [m for m in man if m['id'] in lab]
    Y = np.array([[lab[m['id']][w] for w in WORDS] for m in R], np.float32)
    idx = {s: np.array([i for i, m in enumerate(R) if m['split'] == s]) for s in ('train', 'val', 'test')}
    tr, va, te = idx['train'], idx['val'], idx['test']
    if NSUB:
        import hashlib

        assert not os.path.exists(os.path.join(OUT, 'teacher.pt')), f'{OUT} exists; use a new N'
        tr = np.array(sorted(tr, key=lambda i: hashlib.sha256(R[i]['id'].encode()).hexdigest())[:NSUB])
        tr.sort()
        log(f'learning curve: training on {len(tr)} of the train split (sha256 order)')
    log(f'{len(R)} labeled images: train {len(tr)}, val {len(va)}, test {len(te)}')
    m = timm.create_model(NAME, pretrained=True, num_classes=0).to(DEV).eval()
    tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
    nb = len(m.blocks)
    cut = nb - K
    mixed = getattr(m, 'rope_mixed', False)

    def stem(img):
        x = m.patch_embed(img)
        x, rope = m._pos_embed(x)
        return m.norm_pre(x), rope

    def run_blocks(x, rope, lo, hi):
        for i in range(lo, hi):
            x = m.blocks[i](x, rope=rope[i] if mixed else rope)
        return x

    def tail(x, rope):
        return m.forward_head(m.norm(run_blocks(x, rope, cut, nb)), pre_logits=True)

    with torch.no_grad():
        _, ROPE = stem(torch.zeros(1, 3, 384, 384, device=DEV))

    # 1. cache tokens after the frozen blocks
    cp = os.path.join(CACHE_DIR, f'cache_cut{cut}.pt')
    ids = [m_['id'] for m_ in R]
    cache = None
    if os.path.exists(cp):
        d = torch.load(cp)
        if d['ids'] == ids:
            cache = d['x']
            log(f'loaded {cp} ({cache.numel() * 2 / 1e9:.1f} GB)')
    if cache is None:
        dl = torch.utils.data.DataLoader(
            Images([os.path.join(ROOT, x['path']) for x in R], tf), batch_size=32, num_workers=8
        )
        c = Clock()
        k = 0
        log(f'caching tokens after block {cut} of {nb} for {len(R)} images (8 decode workers)')
        with torch.no_grad():
            for b in dl:
                x, rope = stem(b.to(DEV))
                z = run_blocks(x, rope, 0, cut).half().cpu()
                if cache is None:
                    cache = torch.empty((len(R), *z.shape[1:]), dtype=torch.float16)
                    log(f'cache {tuple(cache.shape)} fp16, {cache.numel() * 2 / 1e9:.1f} GB')
                cache[k : k + len(z)] = z
                k += len(z)
                if c.due() or k == len(R):
                    log(
                        f'.. caching: {k}/{len(R)} ({k / len(R):.0%}), elapsed {c.el():.0f}s, remaining ~{c.el() / k * (len(R) - k):.0f}s'
                    )
        torch.save({'ids': ids, 'x': cache}, cp)
        log(f'saved {cp}')

    def inp(ix):
        return cache[torch.as_tensor(ix)].to(DEV).float(), ROPE

    def feats(ix, label):
        out = []
        c = Clock()
        with torch.no_grad():
            for i in range(0, len(ix), 128):
                out.append(tail(*inp(ix[i : i + 128])).float().cpu())
                if c.due():
                    log(
                        f'.. {label} features: {min(i + 128, len(ix))}/{len(ix)}, remaining ~{c.el() / (i + 128) * (len(ix) - i - 128):.0f}s'
                    )
        return torch.cat(out).numpy()

    JS = {t: font_scores(Y[t]) for t in te}

    def score(P_raw_tr, P_raw_te, label):
        P = qmap_fit(P_raw_tr, Y[tr])(P_raw_te)
        r = np.array([np.corrcoef(P[:, k], Y[te, k])[0, 1] for k in range(len(WORDS))])
        rk = np.array([np.mean(JS[t] > JS[t][int(np.argmax(font_scores(P[i])))]) for i, t in enumerate(te)])
        parts = {
            p: float(
                np.mean(
                    [
                        np.corrcoef(
                            P[[i for i, t in enumerate(te) if R[t]['part'] == p], k],
                            Y[[t for t in te if R[t]['part'] == p], k],
                        )[0, 1]
                        for k in range(len(WORDS))
                    ]
                )
            )
            for p in ('megalith', 'generic', 'genre')
        }
        row = {
            'r': float(r.mean()),
            'per_word': dict(zip(WORDS, np.round(r, 3).tolist())),
            'r_by_part': parts,
            'top5pct': float(np.mean(rk <= 0.05)),
            'rank_median': float(np.median(rk)),
        }
        log(
            f'{label}: test r {row["r"]:.3f}, top5% {row["top5pct"]:.2f}, rank median {row["rank_median"]:.3f}, '
            f'by part ' + ', '.join(f'{p} {v:.3f}' for p, v in parts.items())
        )
        return row

    # 2. frozen baseline head
    log('frozen baseline: features')
    Ftr, Fva, Fte = feats(tr, 'train'), feats(va, 'val'), feats(te, 'test')
    mu, sd = torch.tensor(Ftr.mean(0), device=DEV), torch.tensor(Ftr.std(0) + 1e-6, device=DEV)
    head = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(192, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 10),
        torch.nn.Sigmoid(),
    ).to(DEV)
    H = lambda x: head((x - mu) / sd)
    T = lambda a: torch.tensor(a, device=DEV)
    opt = torch.optim.AdamW(head.parameters(), HLR, weight_decay=1e-4)
    best = (1e9, 0, None)
    bad = 0
    c = Clock()
    for ep in range(1, 3001):
        head.train()
        opt.zero_grad()
        torch.nn.functional.mse_loss(H(T(Ftr)), T(Y[tr])).backward()
        opt.step()
        if ep % 25 == 0:
            head.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(H(T(Fva)), T(Y[va])).item()
            if v < best[0]:
                best = (v, ep, {k: t.clone() for k, t in head.state_dict().items()})
                bad = 0
            else:
                bad += 1
            if c.due():
                log(f'.. head pass {ep}, best val MSE {best[0]:.5f}')
            if bad >= 12:
                break
    head.load_state_dict(best[2])
    head.eval()
    base_val = best[0]
    ph = lambda F: H(T(F)).detach().cpu().numpy()
    report = {
        'recipe': {
            'model': NAME,
            'K': K,
            'backbone_lr': BLR,
            'head_lr': HLR,
            'batch': BS,
            'epochs': EPOCHS,
            'patience': PAT,
            'min_gain': MIN_GAIN,
        },
        'n': {'train': len(tr), 'val': len(va), 'test': len(te)},
        'baseline': {**score(ph(Ftr), ph(Fte), 'frozen baseline'), 'val_mse': base_val},
    }

    # 3. fine-tune last K blocks + norms + pool + head
    for p in m.parameters():
        p.requires_grad = False
    mods = [m.blocks[i] for i in range(cut, nb)] + [
        x for x in (m.norm, m.attn_pool, m.fc_norm) if isinstance(x, torch.nn.Module)
    ]
    bb = [p for mod in mods for p in mod.parameters()]
    for p in bb:
        p.requires_grad = True
    opt = torch.optim.AdamW([{'params': bb, 'lr': BLR}, {'params': head.parameters(), 'lr': HLR}], weight_decay=1e-4)
    nbatch = (len(tr) + BS - 1) // BS
    total = EPOCHS * nbatch
    best = (base_val, 0, None)
    bad = 0
    c = Clock()
    done = 0
    hist = []
    log(
        f'fine-tuning {sum(p.numel() for p in bb) / 1e6:.2f}M backbone params + head: up to {EPOCHS} epochs x {nbatch} batches'
    )
    for ep in range(1, EPOCHS + 1):
        m.train()
        head.train()
        order = np.random.default_rng(ep).permutation(tr)
        ls = 0
        for b in range(nbatch):
            ix = order[b * BS : (b + 1) * BS]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(H(tail(*inp(ix))), T(Y[ix]))
            loss.backward()
            opt.step()
            ls += loss.item()
            done += 1
            if c.due():
                log(
                    f'.. epoch {ep}/{EPOCHS} batch {b + 1}/{nbatch}, train MSE {ls / (b + 1):.5f}; elapsed {c.el() / 60:.1f}m, '
                    f'epoch ~{c.el() / done * (nbatch - b - 1) / 60:.1f}m left, all ~{c.el() / done * (total - done) / 60:.0f}m left if no early stop'
                )
        m.eval()
        head.eval()
        Fv = feats(va, 'val')
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(H(T(Fv)), T(Y[va])).item()
        hist.append({'epoch': ep, 'train_mse': ls / nbatch, 'val_mse': v})
        imp = v < best[0] * (1 - MIN_GAIN)
        if imp:
            best = (
                v,
                ep,
                (
                    {k: t.detach().clone() for k, t in m.state_dict().items()},
                    {k: t.clone() for k, t in head.state_dict().items()},
                ),
            )
            bad = 0
        else:
            bad += 1
        log(
            f'epoch {ep} done: train MSE {ls / nbatch:.5f}, val MSE {v:.5f} (frozen {base_val:.5f}, best {best[0]:.5f})'
            + ('  <- best' if imp else f'  (no gain >= {MIN_GAIN:.1%}: {bad}/{PAT}; stop at {PAT})')
        )
        if bad >= PAT:
            log('stopping early')
            break
    report['history'] = hist
    assert best[2] is not None, 'fine-tuning never beat the frozen baseline on val'
    m.load_state_dict(best[2][0])
    head.load_state_dict(best[2][1])
    m.eval()
    head.eval()
    Ptr = ph(feats(tr, 'train'))
    Pte = ph(feats(te, 'test'))
    report['teacher'] = {**score(Ptr, Pte, f'teacher (epoch {best[1]})'), 'best_epoch': best[1], 'val_mse': best[0]}
    report['jev_ceiling'] = json.load(open(os.path.join(ROOT, 'out', 'jev_ceiling.json')))['mood']
    torch.save(
        {
            'model_name': NAME,
            'trainable_from_block': cut,
            'words': WORDS,
            'backbone': best[2][0],
            'head': best[2][1],
            'feat_mean': mu.cpu(),
            'feat_std': sd.cpu(),
            'qmap_train_pred': Ptr,
            'qmap_train_y': Y[tr],
            'best_epoch': best[1],
            'val_mse': best[0],
            'train_ids': [R[i]['id'] for i in tr],
        },
        os.path.join(OUT, 'teacher.pt'),
    )
    json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)
    log('saved teacher.pt and report.json; all done')


if __name__ == '__main__':
    main()
