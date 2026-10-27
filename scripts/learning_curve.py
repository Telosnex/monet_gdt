# Learning curve: how mood accuracy grows with the
# number of Jev-labeled images. Frozen PE-Core tiny (pretrained weights,
# tokens from out/teacher/cache_cut8.pt) + the frozen-baseline head of
# train_teacher.py, trained on random subsets of the train split (val for
# early stopping), scored on the test split and on the D8 held-out
# Unsplash/Wallhaven images. Fit per set: 1 - r = a x n^-b.
#   <libmonet venv>/bin/python -u scripts/learning_curve.py -> out/learning_curve.json
import json
import os
import sys
import time
import private_paths as PP

LG = PP.libmonet_gdt('wallpaper_student')
sys.path.insert(0, LG)
import numpy as np
import torch
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
DEV = 'mps'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def main():
    import train_teacher as TT
    import mood_student as MS

    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    lab = {
        r['id']: r['raw'] for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'mood.jsonl'))) if 'error' not in r
    }
    R = [m for m in man if m['id'] in lab]
    Y = np.array([[lab[m['id']][w] for w in TT.WORDS] for m in R], np.float32)
    sp = np.array([m['split'] for m in R])
    tr, va, te = (np.where(sp == s)[0] for s in ('train', 'val', 'test'))
    m = timm.create_model(TT.NAME, pretrained=True, num_classes=0).to(DEV).eval()
    cut = 8
    fp = os.path.join(ROOT, 'out', 'learning_curve_feats.npz')
    if os.path.exists(fp):
        z = np.load(fp)
        F, FD = z['f'], z['fd']
    else:
        c = torch.load(os.path.join(ROOT, 'out', 'teacher', 'cache_cut8.pt'))['x']
        out = []
        t0 = last = time.time()
        with torch.no_grad():
            _, rope = m._pos_embed(m.patch_embed(torch.zeros(1, 3, 384, 384, device=DEV)))
            for s in range(0, len(c), 128):
                x = c[s : s + 128].to(DEV).float()
                for b in range(cut, 12):
                    x = m.blocks[b](x, rope=rope)
                out.append(m.forward_head(m.norm(x), pre_logits=True).float().cpu().numpy())
                if time.time() - last >= 10:
                    last = time.time()
                    log(
                        f'.. frozen features {s + 128}/{len(c)}, remaining ~{(last - t0) / (s + 128) * (len(c) - s - 128):.0f}s'
                    )
        F = np.concatenate(out)
        keys = set(json.load(open(os.path.join(LG, 'out', 'mood_student', 'heldout_keys.json'))))
        HR = [r for r in MS.labeled() if r['name'] in keys]
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
        dl = torch.utils.data.DataLoader(TT.Images([r['path'] for r in HR], tf), batch_size=32, num_workers=8)
        with torch.no_grad():
            FD = np.concatenate([m(b.to(DEV)).float().cpu().numpy() for b in dl])
        np.savez(fp, f=F, fd=FD)
        log('frozen features done')
    keys = set(json.load(open(os.path.join(LG, 'out', 'mood_student', 'heldout_keys.json'))))
    YD = np.array([[r['raw'][w] for w in TT.WORDS] for r in MS.labeled() if r['name'] in keys], np.float32)
    T = lambda a: torch.tensor(a, device=DEV)

    def fit(ix, seed):
        torch.manual_seed(seed)
        mu, sd = T(F[ix].mean(0)), T(F[ix].std(0) + 1e-6)
        head = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(192, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 10),
            torch.nn.Sigmoid(),
        ).to(DEV)
        H = lambda x: head((x - mu) / sd)
        opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
        best = (1e9, None)
        bad = 0
        Xt, Yt, Xv, Yv = T(F[ix]), T(Y[ix]), T(F[va]), T(Y[va])
        for ep in range(1, 3001):
            head.train()
            opt.zero_grad()
            torch.nn.functional.mse_loss(H(Xt), Yt).backward()
            opt.step()
            if ep % 25 == 0:
                head.eval()
                with torch.no_grad():
                    v = torch.nn.functional.mse_loss(H(Xv), Yv).item()
                if v < best[0]:
                    best = (v, {k: t.clone() for k, t in head.state_dict().items()})
                    bad = 0
                else:
                    bad += 1
                if bad >= 12:
                    break
        head.load_state_dict(best[1])
        head.eval()
        with torch.no_grad():
            ptr, pte, pd = (H(T(a)).cpu().numpy() for a in (F[ix], F[te], FD))
        q = TT.qmap_fit(ptr, Y[ix])
        r = lambda P, Yy: float(np.mean([np.corrcoef(P[:, k], Yy[:, k])[0, 1] for k in range(10)]))
        return r(q(pte), Y[te]), r(q(pd), YD)

    res = []
    rng = np.random.default_rng(0)
    for n in (400, 800, 1600, 3200, 6400, 12800, len(tr)):
        seeds = 1
        rt, rd = [], []  # one seed at every size
        for s in range(seeds):
            ix = rng.choice(tr, n, replace=False) if n < len(tr) else tr
            a, b = fit(ix, s)
            rt.append(a)
            rd.append(b)
        res.append({'n': int(n), 'test_r': float(np.mean(rt)), 'd8_r': float(np.mean(rd)), 'seeds': seeds})
        log(f'n {n:6d}: test r {np.mean(rt):.3f}, D8 r {np.mean(rd):.3f}')
    out = {'rows': res}
    for k in ('test_r', 'd8_r'):
        n = np.array([x['n'] for x in res], float)
        g = 1 - np.array([x[k] for x in res])
        b, la = np.polyfit(np.log(n), np.log(g), 1)
        a = np.exp(la)
        pred = {N: float(1 - a * N**b) for N in (50000, 100000, 200000)}
        out[f'fit_{k}'] = {'a': float(a), 'b': float(b), 'gap_shrink_per_doubling': float(1 - 2**b), 'pred': pred}
        log(
            f'{k}: 1 - r = {a:.3f} x n^{b:.3f}; gap shrinks {1 - 2**b:.0%} per doubling; predicted r at 50K {pred[50000]:.3f}, 100K {pred[100000]:.3f}, 200K {pred[200000]:.3f}'
        )
    json.dump(out, open(os.path.join(ROOT, 'out', 'learning_curve.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
