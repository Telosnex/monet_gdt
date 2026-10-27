# ADR D5 step 2: the mood student. Recipe = libmonet_gdt shrink_mood.py
# train (224 px, 6 of 12 teacher blocks, AdamW wd 0.01, warmup + cosine,
# batch 64, loss = feature MSE + 10 x head MSE vs the teacher), WITHOUT
# random crops (old corpus: no-crop 1e-4 r 0.700 vs
# crop 0.714 at 12 epochs, half the time), lr 1e-4, mirror flips with
# cached teacher targets. Teacher: out/teacher/teacher.pt (read only).
# Split: manifest train; scored each epoch on val (r vs Jev, font top 5%);
# best epoch by val r is saved; final scores on test and, with D8_DIR set,
# on the D8 held-out Unsplash/Wallhaven images (old Jev labels; D8_DIR =
# libmonet_gdt's wallpaper_student, not public, so outside builds skip it).
#   .venv/bin/python -u scripts/train_student.py
# Out (out/student/): teacher_targets.npz, images224.pt (cached),
#   student.pt (NEW file), report.json. Logs every 10 s.
import json
import os
import sys
import time
import numpy as np
import torch
import timm
from PIL import Image

AUG = os.environ.get('AUG') == '1'  # random crops with an online teacher (shrink_mood.py AUG=1, as shipped)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'out', 'student')
# N=<k>: learning curve. The student trains on the first k
# train images in sha256(id) order (nested, as train_teacher.py N=); the
# teacher (full) and val/test/D8 are unchanged. Writes out/curves/student[_aug]_<k>/.
NSUB = int(os.environ.get('N', 0))
OUT = (
    os.path.join(ROOT, 'out', 'curves', f"student{'_aug' if AUG else ''}_{NSUB}")
    if NSUB
    else os.path.join(ROOT, 'out', 'student_aug' if AUG else 'student')
)
if os.environ.get('OUT_NAME'):
    OUT = os.path.join(ROOT, 'out', os.environ['OUT_NAME'])  # e.g. student_matched50k
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
LG = os.environ.get('D8_DIR')  # libmonet_gdt wallpaper_student: the D8 held-out set (not public); unset = no D8 scores
RES, DEPTH = 224, 6
LR = float(os.environ.get('LR', 1e-4))
EPOCHS = int(os.environ.get('EPOCHS', 20))
BS = 64
DEV = 'mps'
PAT = int(os.environ.get('PATIENCE', 4))  # stop after PAT epochs without a new best val r (+0.002)


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


class Files(torch.utils.data.Dataset):  # shrink_mood.transform: resize short side + center crop, uint8
    def __init__(s, paths, res):
        s.paths, s.res = paths, res

    def __len__(s):
        return len(s.paths)

    def __getitem__(s, i):
        import torchvision.transforms as TV

        t = TV.Compose(
            [TV.Resize(s.res, interpolation=TV.InterpolationMode.BICUBIC, antialias=True), TV.CenterCrop(s.res)]
        )
        return torch.from_numpy(np.asarray(t(Image.open(s.paths[i]).convert('RGB')), dtype=np.uint8).copy()).permute(
            2, 0, 1
        )


def load(paths, res, label):
    dl = torch.utils.data.DataLoader(Files(paths, res), batch_size=32, num_workers=12)
    out, n, t, last = [], 0, time.time(), time.time()
    for b in dl:
        out.append(b)
        n += len(b)
        if time.time() - last >= 10:
            last = time.time()
            log(
                f'.. decoding {label} at {res} px: {n}/{len(paths)}, remaining ~{(last - t) / n * (len(paths) - n):.0f}s'
            )
    log(f'decoded {len(paths)} {label} images at {res} px in {time.time() - t:.0f}s')
    return torch.cat(out)


def norm(u):
    return u.float().div_(127.5).sub_(1.0)


def main():
    import train_teacher as TT

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    ck = torch.load(os.path.join(ROOT, 'out', 'teacher', 'teacher.pt'), weights_only=False)
    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    lab = {
        r['id']: r['raw'] for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'mood.jsonl'))) if 'error' not in r
    }
    R = [m for m in man if m['id'] in lab]
    ids = [m['id'] for m in R]
    Y = np.array([[lab[i][w] for w in TT.WORDS] for i in ids], np.float32)
    sp = np.array([m['split'] for m in R])
    tr, va, te = (np.where(sp == s)[0] for s in ('train', 'val', 'test'))
    # D8 held-out (old labels), only with D8_DIR
    HR = []
    if LG:
        sys.path.append(LG)
        import mood_student as MS

        keys = set(json.load(open(os.path.join(LG, 'out', 'mood_student', 'heldout_keys.json'))))
        HR = [r for r in MS.labeled() if r['name'] in keys]
        HY = np.array([[r['raw'][w] for w in TT.WORDS] for r in HR], np.float32)
        hsrc = np.array([r['name'].split('__')[0] for r in HR])

    tm = timm.create_model(ck['model_name'], pretrained=True, num_classes=0)
    tm.load_state_dict(ck['backbone'])
    tm = tm.to(DEV).eval()
    l1, l2 = torch.nn.Linear(192, 256), torch.nn.Linear(256, 10)
    l1.load_state_dict({'weight': ck['head']['1.weight'], 'bias': ck['head']['1.bias']})
    l2.load_state_dict({'weight': ck['head']['4.weight'], 'bias': ck['head']['4.bias']})
    head = torch.nn.Sequential(l1, torch.nn.GELU(), l2, torch.nn.Sigmoid()).to(DEV).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    mu, sd = ck['feat_mean'].to(DEV).reshape(1, -1), ck['feat_std'].to(DEV).reshape(1, -1)
    qmap = TT.qmap_fit(np.asarray(ck['qmap_train_pred']), np.asarray(ck['qmap_train_y']))

    # 1. teacher targets (plain + flipped) from the token cache
    tp = os.path.join(BASE, 'teacher_targets.npz')
    if os.path.exists(tp) and list(np.load(tp)['ids']) == ids:
        z = np.load(tp)
        T, Tf = torch.tensor(z['t']), torch.tensor(z['tf'])
        log('loaded teacher targets')
    else:
        c = torch.load(os.path.join(ROOT, 'out', 'teacher', 'cache_cut8.pt'))
        assert c['ids'] == ids
        X = c['x']
        cut = ck['trainable_from_block']
        with torch.no_grad():
            _, rope = tm._pos_embed(tm.patch_embed(torch.zeros(1, 3, 384, 384, device=DEV)))
            out, t0, last = [], time.time(), time.time()
            for s in range(0, len(ids), 128):
                x = X[s : s + 128].to(DEV).float()
                for b in range(cut, len(tm.blocks)):
                    x = tm.blocks[b](x, rope=rope)
                out.append(tm.forward_head(tm.norm(x), pre_logits=True).float().cpu())
                if time.time() - last >= 10:
                    last = time.time()
                    log(
                        f'.. teacher targets {s + 128}/{len(ids)}, remaining ~{(last - t0) / (s + 128) * (len(ids) - s - 128):.0f}s'
                    )
        T = torch.cat(out)
        log('teacher targets (plain) done; flipped needs pixels')
        Tf = torch.empty_like(T)
        dl = torch.utils.data.DataLoader(
            Files([os.path.join(ROOT, R[i]['path']) for i in range(len(ids))], 384), batch_size=64, num_workers=12
        )
        k, t0, last = 0, time.time(), time.time()
        with torch.no_grad():
            for b in dl:
                Tf[k : k + len(b)] = tm(norm(b).flip(-1).to(DEV)).float().cpu()
                k += len(b)
                if time.time() - last >= 10:
                    last = time.time()
                    log(
                        f'.. teacher targets (flipped) {k}/{len(ids)}, remaining ~{(last - t0) / k * (len(ids) - k):.0f}s'
                    )
        np.savez(tp, ids=np.array(ids), t=T.numpy(), tf=Tf.numpy())
        log(f'saved {tp}')

    # 2. images at 224 (cached)
    ip = os.path.join(BASE, 'images224.pt')
    if os.path.exists(ip) and torch.load(ip)['ids'] == ids:
        U = torch.load(ip)['u']
        log('loaded 224 px images')
    else:
        U = load([os.path.join(ROOT, m['path']) for m in R], RES, 'corpus')
        torch.save({'ids': ids, 'u': U}, ip)
    if HR:
        HU = load([r['path'] for r in HR], RES, 'D8 held-out')
    if AUG:
        import augment as SM

    if NSUB or os.environ.get('OUT_NAME'):
        assert not os.path.exists(os.path.join(OUT, 'student.pt')), f'{OUT} exists; use a new name'
    if NSUB:  # after the caches
        import hashlib

        tr = np.sort(np.array(sorted(tr, key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())[:NSUB]))
        log(f'learning curve: student trains on {len(tr)} of the train split (sha256 order)')
    os.makedirs(OUT, exist_ok=True)
    # 3. student = 6 evenly spaced teacher blocks (shrink_mood.make_student)
    stu = timm.create_model(ck['model_name'], pretrained=False, num_classes=0, img_size=RES, depth=DEPTH)
    sdict = dict(ck['backbone'])
    keep = np.unique(np.round(np.linspace(0, 11, DEPTH)).astype(int)).tolist()
    new = {k: v for k, v in sdict.items() if not k.startswith('blocks.')}
    for j, b in enumerate(keep):
        for k, v in sdict.items():
            if k.startswith(f'blocks.{b}.'):
                new[f'blocks.{j}.' + k[len(f'blocks.{b}.') :]] = v
    from timm.layers import resample_abs_pos_embed

    new['pos_embed'] = resample_abs_pos_embed(sdict['pos_embed'], (RES // 16, RES // 16), num_prefix_tokens=1)
    r_ = stu.load_state_dict(new, strict=False)
    assert not r_.unexpected_keys and all('rope' in k for k in r_.missing_keys), r_
    stu = stu.to(DEV)

    JS = {}

    @torch.no_grad()
    def score(Uix, Yt, label=None):
        stu.eval()
        F = torch.cat([stu(norm(Uix[i : i + 128]).to(DEV)) for i in range(0, len(Uix), 128)])
        P = qmap(head((F - mu) / sd).cpu().numpy())
        stu.train()
        r = float(np.mean([np.corrcoef(P[:, k], Yt[:, k])[0, 1] for k in range(10)]))
        key = id(Yt)
        if key not in JS:
            JS[key] = [TT.font_scores(y) for y in Yt]
        rk = np.array([np.mean(JS[key][i] > JS[key][i][int(np.argmax(TT.font_scores(P[i])))]) for i in range(len(P))])
        return {'r': r, 'top5pct': float(np.mean(rk <= 0.05)), 'rank_median': float(np.median(rk))}, P

    Yva = Y[va]
    s0, _ = score(U[va], Yva)
    log(
        f'student before training (6 teacher blocks): val r {s0["r"]:.3f}, top5% {s0["top5pct"]:.2f}; {sum(p.numel() for p in stu.parameters()) / 1e6:.2f}M params, blocks {keep}'
    )
    opt = torch.optim.AdamW(stu.parameters(), lr=LR, weight_decay=0.01)
    # AUG: train images stream from their JPEGs at 384 (same Files transform as
    # the old in-memory cache, so the same pixels), decoded by workers; no
    # raw-pixel cache (11 GB per 25K images). EXTRA=<jsonl with 'path'>: more
    # images for the student only (no labels needed; the teacher scores them
    # online), e.g. data/matched.jsonl.
    if AUG:
        TP = [os.path.join(ROOT, R[i]['path']) for i in tr]
        if os.environ.get('EXTRA'):
            ex = [json.loads(l)['path'] for l in open(os.environ['EXTRA'])]
            ex = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in ex]
            TP += ex
            log(f'student train images: {len(tr)} corpus + {len(ex)} extra = {len(TP)}')

        class Batches:
            def __init__(s):
                s.b = []

            def __iter__(s):
                return iter(s.b)

            def __len__(s):
                return len(s.b)

        BSAMP = Batches()
        LOADER = torch.utils.data.DataLoader(
            Files(TP, 384), batch_sampler=BSAMP, num_workers=10, persistent_workers=True, prefetch_factor=4
        )
        NTR = len(TP)
        # shrink_mood.crops leaves the first half of each batch uncropped (flip
        # only): its teacher targets come from a plain + flipped cache made with
        # the same pixels (Files at 384, norm, .flip). Exact; teacher work
        # halves. NOHALF=1 runs the teacher on the whole batch (as before).
        HALF = os.environ.get('NOHALF') != '1'
        if HALF:
            import hashlib

            key = hashlib.sha256('\n'.join(TP).encode()).hexdigest()[:12]
            hp = os.path.join(BASE, f'targets384_{key}.pt')
            if os.path.exists(hp):
                HT, HTf = torch.load(hp)
                log(f'loaded uncropped-half teacher targets ({hp})')
            else:
                HT = torch.empty(NTR, 192)
                HTf = torch.empty(NTR, 192)
                hl = torch.utils.data.DataLoader(Files(TP, 384), batch_size=128, num_workers=10)
                k, t0, last = 0, time.time(), time.time()
                with torch.no_grad():
                    for u in hl:
                        x = norm(u).to(DEV)
                        HT[k : k + len(u)] = tm(x).float().cpu()
                        HTf[k : k + len(u)] = tm(x.flip(-1)).float().cpu()
                        k += len(u)
                        if time.time() - last >= 10:
                            last = time.time()
                            log(f'.. uncropped-half targets {k}/{NTR}, remaining ~{(last - t0) / k * (NTR - k):.0f}s')
                torch.save((HT, HTf), hp)
                log(f'saved {hp} in {time.time() - t0:.0f}s')
    else:
        NTR = len(tr)
    per = NTR // BS
    steps = EPOCHS * per
    warm = max(1, per // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, steps) / steps))
    )
    g = torch.Generator().manual_seed(0)
    t0 = last = time.time()
    best = (-1, 0, None)
    bad = 0
    hist = []
    done = 0
    log(
        f'training: up to {EPOCHS} epochs x {per} batches, lr {LR}, {"random crops (online teacher)" if AUG else "no crops"}; stop after {PAT} epochs without +0.002 val r'
    )
    stu.train()
    for ep in range(1, EPOCHS + 1):
        if AUG:
            order = torch.randperm(NTR, generator=g).tolist()
            BSAMP.b = [order[i * BS : (i + 1) * BS] for i in range(per)]
            it = iter(LOADER)
        else:
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        tot = 0.0
        for i in range(per):
            b = BSAMP.b[i] if AUG else perm[i * BS : (i + 1) * BS]
            fl = torch.rand(len(b), generator=g) < 0.5
            if AUG:
                xt = SM.crops(norm(next(it)).to(DEV), fl, g)
                if HALF:
                    h = len(b) // 2
                    bi = torch.tensor(b[:h])
                    with torch.no_grad():
                        tc = tm(xt[h:])
                    tgt = torch.cat([torch.where(fl[:h, None], HTf[bi], HT[bi]).to(DEV), tc])
                    if ep == 1 and i == 0:  # exactness check on the first batch
                        with torch.no_grad():
                            on = tm(xt[:h])
                        d = (on - tgt[:h]).abs().max().item()
                        cs = torch.nn.functional.cosine_similarity(on, tgt[:h]).min().item()
                        log(f'uncropped half: cached vs online teacher max |diff| {d:.2e}, min cosine {cs:.6f}')
                        assert cs > 0.9999, 'cached targets differ'
                else:
                    with torch.no_grad():
                        tgt = tm(xt)
                xs = torch.nn.functional.interpolate(
                    xt, size=(RES, RES), mode='bicubic', antialias=True, align_corners=False
                )
            else:
                x = norm(U[b])
                x[fl] = x[fl].flip(-1)
                xs = x.to(DEV)
                tgt = torch.where(fl[:, None], Tf[b], T[b]).to(DEV)
            zs, zt = (stu(xs) - mu) / sd, (tgt - mu) / sd
            loss = torch.nn.functional.mse_loss(zs, zt) + 10 * torch.nn.functional.mse_loss(head(zs), head(zt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            done += 1
            if time.time() - last >= 10:
                last = time.time()
                el = last - t0
                log(
                    f'.. epoch {ep}/{EPOCHS} batch {i + 1}/{per}, loss {tot / (i + 1):.4f}; elapsed {el / 60:.1f}m, all ~{el / done * (steps - done) / 60:.0f}m left if no early stop'
                )
        s, _ = score(U[va], Yva)
        hist.append({'epoch': ep, 'loss': tot / per, **s})
        imp = s['r'] > best[0] + 0.002
        if HR and str(ep) in os.environ.get('D8_AT', '').split(','):  # D8 at chosen epochs (how many epochs are needed)
            d8e, _ = score(HU, HY)
            hist[-1]['d8'] = d8e
            log(f'epoch {ep}: D8 r {d8e["r"]:.3f}, top5% {d8e["top5pct"]:.2f}')
        if imp:
            best = (s['r'], ep, {k: v.detach().cpu().clone() for k, v in stu.state_dict().items()})
            bad = 0
        else:
            bad += 1
        log(
            f'epoch {ep} done: loss {tot / per:.4f}; val r {s["r"]:.3f}, top5% {s["top5pct"]:.2f}'
            + ('  <- best' if imp else f'  (no gain {bad}/{PAT})')
        )
        if bad >= PAT:
            log('stopping early')
            break
    stu.load_state_dict(best[2])
    stu.eval()
    rep = {
        'recipe': {
            'res': RES,
            'depth': DEPTH,
            'blocks': keep,
            'lr': LR,
            'crops': AUG,
            'epochs_max': EPOCHS,
            'patience': PAT,
            'extra': os.environ.get('EXTRA'),
            'n_train_images': NTR,
            'half_cached': bool(AUG and HALF),
        },
        'best_epoch': best[1],
        'history': hist,
    }
    rep['test'], _ = score(U[te], Y[te])
    log(f'student test: r {rep["test"]["r"]:.3f}, top5% {rep["test"]["top5pct"]:.2f}')
    if HR:
        d8, P = score(HU, HY)
        rep['d8_all'] = d8
        for s in sorted(set(hsrc)):
            ix = np.where(hsrc == s)[0]
            rep[f'd8_{s}'] = {'r': float(np.mean([np.corrcoef(P[ix, k], HY[ix, k])[0, 1] for k in range(10)]))}
        log(
            f'student D8 held-out: r {d8["r"]:.3f}, top5% {d8["top5pct"]:.2f} ('
            + ', '.join(f"{s} {rep[f'd8_{s}']['r']:.3f}" for s in sorted(set(hsrc)))
            + ')'
        )
    torch.save(
        {
            'model_name': ck['model_name'],
            'res': RES,
            'depth': DEPTH,
            'keep': keep,
            'backbone': best[2],
            'teacher': 'out/teacher/teacher.pt',
            'best_epoch': best[1],
            'val_r': best[0],
        },
        os.path.join(OUT, 'student.pt'),
    )
    json.dump(rep, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)
    log('saved student.pt, report.json; all done')


if __name__ == '__main__':
    main()
