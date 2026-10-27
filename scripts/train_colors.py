# ADR D5 step 4: the color model for rule_v1. Same network as the shipped
# one (lib/hist_colors.HistNet, from libmonet_gdt: the image's color list + the
# 192-value feature), but each role is predicted on its own (no earlier
# picks), like Jev's one call, and the target is Jev's per-role
# probabilities over the app's options (jev/colors.jsonl), spread onto the
# image's colors: t_r(c) = sum_o p_r(o) x softmax_c(-ΔE(c, o) / 5).
# Loss: cross-entropy. Color lists: fit224 (the graph's front end).
# Features: the teacher's (FEAT=teacher, as the shipped color model) or the
# student's (FEAT=student). Model selection on val by mean ΔE between the
# graph's rule_v1 triad (graph_rule.Rule on the model's probabilities) and
# the app's rule_v1 triad on Jev's answer. Test and val reported.
#   .venv/bin/python -u scripts/train_colors.py
# Out: out/colors/lists_fit224.npz (all images), color_model.pt (NEW), report.json
import json
import os
import sys
import time

os.environ['LISTS'] = 'fit224'
import numpy as np
import torch  # noqa: E402
from multiprocessing import get_context  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out', 'colors')
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
FEAT = os.environ.get('FEAT', 'teacher')
EPOCHS = int(os.environ.get('EPOCHS', 30))
LR = 3e-4
TAU = 5.0
DEV = 'mps'
ROLES = ('primary', 'secondary', 'tertiary')


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def hist_one(p):
    import hist_colors as HC

    return HC.hist(p)


def main():
    import hist_colors as HC
    import color_space as D
    import graph_rule as GR

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    jev = {r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'jev', 'colors.jsonl'))) if 'error' not in r}
    dart = {
        r['id']: r['triad']
        for r in map(json.loads, open(os.path.join(ROOT, 'out', 'teacher_colors', 'rule_triads.jsonl')))
    }
    M = [m for m in man if m['id'] in jev]
    ids = [m['id'] for m in M]
    sp = np.array([m['split'] for m in M])
    tr, va, te = (np.where(sp == s)[0] for s in ('train', 'val', 'test'))
    # 1. color lists for every image (reuse the teacher A/B cache)
    lp = os.path.join(OUT, 'lists_fit224.npz')
    if os.path.exists(lp) and list(np.load(lp)['ids']) == ids:
        z = np.load(lp)
        RGB, SH, K = z['rgb'], z['sh'], z['k']
        log('loaded color lists')  # once each
    else:
        RGB = np.zeros((len(ids), HC.MAXK, 3), np.uint8)
        SH = np.zeros((len(ids), HC.MAXK), np.float32)
        K = np.zeros(len(ids), np.int32)
        old = os.path.join(
            ROOT, 'out', 'teacher_colors', 'lists_fit224.npz'
        )  # teacher_colors.py's cache (maintainer-only); optional
        z = np.load(old) if os.path.exists(old) else {'ids': np.array([], str), 'rgb': None, 'sh': None, 'k': None}
        o_ids, o_rgb, o_sh, o_k = z['ids'], z['rgb'], z['sh'], z['k']  # npz members load on every access: once
        orow = {i: k for k, i in enumerate(o_ids)}
        todo = [k for k, i in enumerate(ids) if i not in orow]
        have = [k for k, i in enumerate(ids) if i in orow]
        src = [orow[ids[k]] for k in have]
        if have:
            RGB[have], SH[have], K[have] = o_rgb[src], o_sh[src], o_k[src]
        log(f'color lists: {len(ids) - len(todo)} cached, counting {len(todo)}')
        t0 = last = time.time()
        with get_context('spawn').Pool(12) as pool:
            for n, (k, (rgb, sh, _)) in enumerate(
                zip(todo, pool.imap(hist_one, [os.path.join(ROOT, M[k]['path']) for k in todo], chunksize=16))
            ):
                RGB[k, : len(rgb)], SH[k, : len(sh)], K[k] = rgb, sh, len(rgb)
                if time.time() - last >= 10:
                    last = time.time()
                    log(f'.. colors {n + 1}/{len(todo)}, remaining ~{(last - t0) / (n + 1) * (len(todo) - n - 1):.0f}s')
        np.savez(lp, ids=np.array(ids), rgb=RGB, sh=SH, k=K)
        log(f'saved {lp}')
    # 2. features
    if FEAT == 'teacher':
        z = np.load(os.path.join(ROOT, 'out', 'student', 'teacher_targets.npz'))
        assert list(z['ids']) == ids
        F = z['t']  # one access
    else:  # the graph's own `feature` (w8 graph, app decode), cached by train_shape_head.py
        import glob

        fs = sorted(glob.glob(os.path.join(ROOT, 'out', 'shapes', 'features_*.npy')), key=os.path.getmtime)
        assert fs
        F = np.load(fs[-1])
        log(f'student features: {os.path.basename(fs[-1])} (the w8 graph)')
        assert len(F) == len(ids)
    F = torch.nn.functional.normalize(torch.tensor(F, dtype=torch.float32), dim=-1)
    # 3. targets: options Lab (/100) and per-role probabilities, padded to 12
    OL = np.zeros((len(ids), 12, 3), np.float32)
    OP = np.zeros((len(ids), 3, 12), np.float32)
    for k, i in enumerate(ids):
        o = [x['hex'] for x in jev[i]['options']]
        OL[k, : len(o)] = D.rgb2lab([D.hex2rgb(h) for h in o]) / 100
        for r, role in enumerate(ROLES):
            OP[k, r, : len(o)] = [jev[i]['probabilities'][role].get(h, 0.0) for h in o]
    OP /= OP.sum(-1, keepdims=True).clip(1e-9)
    LAB = D.rgb2lab(RGB / 255).astype(np.float32) / 100

    def batch(rows):
        k = int(K[rows].max())
        return (
            torch.from_numpy(LAB[rows, :k]).to(DEV),
            torch.from_numpy(SH[rows, :k]).to(DEV),
            torch.arange(k, device=DEV)[None] < torch.from_numpy(K[rows]).to(DEV)[:, None],
            F[rows].to(DEV),
        )

    def role_logits(net, lab, sh, mask, emb):
        h, q = net.context(lab, sh, mask, emb)
        return [net.logits(h, q, r, lab, mask) for r in range(3)]

    def loss_fn(net, rows):
        lab, sh, mask, emb = batch(rows)
        d = (lab[:, :, None] - torch.from_numpy(OL[rows]).to(DEV)[:, None]).norm(dim=-1) * 100  # (B, n, 12)
        sm = torch.softmax((-d / TAU).masked_fill(~mask[..., None], -1e4), 1)  # over colors
        tot = 0
        for r, lg in enumerate(role_logits(net, lab, sh, mask, emb)):
            t = (sm * torch.from_numpy(OP[rows, r]).to(DEV)[:, None]).sum(-1)
            tot = tot - (t * torch.log_softmax(lg, -1)).sum(-1).mean()
        return tot / 3

    rule = GR.Rule(os.path.join(ROOT, 'out', 'graph', 'hct_lut.f32'))
    tlab = lambda hexes: D.rgb2lab([D.hex2rgb(h) for h in hexes])

    @torch.no_grad()
    def triad_de(net, rows, limit=None):
        net.eval()
        rows = rows[:limit] if limit else rows
        de = []
        for b in range(0, len(rows), 256):
            rr = rows[b : b + 256]
            lab, sh, mask, emb = batch(rr)
            P = torch.stack([torch.softmax(l, -1) for l in role_logits(net, lab, sh, mask, emb)], 1).cpu()  # (B, 3, n)
            for j, k in enumerate(rr):
                n = int(K[k])
                m = max(n, GR.CAND)  # the graph pads the list to >= 64 (share 0)
                codes = torch.zeros(m, dtype=torch.int64)
                codes[:n] = torch.tensor((RGB[k, :n].astype(np.int64) // 8) @ np.array([1024, 32, 1]))
                sh = torch.zeros(m)
                sh[:n] = torch.tensor(SH[k, :n])
                pp = torch.zeros(3, m)
                pp[:, :n] = P[j, :, :n]
                rgb = rule(codes, sh, pp).numpy()
                de.append(np.linalg.norm(D.rgb2lab(rgb / 255) - tlab(dart[ids[k]]), axis=-1))
        net.train()
        return np.array(de)  # (rows, 3)

    net = HC.HistNet(192).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-2)
    best = (1e9, 0, None)
    hist = []
    t0 = last = time.time()
    per = (len(tr) + 127) // 128
    log(
        f'{len(tr)} train, {len(va)} val, {len(te)} test; features {FEAT}; {EPOCHS} epochs; selection: val ΔE to the app rule triad (500 images)'
    )
    for ep in range(1, EPOCHS + 1):
        net.train()
        tl = []
        for k, b in enumerate(torch.randperm(len(tr)).split(128)):
            opt.zero_grad()
            l = loss_fn(net, tr[b.numpy()])
            l.backward()
            opt.step()
            tl.append(l.item())
            if time.time() - last >= 10:
                last = time.time()
                done = (ep - 1) * per + k + 1
                log(
                    f'.. epoch {ep}/{EPOCHS} batch {k + 1}/{per}, loss {np.mean(tl):.4f}; all ~{(last - t0) / done * (EPOCHS * per - done) / 60:.1f}m left'
                )
        de = triad_de(net, va, 500)
        m = float(de.mean())
        hist.append({'epoch': ep, 'loss': float(np.mean(tl)), 'val_de': m, 'val_de_role': de.mean(0).round(2).tolist()})
        imp = m < best[0]
        if imp:
            best = (m, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        log(
            f'epoch {ep} done: loss {np.mean(tl):.4f}; val ΔE to app triad {m:.2f} (P/S/T {"/".join(f"{x:.1f}" for x in de.mean(0))})'
            + ('  <- best' if imp else '')
        )
    net.load_state_dict(best[2])
    rep = {'features': FEAT, 'epochs': EPOCHS, 'best_epoch': best[1], 'history': hist}
    for name, rows in (('val', va), ('test', te)):
        de = triad_de(net, rows)
        rep[name] = {
            'n': len(rows),
            'de_mean': float(de.mean()),
            'de_role': de.mean(0).round(2).tolist(),
            'within_10_all_roles': float(np.mean((de < 10).all(1))),
        }
        log(
            f'{name}: ΔE to app triad {de.mean():.2f} (P/S/T {"/".join(f"{x:.1f}" for x in de.mean(0))}), all 3 within ΔE 10: {np.mean((de < 10).all(1)):.0%}'
        )
    torch.save(
        {
            'state_dict': best[2],
            'features': FEAT,
            'roles_independent': True,
            'tau': TAU,
            'lists': 'fit224',
            'best_epoch': best[1],
            'train_ids': [ids[i] for i in tr],
        },
        os.path.join(OUT, f'color_model_{FEAT}.pt'),
    )
    json.dump(rep, open(os.path.join(OUT, f'report_{FEAT}.json'), 'w'), indent=1)
    log('saved; all done')


if __name__ == '__main__':
    main()
