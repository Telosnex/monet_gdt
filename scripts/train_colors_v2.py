# Color head on labels from monet_ui's MonetGdtRule20261030 (QuantizerCelebi,
# 128 clusters, on telosnex's 96 px thumbnail; tools/v2_labels). No Jev labels.
# Inputs from scripts/v2_prep.py: the base graph's feature and color list
# per image. Network: lib/hist_colors.HistNet with neigh=True: each color also
# gets its neighborhood share (list colors within ΔE 10).
# Loss: per role, cross-entropy to softmax_c(-ΔE(c, label role color) / 5) over
# the image's colors. S is scored given P, and T given P and S (the label's
# colors, or the model's own pick with probability OWN).
# Pick: argmax per role. S and T are never within ΔE 10 of an earlier pick
# (any color, if none is left).
# Split: corpus train + matched (minus a 4% hash holdout) for training; corpus
# val for early stopping; test = corpus test and the matched holdout.
#   .venv/bin/python -u scripts/train_colors_v2.py
# Out: out/v2/color_model.pt, out/v2/report.json, out/v2/test.jsonl
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
V2 = os.path.join(ROOT, 'out', 'v2')
EPOCHS = int(os.environ.get('EPOCHS', 30))
PATIENCE = int(os.environ.get('PATIENCE', 4))
OWN = float(os.environ.get('OWN', 0.5))
LIMIT = int(os.environ.get('LIMIT', 0))  # train images, for a quick check
BS, LR, TAU, EXCL = 128, 3e-4, 5.0, 10.0
DEV = 'mps'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def holdout(i):
    return int(hashlib.sha256(i.encode()).hexdigest(), 16) % 100 < 4


def load():
    import color_space as D

    meta = {m['id']: m for m in json.load(open(os.path.join(V2, 'ids.json')))}
    ids, F, L, C, S, Y, SHIP = [], [], [], [], [], [], []
    for k in range(10_000):
        p = os.path.join(V2, f'prep_{k:03d}.npz')
        if not os.path.exists(p):
            break
        z = dict(np.load(p))  # npz members re-read on every access: read each once
        lab = [json.loads(l)['triad'] for l in open(os.path.join(V2, f'labels_{k:03d}.jsonl'))]
        ok = z['ok']
        ids += [str(i) for i, g in zip(z['ids'], ok) if g]
        F.append(z['feature'][ok])
        SHIP.append(z['ship'][ok])
        keep = np.repeat(ok, z['lens'])  # per color
        C.append(z['codes'][keep])
        S.append(z['share'][keep])
        L.append(z['lens'][ok])
        Y += [lab[j] for j in np.where(ok)[0]]
    lens = np.concatenate(L)
    y = np.stack([D.rgb2lab([D.hex2rgb(h) for h in t]) for t in Y]).astype(np.float32) / 100
    split = []
    for i in ids:
        m = meta[i]
        split.append(m['split'] if m['src'] == 'corpus' else ('holdout' if holdout(i) else 'train'))
    return dict(
        ids=ids,
        feat=np.concatenate(F),
        lens=lens,
        codes=np.concatenate(C).astype(np.int64),
        share=np.concatenate(S).astype(np.float32),
        y=y,
        hexes=Y,
        ship=np.concatenate(SHIP),
        split=np.array(split),
        src=np.array([meta[i]['src'] for i in ids]),
    )


class Data:
    """Everything on the GPU; a batch is a gather by offsets."""

    def __init__(self, d):
        import color_space as D

        allc = np.arange(32768)
        rgb = np.stack([allc // 1024, (allc // 32) % 32, allc % 32], -1) * 8 + 4
        self.lut = torch.tensor(D.rgb2lab(rgb / 255) / 100, dtype=torch.float32, device=DEV)
        self.codes = torch.tensor(d['codes'], device=DEV)
        self.share = torch.tensor(d['share'], device=DEV)
        self.lens = torch.tensor(d['lens'], device=DEV)
        self.off = torch.tensor(np.concatenate([[0], np.cumsum(d['lens'])[:-1]]), device=DEV)
        self.feat = torch.nn.functional.normalize(torch.tensor(d['feat'], device=DEV), dim=-1)
        self.y = torch.tensor(d['y'], device=DEV)
        self.lens_np = d['lens']
        self.nb = self._neigh()

    def _neigh(self):
        """Per list color, share of the image within ΔE 10 (as the graph: |a|^2+|b|^2-2ab < 100)."""
        out = torch.zeros_like(self.share)
        for b in batches(self, np.arange(len(self.lens_np)), False, 16):
            r = torch.as_tensor(b, device=DEV)
            k = int(self.lens[r].max())
            ar = torch.arange(k, device=DEV)
            mask = ar[None] < self.lens[r][:, None]
            idx = (self.off[r][:, None] + ar[None]).clamp(max=len(self.codes) - 1)
            L = self.lut[self.codes[idx]] * 100
            sh = self.share[idx] * mask
            sq = (L * L).sum(-1)
            d2 = sq[:, :, None] + sq[:, None] - 2 * L @ L.transpose(1, 2)
            nb = ((d2 < 100).float() @ sh[..., None]).squeeze(-1)
            out[idx[mask]] = nb[mask]
        return out

    def batch(self, rows):
        r = torch.as_tensor(rows, device=DEV)
        k = int(self.lens[r].max())
        ar = torch.arange(k, device=DEV)
        mask = ar[None] < self.lens[r][:, None]
        idx = (self.off[r][:, None] + ar[None]).clamp(max=len(self.codes) - 1)
        lab = self.lut[self.codes[idx]] * mask[..., None]
        sh = self.share[idx] * mask
        nb = self.nb[idx] * mask
        return lab, sh, mask, self.feat[r], self.y[r], nb


def batches(D, rows, shuffle, bs=None):
    """Batches of similar list length (less padding), in random order."""
    o = rows[np.argsort(D.lens_np[rows] + (np.random.rand(len(rows)) * 64 if shuffle else 0), kind='stable')]
    n = bs or BS
    bs = [o[i : i + n] for i in range(0, len(o), n)]
    if shuffle:
        np.random.shuffle(bs)
    return bs


def target(lab, mask, yr):
    d = (lab - yr[:, None]).norm(dim=-1) * 100
    return torch.softmax((-d / TAU).masked_fill(~mask, -1e4), -1)


def loss_fn(net, b):
    lab, sh, mask, emb, y, nb = b
    h, q = net.context(lab, sh, mask, emb, nb)
    ar = torch.arange(len(lab), device=DEV)
    tot, refs = 0, []
    for r in range(3):
        lg = net.logits(h, q, r, lab, mask, *(refs + [None, None])[:2])
        tot = tot - (target(lab, mask, y[:, r]) * torch.log_softmax(lg, -1)).sum(-1).mean()
        if r < 2:
            mine = lab[ar, lg.argmax(-1)].detach()
            use = torch.rand(len(lab), 1, device=DEV) < OWN
            refs.append(torch.where(use, mine, y[:, r]))
    return tot / 3


@torch.no_grad()
def pick(net, lab, sh, mask, emb, nb):
    h, q = net.context(lab, sh, mask, emb, nb)
    ar = torch.arange(len(lab), device=DEV)
    refs, out = [], []
    for r in range(3):
        lg = net.logits(h, q, r, lab, mask, *(refs + [None, None])[:2])
        if refs:
            dmin = torch.stack([(lab - c[:, None]).norm(dim=-1) * 100 for c in refs]).min(0).values
            ok = mask & (dmin >= EXCL)
            ok = torch.where(ok.any(-1, keepdim=True), ok, mask)
            lg = lg.masked_fill(~ok, -1e9)
        i = lg.argmax(-1)
        out.append(lab[ar, i])
        refs.append(lab[ar, i])
    return torch.stack(out, 1)  # (B, 3, 3) Lab / 100


@torch.no_grad()
def predict(net, D, rows):
    """Picked triads for rows, Lab / 100 (len, 3, 3)."""
    net.eval()
    o = torch.zeros(len(D.lens_np), 3, 3, device=DEV)
    for b in batches(D, rows, False):
        lab, sh, mask, emb, _, nb = D.batch(b)
        o[torch.as_tensor(b, device=DEV)] = pick(net, lab, sh, mask, emb, nb)
    net.train()
    return o[torch.as_tensor(rows, device=DEV)]


def de(pred, D, rows):
    return ((pred - D.y[torch.as_tensor(rows, device=DEV)]).norm(dim=-1) * 100).cpu().numpy()  # (n, 3)


def main():
    import color_space as CS

    torch.manual_seed(0)
    np.random.seed(0)
    t0 = time.time()
    d = load()
    D = Data(d)
    sp = d['split']
    tr, va = np.where(sp == 'train')[0], np.where(sp == 'val')[0]
    if LIMIT:
        tr = np.random.RandomState(0).choice(tr, LIMIT, replace=False)
    tests = {'corpus_test': np.where(sp == 'test')[0], 'matched_holdout': np.where(sp == 'holdout')[0]}
    log(
        f'{len(tr)} train ({(d["src"][tr] == "matched").sum()} matched), {len(va)} val, '
        + ', '.join(f'{k} {len(v)}' for k, v in tests.items())
        + f'; loaded in {time.time() - t0:.0f}s; select on val ΔE'
    )
    import hist_colors as HC

    net = HC.HistNet(192, neigh=True).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-2)
    best, hist, bad = (1e9, 0, None), [], 0
    for ep in range(1, EPOCHS + 1):
        te0, tl = time.time(), []
        for b in batches(D, tr, True):
            opt.zero_grad()
            l = loss_fn(net, D.batch(b))
            l.backward()
            opt.step()
            tl.append(l.item())
        e = de(predict(net, D, va), D, va)
        m = float(e.mean())
        hist.append(
            {
                'epoch': ep,
                'loss': float(np.mean(tl)),
                'val_de': m,
                'val_de_role': e.mean(0).round(2).tolist(),
                's': round(time.time() - te0),
            }
        )
        imp = m < best[0] - 1e-3
        if imp:
            best, bad = (m, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}), 0
        else:
            bad += 1
        log(
            f'epoch {ep}: loss {np.mean(tl):.4f}, val ΔE {m:.2f} (P/S/T {"/".join(f"{x:.1f}" for x in e.mean(0))}), {time.time() - te0:.0f}s'
            + ('  <- best' if imp else '')
        )
        if bad >= PATIENCE:
            log(f'no gain for {PATIENCE} epochs; stop')
            break
    net.load_state_dict(best[2])
    rep = {
        'own': OWN,
        'best_epoch': best[1],
        'history': hist,
        'train_n': len(tr),
        'labels': json.load(open(os.path.join(V2, 'labels_meta.json'))),
    }
    rows_out = []
    for name, rows in [('val', va), *tests.items()]:
        P = {'model': predict(net, D, rows)}
        ship = torch.tensor(
            CS.rgb2lab(d['ship'][rows].reshape(-1, 3) / 255).reshape(-1, 3, 3) / 100, dtype=torch.float32, device=DEV
        )
        r = {'n': len(rows)}
        for k, v in list(P.items()) + [('ship', ship)]:
            e = de(v, D, rows)
            r[k] = {
                'de_mean': round(float(e.mean()), 2),
                'de_role': [round(float(x), 2) for x in e.mean(0)],
                'all_within_10': round(float((e < 10).all(1).mean()), 3),
            }
        rep[name] = r
        log(
            f'{name}: '
            + ' | '.join(
                f'{k} ΔE {v["de_mean"]} (P/S/T {"/".join(str(x) for x in v["de_role"])}, all<10 {v["all_within_10"]:.0%})'
                for k, v in r.items()
                if k != 'n'
            )
        )
        if name != 'val':
            pk = P['model'].cpu().numpy()
            for j, k in enumerate(rows):
                rows_out.append(
                    {
                        'id': d['ids'][k],
                        'set': name,
                        'v2': d['hexes'][k],
                        'pred_lab': (pk[j] * 100).round(2).tolist(),
                        'ship': ['#%02x%02x%02x' % tuple(int(c) for c in x) for x in d['ship'][k]],
                    }
                )
    torch.save(
        {
            'state_dict': best[2],
            'neigh': True,
            'arm': 'direct',
            'own': OWN,
            'excl_de': EXCL,
            'tau': TAU,
            'lists': 'graph',
            'best_epoch': best[1],
            'labels': rep['labels'],
        },
        os.path.join(V2, 'color_model.pt'),
    )
    json.dump(rep, open(os.path.join(V2, 'report.json'), 'w'), indent=1)
    with open(os.path.join(V2, 'test.jsonl'), 'w') as f:
        for r in rows_out:
            f.write(json.dumps(r) + '\n')
    log(f'saved; {time.time() - t0:.0f}s total')


if __name__ == '__main__':
    main()
