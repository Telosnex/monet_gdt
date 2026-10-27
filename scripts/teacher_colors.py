# Teacher-level color A/B: the shipped color model
# recipe (libmonet_gdt hist_colors.py, hist_fit224_conf_e30: fit224 color
# lists, CE to softmax(-ΔE/5) per role, reuse penalty 1, 30 epochs, area
# pick chroma1_x10) trained on the NEW teacher's features and the new
# rule_v1 targets (out/teacher_colors/rule_triads.jsonl, app code), vs the
# shipped color model on the OLD teacher's features (it was trained on
# them). Images: 60 new-corpus test-split images neither side trained on.
# Teachers are read only; the new color model is a new file.
#   .venv/bin/python -u scripts/teacher_colors.py
# Out: out/teacher_colors/ (lists, features, hist_fit224_newteacher_e30.pt),
#   telosnex dev/wallpaper_colors/out/monet_teacher_ab/raw.jsonl arms
#   old_teacher, new_teacher. Every step is cached; logs every 10 s.
import hashlib
import json
import os
import shutil
import sys
import time

os.environ['LISTS'] = 'fit224'
os.environ.setdefault('EMB', 'petiny-mood-full')
import private_paths as PP

LG = PP.libmonet_gdt('wallpaper_student')
sys.path.insert(0, LG)
import numpy as np
import torch
import timm  # noqa: E402
from multiprocessing import get_context  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out', 'teacher_colors')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
TX = PP.telosnex('dev', 'wallpaper_colors')
AB = 'monet_teacher_ab'
EPOCHS = 30
DEV = 'mps'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def hist_one(p):
    import hist_colors as HC

    return HC.hist(p)


def main():
    import hist_colors as HC
    import direct_colors as D
    import train_teacher as TT

    os.makedirs(OUT, exist_ok=True)
    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    byid = {m['id']: m for m in man}
    tri = {r['id']: r['triad'] for r in map(json.loads, open(os.path.join(OUT, 'rule_triads.jsonl')))}
    train = [m['id'] for m in man if m['split'] == 'train' and m['id'] in tri]
    ab = []
    for part, k in (('megalith', 30), ('generic', 15), ('genre', 15)):
        ab += sorted(
            (m['id'] for m in man if m['split'] == 'test' and m['part'] == part),
            key=lambda i: hashlib.sha256(f'teacherab:{i}'.encode()).hexdigest(),
        )[:k]
    os.makedirs(f'{TX}/{AB}', exist_ok=True)
    abname = {i: f"{byid[i]['part']}__{i}.jpg" for i in ab}
    for i in ab:
        dst = f'{TX}/{AB}/{abname[i]}'
        if not os.path.exists(dst):
            shutil.copy(os.path.join(ROOT, byid[i]['path']), dst)
    ids = train + ab
    paths = [os.path.join(ROOT, byid[i]['path']) for i in ids]
    log(f'{len(train)} training images, {len(ab)} A/B images (test split) -> {TX}/{AB}')

    # 1. color lists (fit224, as the shipped model)
    lp = os.path.join(OUT, 'lists_fit224.npz')
    if os.path.exists(lp) and list(np.load(lp)['ids']) == ids:
        z = np.load(lp)
        RGB, SH, K = z['rgb'], z['sh'], z['k']
        log('loaded color lists')
    else:
        RGB = np.zeros((len(ids), HC.MAXK, 3), np.uint8)
        SH = np.zeros((len(ids), HC.MAXK), np.float32)
        K = np.zeros(len(ids), np.int32)
        t0 = last = time.time()
        log(f'counting colors in {len(ids)} images (12 processes; estimate ~3 min)')
        with get_context('spawn').Pool(12) as pool:
            for i, (rgb, sh, _) in enumerate(pool.imap(hist_one, paths, chunksize=32)):
                RGB[i, : len(rgb)], SH[i, : len(sh)], K[i] = rgb, sh, len(rgb)
                if time.time() - last >= 10:
                    last = time.time()
                    log(
                        f'.. colors {i + 1}/{len(ids)}, elapsed {last - t0:.0f}s, remaining ~{(last - t0) / (i + 1) * (len(ids) - i - 1):.0f}s'
                    )
        np.savez(lp, ids=np.array(ids), rgb=RGB, sh=SH, k=K)
        log(f'saved {lp}')

    # 2. features: new teacher from the token cache; old teacher on the A/B images
    fp = os.path.join(OUT, 'feat_new.npy')
    if os.path.exists(fp):
        FN = np.load(fp)
        log('loaded new-teacher features')
    else:
        ck = torch.load(os.path.join(ROOT, 'out', 'teacher', 'teacher.pt'), weights_only=False)
        m = timm.create_model(ck['model_name'], pretrained=True, num_classes=0)
        m.load_state_dict(ck['backbone'])
        m = m.to(DEV).eval()
        c = torch.load(os.path.join(ROOT, 'out', 'teacher', 'cache_cut8.pt'))
        row = {i: k for k, i in enumerate(c['ids'])}
        X = c['x']
        cut = ck['trainable_from_block']
        mixed = getattr(m, 'rope_mixed', False)
        with torch.no_grad():
            _, rope = m._pos_embed(m.patch_embed(torch.zeros(1, 3, 384, 384, device=DEV)))
            out, t0, last = [], time.time(), time.time()
            ix = np.array([row[i] for i in ids])
            for s in range(0, len(ix), 128):
                x = X[torch.as_tensor(ix[s : s + 128])].to(DEV).float()
                for b in range(cut, len(m.blocks)):
                    x = m.blocks[b](x, rope=rope[b] if mixed else rope)
                out.append(
                    torch.nn.functional.normalize(m.forward_head(m.norm(x), pre_logits=True), dim=-1).cpu().numpy()
                )
                if time.time() - last >= 10:
                    last = time.time()
                    log(
                        f'.. new-teacher features {s + 128}/{len(ix)}, remaining ~{(last - t0) / (s + 128) * (len(ix) - s - 128):.0f}s'
                    )
        FN = np.concatenate(out)
        np.save(fp, FN)
        log('new-teacher features done')
    op = os.path.join(OUT, 'feat_old_ab.npy')
    if os.path.exists(op):
        FO = np.load(op)
    else:
        ck = torch.load(os.path.join(LG, 'out', 'mood_student', 'unfreeze', 'final_k4.pt'), weights_only=False)
        m = timm.create_model(ck['model_name'], pretrained=True, num_classes=0)
        m.load_state_dict(ck['backbone'])
        m = m.to(DEV).eval()
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
        dl = torch.utils.data.DataLoader(TT.Images(paths[len(train) :], tf), batch_size=30, num_workers=4)
        with torch.no_grad():
            FO = np.concatenate([torch.nn.functional.normalize(m(b.to(DEV)), dim=-1).cpu().numpy() for b in dl])
        np.save(op, FO)
        log('old-teacher features (A/B images) done')

    LAB = D.rgb2lab(RGB / 255).astype(np.float32) / 100

    def batch(rows, emb):
        k = int(K[rows].max())
        return (
            torch.from_numpy(LAB[rows, :k]).to(DEV),
            torch.from_numpy(SH[rows, :k]).to(DEV),
            torch.arange(k, device=DEV)[None] < torch.from_numpy(K[rows]).to(DEV)[:, None],
            emb.to(DEV),
        )

    HC.AREA = True

    # 3. train the color model on the new teacher (new file)
    mp = os.path.join(OUT, f'hist_fit224_newteacher_e{EPOCHS}.pt')
    net = HC.HistNet(192).to(DEV)
    if os.path.exists(mp):
        net.load_state_dict(torch.load(mp)['state_dict'])
        log(f'loaded {mp}')
    else:
        torch.manual_seed(0)
        nt = len(train)
        E = torch.tensor(FN[:nt], dtype=torch.float32)
        Y = torch.tensor(
            np.stack([D.rgb2lab([D.hex2rgb(h) for h in tri[i]]) for i in train]) / 100, dtype=torch.float32
        )
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
        t0 = last = time.time()
        log(f'training color model: {EPOCHS} epochs x {nt} images (estimate ~20 s/epoch)')
        for ep in range(1, EPOCHS + 1):
            net.train()
            tl = []
            for b in torch.randperm(nt).split(128):
                opt.zero_grad()
                loss = HC.loss_fn(net, *batch(b.numpy(), E[b]), Y[b].to(DEV), 0.0, 0.0, 1.0, True)
                loss.backward()
                opt.step()
                tl.append(loss.item())
                if time.time() - last >= 10:
                    last = time.time()
                    log(
                        f'.. epoch {ep}/{EPOCHS}, loss {np.mean(tl):.4f}, elapsed {last - t0:.0f}s, remaining ~{(last - t0) / (ep - 1 + len(tl) / (nt / 128)) * (EPOCHS - ep + 1 - len(tl) / (nt / 128)):.0f}s'
                    )
            log(f'epoch {ep} done: train loss {np.mean(tl):.4f}')
        torch.save(
            {
                'state_dict': net.state_dict(),
                'epochs': EPOCHS,
                'lists': 'fit224',
                'features': 'monet_gdt out/teacher/teacher.pt',
                'targets': 'rule_v1 (rule_triads.jsonl)',
                'train_ids': train,
            },
            mp,
        )
        log(f'saved {mp}')
    net.eval()

    # 4. picks on the A/B images, both sides
    old = HC.HistNet(192)
    old.load_state_dict(
        torch.load(
            os.path.join(LG, 'out', 'color_models', 'hist_fit224_conf_e30_seed0.pt'),
            map_location='cpu',
            weights_only=False,
        )['state_dict']
    )
    old = old.to(DEV).eval()
    rows = np.arange(len(train), len(ids))
    hexes = lambda r, idx: ['#%02x%02x%02x' % tuple(RGB[r, j]) for j in idx]
    res = {}
    with torch.no_grad():
        for arm, n_, F in (('old_teacher', old, FO), ('new_teacher', net, FN[len(train) :])):
            idx = HC.pick(n_, *batch(rows, torch.tensor(F, dtype=torch.float32))).cpu().numpy()
            res[arm] = {abname[i]: hexes(r, ii) for i, r, ii in zip(ab, rows, idx)}
    os.makedirs(f'{TX}/out/{AB}', exist_ok=True)
    raw = f'{TX}/out/{AB}/raw.jsonl'
    keep = [l for l in open(raw)] if os.path.exists(raw) else []
    keep = [l for l in keep if json.loads(l)['arm'] not in res]
    with open(raw, 'w') as f:
        f.writelines(keep)
        for arm, t in res.items():
            for n_, h in sorted(t.items()):
                f.write(json.dumps({'arm': arm, 'wallpaper': n_, 'triad': h}) + '\n')
    same = sum(res['old_teacher'][n_] == res['new_teacher'][n_] for n_ in res['old_teacher'])
    log(f'wrote {raw}: 2 arms x {len(ab)} images; identical triads {same}; all done')


if __name__ == '__main__':
    main()
