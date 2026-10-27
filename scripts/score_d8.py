# D8 score for any teacher checkpoint (same method as compare_teachers.py:
# its own quantile map; libmonet_gdt held-out Unsplash + Wallhaven, old Jev
# labels). Writes <dir>/d8.json next to each teacher.pt.
#   <venv>/bin/python -u scripts/score_d8.py out/curves/teacher_3154 [...]
import json
import os
import sys
import time
import numpy as np
import torch
import timm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import private_paths as PP

LG = PP.libmonet_gdt('wallpaper_student')
DEV = 'mps'


def main():
    import train_teacher as TT

    sys.path.insert(0, LG)
    import mood_student as MS

    keys = set(json.load(open(os.path.join(LG, 'out', 'mood_student', 'heldout_keys.json'))))
    rows = [r for r in MS.labeled() if r['name'] in keys]
    Y = np.array([[r['raw'][w] for w in TT.WORDS] for r in rows], np.float32)
    src = np.array([r['name'].split('__')[0] for r in rows])
    JS = [TT.font_scores(y) for y in Y]
    for d in sys.argv[1:]:
        ck = torch.load(os.path.join(d, 'teacher.pt'), weights_only=False)
        t = time.time()
        m = timm.create_model(ck['model_name'], pretrained=True, num_classes=0)
        m.load_state_dict(ck['backbone'])
        m = m.to(DEV).eval()
        head = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(192, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 10),
            torch.nn.Sigmoid(),
        )
        head.load_state_dict(ck['head'])
        head = head.to(DEV).eval()
        mu, sd = ck['feat_mean'].to(DEV), ck['feat_std'].to(DEV)
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
        dl = torch.utils.data.DataLoader(TT.Images([r['path'] for r in rows], tf), batch_size=32, num_workers=8)
        P, done, last = [], 0, time.time()
        with torch.no_grad():
            for b in dl:
                P.append(head((m(b.to(DEV)) - mu) / sd).cpu().numpy())
                done += len(b)
                if time.time() - last >= 10:
                    last = time.time()
                    TT.log(f'.. {d}: D8 {done}/{len(rows)}, remaining ~{(last - t) / done * (len(rows) - done):.0f}s')
        P = TT.qmap_fit(np.asarray(ck['qmap_train_pred']), np.asarray(ck['qmap_train_y']))(np.concatenate(P))
        res = {}
        for s in ['all'] + sorted(set(src)):
            ix = np.arange(len(rows)) if s == 'all' else np.where(src == s)[0]
            r = [float(np.corrcoef(P[ix, k], Y[ix, k])[0, 1]) for k in range(10)]
            rk = np.array([np.mean(JS[i] > JS[i][int(np.argmax(TT.font_scores(P[i])))]) for i in ix])
            res[s] = {
                'r': float(np.mean(r)),
                'per_word': dict(zip(TT.WORDS, np.round(r, 3).tolist())),
                'top5pct': float(np.mean(rk <= 0.05)),
            }
        json.dump(res, open(os.path.join(d, 'd8.json'), 'w'), indent=1)
        TT.log(f'{d}: D8 r {res["all"]["r"]:.3f}, top5% {res["all"]["top5pct"]:.2f} ({time.time() - t:.0f}s)')


if __name__ == '__main__':  # macOS DataLoader workers re-import this file
    main()
