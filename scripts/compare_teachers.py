# ADR D8: old teacher (libmonet_gdt final_k4.pt, the shipped build) vs the
# new teacher (out/teacher/teacher.pt) on the same images neither trained
# on: libmonet_gdt's held-out Unsplash + Wallhaven keys (heldout_keys.json;
# 0 overlap with final_k4's train_names, checked). Labels: the old Jev mood
# labels for those images (same prompt; mood repeats at r 0.998).
# Each teacher uses its own quantile map (its training predictions).
#   .venv/bin/python -u scripts/compare_teachers.py -> out/teacher/compare_d8.json
import importlib.util
import json
import os
import sys
import time
import numpy as np
import torch
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import train_teacher as TT  # noqa: E402  (WORDS, qmap_fit, font_scores, Images)
import private_paths as PP

LG = PP.libmonet_gdt('wallpaper_student')
sys.path.insert(0, LG)
spec = importlib.util.spec_from_file_location('mood_student', os.path.join(LG, 'mood_student.py'))
MS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MS)
DEV = 'mps'


def main():
    keys = set(json.load(open(os.path.join(LG, 'out', 'mood_student', 'heldout_keys.json'))))
    rows = [r for r in MS.labeled() if r['name'] in keys]
    Y = np.array([[r['raw'][w] for w in TT.WORDS] for r in rows], np.float32)
    src = np.array([r['name'].split('__')[0] for r in rows])
    TT.log(f'{len(rows)} held-out images: ' + ', '.join(f'{s} {np.sum(src == s)}' for s in sorted(set(src))))
    old = torch.load(os.path.join(LG, 'out', 'mood_student', 'unfreeze', 'final_k4.pt'), weights_only=False)
    new = torch.load(os.path.join(ROOT, 'out', 'teacher', 'teacher.pt'), weights_only=False)
    assert not keys & set(old['train_names']), 'held-out image in the old training set'
    out = {'n': {s: int(np.sum(src == s)) for s in sorted(set(src))}}
    JS = [TT.font_scores(y) for y in Y]
    for name, ck in (('old_final_k4', old), ('new_teacher', new)):
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
        cache = os.path.join(ROOT, 'out', 'teacher', f'd8_pred_{name}.npy')
        t = time.time()
        if os.path.exists(cache):
            P = [np.load(cache)]
            TT.log(f'{name}: loaded saved predictions')
        else:
            P, done, last = [], 0, time.time()
            TT.log(f'{name}: scoring {len(rows)} images (estimate ~1 min)')
            with torch.no_grad():
                for b in dl:
                    P.append(head((m(b.to(DEV)) - mu) / sd).cpu().numpy())
                    done += len(b)
                    if time.time() - last >= 10:
                        last = time.time()
                        TT.log(
                            f'.. {name}: {done}/{len(rows)}, elapsed {time.time() - t:.0f}s, remaining ~{(time.time() - t) / done * (len(rows) - done):.0f}s'
                        )
            np.save(cache, np.concatenate(P))
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
                'rank_median': float(np.median(rk)),
            }
        out[name] = res
        TT.log(
            f'{name} ({time.time() - t:.0f}s): '
            + '; '.join(f"{s} r {v['r']:.3f} top5% {v['top5pct']:.2f}" for s, v in res.items())
        )
    json.dump(out, open(os.path.join(ROOT, 'out', 'teacher', 'compare_d8.json'), 'w'), indent=1)
    o, n = out['old_final_k4']['all']['per_word'], out['new_teacher']['all']['per_word']
    print('per word (old -> new):', ', '.join(f'{w} {o[w]:.2f}->{n[w]:.2f}' for w in TT.WORDS))


if __name__ == '__main__':  # macOS DataLoader workers re-import this file
    main()
