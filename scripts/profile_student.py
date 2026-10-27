# Profile one crop-student training step (2026-10-25): where the
# time goes, and what reduced precision does to the teacher's outputs.
# Real images (64, decoded as train_student.Files at 384), real teacher
# and student weights. Each stage timed alone (MPS synchronized), 3 warmup
# + 10 timed iterations. Data loading timed separately (10 workers).
#   .venv/bin/python -u scripts/profile_student.py
import json
import os
import sys
import time
import torch
import timm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'lib'), HERE]
DEV = 'mps'
BS = 64


def sync():
    torch.mps.synchronize()


def timed(f, n=10, warm=3):
    for _ in range(warm):
        f()
    sync()
    t = time.perf_counter()
    for _ in range(n):
        f()
    sync()
    return (time.perf_counter() - t) / n * 1000


def main():
    import train_student as TS
    import augment as SM

    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    paths = [os.path.join(ROOT, m['path']) for m in man if m['split'] == 'train']
    # data loading alone: 20 batches at 384, 10 workers (as training)
    dl = torch.utils.data.DataLoader(TS.Files(paths[: BS * 23], 384), batch_size=BS, num_workers=10)
    it = iter(dl)
    [next(it) for _ in range(3)]
    t = time.perf_counter()
    U = None
    for _ in range(20):
        U = next(it)
    load_ms = (time.perf_counter() - t) / 20 * 1000
    ck = torch.load(os.path.join(ROOT, 'out', 'teacher', 'teacher.pt'), weights_only=False)
    tm = timm.create_model(ck['model_name'], pretrained=True, num_classes=0)
    tm.load_state_dict(ck['backbone'])
    tm = tm.to(DEV).eval()
    sk = torch.load(os.path.join(ROOT, 'out', 'student_aug', 'student.pt'), weights_only=False)
    stu = timm.create_model(sk['model_name'], pretrained=False, num_classes=0, img_size=sk['res'], depth=sk['depth'])
    stu.load_state_dict(sk['backbone'])
    stu = stu.to(DEV).train()
    opt = torch.optim.AdamW(stu.parameters(), 1e-4, weight_decay=0.01)
    mu, sd = ck['feat_mean'].to(DEV).reshape(1, -1), ck['feat_std'].to(DEV).reshape(1, -1)
    g = torch.Generator().manual_seed(0)
    fl = torch.rand(BS, generator=g) < 0.5
    xn = TS.norm(U).to(DEV)
    xt = SM.crops(xn, fl, g)
    with torch.no_grad():
        tgt = tm(xt)
    xs = torch.nn.functional.interpolate(xt, size=(224, 224), mode='bicubic', antialias=True, align_corners=False)
    res = {'batch': BS, 'load_ms_per_batch_10_workers': load_ms}
    res['h2d_and_norm_ms'] = timed(lambda: TS.norm(U).to(DEV))
    res['crops_ms'] = timed(lambda: SM.crops(xn, fl, g))

    def teach(dtype=None):
        with torch.no_grad():
            if dtype is None:
                return tm(xt)
            with torch.autocast('mps', dtype=dtype):
                return tm(xt)

    res['teacher_fp32_ms'] = timed(teach)
    res['resize_224_ms'] = timed(
        lambda: torch.nn.functional.interpolate(
            xt, size=(224, 224), mode='bicubic', antialias=True, align_corners=False
        )
    )

    def step(dtype=None):
        def f():
            if dtype is None:
                zs = (stu(xs) - mu) / sd
            else:
                with torch.autocast('mps', dtype=dtype):
                    zs = (stu(xs).float() - mu) / sd
            loss = torch.nn.functional.mse_loss(zs, (tgt - mu) / sd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        return f

    res['student_step_fp32_ms'] = timed(step())
    for name, dt in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        try:
            res[f'teacher_{name}_ms'] = timed(lambda: teach(dt))
            o = teach(dt).float()
            z32, zlo = (tgt - mu) / sd, (o - mu) / sd
            res[f'teacher_{name}_feature_cos_min'] = float(torch.nn.functional.cosine_similarity(z32, zlo).min())
            res[f'teacher_{name}_feature_max_abs_z'] = float((z32 - zlo).abs().max())
        except Exception as e:
            res[f'teacher_{name}'] = f'{type(e).__name__}: {str(e)[:80]}'
        try:
            res[f'student_step_{name}_ms'] = timed(step(dt))
        except Exception as e:
            res[f'student_step_{name}'] = f'{type(e).__name__}: {str(e)[:80]}'
    gpu = (
        res['h2d_and_norm_ms']
        + res['crops_ms']
        + res['teacher_fp32_ms']
        + res['resize_224_ms']
        + res['student_step_fp32_ms']
    )
    res['gpu_total_fp32_ms'] = gpu
    res['share'] = {
        k: round(res[k] / gpu, 3)
        for k in ('h2d_and_norm_ms', 'crops_ms', 'teacher_fp32_ms', 'resize_224_ms', 'student_step_fp32_ms')
    }
    res['observed_training_ms_per_batch'] = 1000 / 4.3
    json.dump(res, open(os.path.join(ROOT, 'out', 'profile_student.json'), 'w'), indent=1)
    for k, v in res.items():
        print(f'{k:40s} {v if isinstance(v, (str, dict)) else round(v, 4)}')


if __name__ == '__main__':
    main()
