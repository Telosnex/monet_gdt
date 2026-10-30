# ADR D5 step 6: the on-device graph, with the shipped graph's front end and
# settings (lib/graph_front.py, from libmonet_gdt onnx_graph.py; the app's file is
# wallpaper12_o20_fr_cg_c1x10_w8_shapes.onnx: SINGLE, TWO, opset 20, fast
# RoPE, backbone MatMuls 8-bit block 32, large weights stored fp16).
# Core: out/$STUDENT/student.pt (default student_aug, the crop student) + the
# teacher's font head (out/teacher).
# Outputs: scores (1, 10), feature (1, 192), colors (3, 3) int64 RGB;
# `shapes` is added by add_shape_head.py. This is the base graph: its colors
# are the three largest colors of the image's color list, and
# build_graph_v2.py replaces them with the color step (ADR D12).
#   .venv/bin/python -u scripts/build_graph.py build | check
# check: on the 1,342 test images (long side <= 2048, Lanczos, RGBA as the
#   app decodes): fp32 graph vs PyTorch on the graph's own front-end
#   output (colors equal, scores |diff|), w8 vs fp32 (same triad, same
#   font), timing. Writes out/graph/check.json. Logs every 10 s.
import hashlib
import json
import os
import sys
import time

for k, v in (('SINGLE', '1'), ('TWO', '1'), ('OPSET', '20'), ('FASTROPE', '1')):
    os.environ.setdefault(k, v)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
import numpy as np
import torch
import timm  # noqa: E402
import graph_front as OG  # noqa: E402  (front(), fast_rope, settings from the env above)

OUT = os.path.join(ROOT, 'out', 'graph')
NAME = 'monet_gdt'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sk = torch.load(
            os.path.join(ROOT, 'out', os.environ.get('STUDENT', 'student_aug'), 'student.pt'), weights_only=False
        )
        self.bb = timm.create_model(
            sk['model_name'], pretrained=False, num_classes=0, img_size=sk['res'], depth=sk['depth']
        )
        self.bb.load_state_dict(sk['backbone'])
        self.bb.eval()
        tk = torch.load(os.path.join(ROOT, 'out', 'teacher', 'teacher.pt'), weights_only=False)
        self.l1, self.l2 = torch.nn.Linear(192, 256), torch.nn.Linear(256, 10)
        self.l1.load_state_dict({'weight': tk['head']['1.weight'], 'bias': tk['head']['1.bias']})
        self.l2.load_state_dict({'weight': tk['head']['4.weight'], 'bias': tk['head']['4.bias']})
        self.register_buffer('mu', torch.as_tensor(tk['feat_mean']).float().reshape(1, -1))
        self.register_buffer('sd', torch.as_tensor(tk['feat_std']).float().reshape(1, -1))

    @staticmethod
    def largest(codes, share):
        """Placeholder colors: the three colors with the largest share, RGB (3, 3)."""
        c = codes[torch.topk(share, 3).indices]
        return (torch.stack([c // 1024, (c // 32) % 32, c % 32], -1) * 8 + 4).long()

    def forward(self, o_img, codes, share):
        f = self.bb(o_img / 127.5 - 1)
        scores = torch.sigmoid(self.l2(torch.nn.functional.gelu(self.l1((f - self.mu) / self.sd))))
        feat = torch.nn.functional.normalize(f, dim=-1)
        return scores, feat, self.largest(codes, share)


def export_core(path):
    import onnx

    core = Core().eval()
    for p in core.parameters():
        p.requires_grad_(False)
    x = torch.rand(2, 3, 224, 224) * 2 - 1
    with torch.no_grad():
        ref = core.bb(x)
    OG.fast_rope(core.bb)
    with torch.no_grad():
        log(f'fast RoPE: max feature difference {(core.bb(x) - ref).abs().max().item():.2e}')
    torch.backends.mha.set_fastpath_enabled(False)
    hyp = torch.hypot
    torch.hypot = lambda a, b: torch.sqrt(a * a + b * b)
    try:
        o = torch.zeros(1, 3, 224, 224)
        n0 = 900
        codes = torch.randperm(32768)[:n0]
        share = torch.rand(n0)
        share[-100:] = 0
        nd = torch.export.Dim('n', min=64, max=OG.K)
        prog = torch.onnx.export(
            core,
            (o, codes, share),
            input_names=['o_img', 'codes', 'share'],
            output_names=['scores', 'feature', 'colors'],
            dynamic_shapes={'o_img': None, 'codes': {0: nd}, 'share': {0: nd}},
            opset_version=OG.OPSET,
            dynamo=True,
            optimize=True,
        )
        prog.save(path, external_data=False)
    finally:
        torch.hypot = hyp
    m = onnx.load(path)
    m.ir_version = OG.IR
    for nd in m.graph.node:
        nd.doc_string = ''
        del nd.metadata_props[:]
    del m.graph.value_info[:]
    del m.metadata_props[:]
    m.graph.doc_string = ''
    onnx.save(m, path)
    return core


def build():
    import onnx
    from onnx import compose, numpy_helper as nh

    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    core_path = os.path.join(OUT, 'core.onnx')
    core_t = export_core(core_path)
    log(f'core exported in {time.time() - t:.0f}s')
    core = onnx.load(core_path)
    fr = OG.front()

    def names(g):
        out = {i.name for i in g.initializer}
        for nd in g.node:
            out |= set(nd.input) | set(nd.output)
            for at in nd.attribute:
                if at.g.node:
                    out |= names(at.g)
        return out

    io = {'image', 'o_img', 'codes', 'share', ''}
    clash = {x: 'fe_' + x for x in names(fr.graph) & names(core.graph) if x not in io}

    def ren(g):
        for i in g.initializer:
            i.name = clash.get(i.name, i.name)
        for nd in g.node:
            nd.input[:] = [clash.get(x, x) for x in nd.input]
            nd.output[:] = [clash.get(x, x) for x in nd.output]
            for at in nd.attribute:
                if at.g.node:
                    ren(at.g)

    ren(fr.graph)
    m = OG.dedup(compose.merge_models(fr, core, io_map=[('o_img', 'o_img'), ('codes', 'codes'), ('share', 'share')]))
    onnx.checker.check_model(m)
    fp32 = os.path.join(OUT, f'{NAME}_fp32.onnx')
    onnx.save(m, fp32)
    os.remove(core_path)
    log(f'built {fp32} ({os.path.getsize(fp32) / 1e6:.1f} MB)')
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

    hs = {
        hashlib.sha1(np.ascontiguousarray(v.detach().numpy().T).tobytes()).hexdigest()
        for k, v in core_t.bb.state_dict().items()
        if k.startswith('blocks.') and k.endswith('.weight') and v.ndim == 2
    }
    inits = {i.name: i for i in m.graph.initializer}

    def is_bb(x):
        i = inits.get(x.input[1])
        return i is not None and hashlib.sha1(np.ascontiguousarray(nh.to_array(i)).tobytes()).hexdigest() in hs

    keep = [x.name for x in m.graph.node if x.op_type == 'MatMul' and not is_bb(x)]
    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8)
    q = MatMulNBitsQuantizer(onnx.load(fp32), block_size=32, is_symmetric=True, algo_config=cfg, nodes_to_exclude=keep)
    q.process()
    w8 = os.path.join(OUT, f'{NAME}_w8.onnx')
    onnx.save(OG.fp16_store(OG.dedup(q.model.model)), w8)
    nq = sum(1 for x in onnx.load(w8).graph.node if x.op_type == 'MatMulNBits')
    log(f'built {w8} ({os.path.getsize(w8) / 1e6:.2f} MB); backbone MatMuls quantized: {nq}')


def load_rgba(path):
    from PIL import Image

    im = Image.open(path).convert('RGBA')
    im.thumbnail((2048, 2048), Image.LANCZOS)
    return np.asarray(im)


def check():
    import onnxruntime as ort
    import onnx
    import train_teacher as TT
    import mood_words as VM

    man = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
    test = [m for m in man if m['split'] == 'test'][: int(os.environ.get('N', 300))]
    fp32, w8 = (os.path.join(OUT, f'{NAME}_{v}.onnx') for v in ('fp32', 'w8'))
    fe = os.path.join(OUT, 'front.onnx')
    onnx.save(OG.front(), fe)
    sF, s32, s8 = (ort.InferenceSession(p, providers=['CPUExecutionProvider']) for p in (fe, fp32, w8))
    core = Core().eval()
    OG.fast_rope(core.bb)
    torch.backends.mha.set_fastpath_enabled(False)
    tk = torch.load(os.path.join(ROOT, 'out', 'teacher', 'teacher.pt'), weights_only=False)
    q = TT.qmap_fit(np.asarray(tk['qmap_train_pred']), np.asarray(tk['qmap_train_y']))
    fm = json.load(open(os.path.join(ROOT, 'fonts', 'mood_font_model.json')))
    font = lambda s: VM.headline(fm, dict(zip(VM.WORDS, q(np.asarray(s, np.float64)[None])[0])))
    same_pt = same_w8 = same_font = 0
    dmax = 0.0
    t0 = last = time.time()
    ms = []
    for k, m in enumerate(test):
        a = load_rgba(os.path.join(ROOT, m['path']))
        o_img, codes, share = sF.run(None, {'image': a})
        with torch.no_grad():
            ps, _, pc = core(torch.tensor(o_img), torch.tensor(codes), torch.tensor(share))
        s1, _, c1 = s32.run(None, {'image': a})
        b = time.perf_counter()
        s2, _, c2 = s8.run(None, {'image': a})
        ms.append((time.perf_counter() - b) * 1000)
        same_pt += np.array_equal(c1, pc.numpy())
        dmax = max(dmax, float(np.abs(s1 - ps.numpy()).max()))
        same_w8 += np.array_equal(c1, c2)
        same_font += font(s1[0]) == font(s2[0])
        if time.time() - last >= 10:
            last = time.time()
            log(f'.. check {k + 1}/{len(test)}, remaining ~{(last - t0) / (k + 1) * (len(test) - k - 1):.0f}s')
    n = len(test)
    rep = {
        'n': n,
        'fp32_colors_equal_pytorch': f'{same_pt}/{n}',
        'fp32_scores_max_abs_diff_pytorch': dmax,
        'w8_colors_equal_fp32': f'{same_w8}/{n}',
        'w8_font_equal_fp32': f'{same_font}/{n}',
        'w8_ms_median_cpu_default_threads': round(float(np.median(ms)), 1),
        'mb_w8': round(os.path.getsize(w8) / 1e6, 2),
        'sha256_w8': hashlib.sha256(open(w8, 'rb').read()).hexdigest(),
    }
    json.dump(rep, open(os.path.join(OUT, 'check.json'), 'w'), indent=1)
    log(json.dumps(rep))


if __name__ == '__main__':
    {'build': build, 'check': check}[sys.argv[1]]()
