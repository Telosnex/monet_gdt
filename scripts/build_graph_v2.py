# Step 2 of the graph: the color step. The base graph
# (out/graph/monet_gdt_w8_shapes.onnx, from build_graph.py and
# add_shape_head.py) gives scores, feature, shapes and the color list. The
# base graph's `codes`, `share` and `feature` tensors feed the color step,
# and `colors` comes from it. Scores, feature and shapes are unchanged.
# Color step (ADR D12):
# 1. Head: out/v2/color_model.pt (scripts/train_colors_v2.py). Inputs per
#    color: CIELAB, chroma, share, and neighborhood share (list colors within
#    ΔE 10); plus the image feature.
# 2. Pick: per role, argmax of the head's logits over the image's color list.
#    S and T skip colors within ΔE 10 of an earlier pick.
# 3. Role order (MonetGdtRule20261030): S and T swap when S is the larger
#    step from P (HCT hue if all three are colorful, else ΔE76). The hue and
#    colorful tables come from monet_ui (out/v2/v2_hue.f32,
#    out/v2/v2_colorful.u8; tools/v2_labels/test/v2_tables_test.dart).
# The head weights are stored as fp16 and cast to fp32 at load.
#   .venv/bin/python -u scripts/build_graph_v2.py build | check
# Out: out/v2/monet_gdt_v2.onnx (the released graph), out/v2/check.json,
# out/v2/check_triads.jsonl (the graph's triads, for v2_order_test.dart).
# check: graph vs PyTorch (same picks) on N held-out matched images, timing vs
# the base graph, and that scores, feature and shapes equal the base graph's.
import hashlib
import json
import os
import sys
import time

for k, v in (('SINGLE', '1'), ('TWO', '1'), ('OPSET', '20')):
    os.environ.setdefault(k, v)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
import numpy as np
import onnx
import torch

import graph_front as OG

BASE = os.path.join(ROOT, 'out', 'graph', 'monet_gdt_w8_shapes.onnx')
HEAD = os.path.join(ROOT, 'out', 'v2', 'color_model.pt')
OUT = os.path.join(ROOT, 'out', 'v2', 'monet_gdt_v2.onnx')
EXCL = R_DE = 10.0  # ΔE: S/T exclusion; neighborhood radius


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


class Pick(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import hist_colors as HC

        ck = torch.load(HEAD, map_location='cpu', weights_only=False)
        assert ck['arm'] == 'direct' and ck['neigh']
        self.net = HC.HistNet(192, neigh=True)
        self.net.load_state_dict(ck['state_dict'])
        self.net.eval()
        # monet_ui tables (tools/v2_labels/test/v2_tables_test.dart) for the role step
        d = os.path.join(ROOT, 'out', 'v2')
        self.register_buffer('t_hue', torch.tensor(np.fromfile(os.path.join(d, 'v2_hue.f32'), np.float32)))
        # bool: 1 byte per color (ONNX stores bool as uint8)
        self.register_buffer('t_col', torch.tensor(np.fromfile(os.path.join(d, 'v2_colorful.u8'), np.uint8) > 0))

    def forward(self, codes, share, feature):
        mask = (share > 0)[None]
        rgb = torch.stack([codes // 1024, (codes // 32) % 32, codes % 32], -1).float() * 8 + 4
        lab = (OG.rgb2lab_t(rgb / 255) / 100)[None]  # (1, n, 3)
        sh = share[None]
        L = lab[0] * 100
        # neighborhood share: squared distances as one matmul (|a|^2 + |b|^2 - 2ab),
        # then a matmul with share; compares d^2 < R_DE^2 (no sqrt)
        sq = (L * L).sum(-1)
        d2 = sq[:, None] + sq[None] - 2.0 * (L @ L.T)  # (n, n)
        nb = ((d2 < R_DE * R_DE).float() @ share)[None] * mask  # (1, n)
        h, q = self.net.context(lab, sh, mask, feature, nb)
        refs, picks = [], []
        for r in range(3):
            lg = self.net.logits(h, q, r, lab, mask, *(refs + [None, None])[:2])
            ok = mask
            if r > 0:
                dmin = torch.stack([((lab - c[:, None]) ** 2).sum(-1).sqrt() * 100 for c in refs]).min(0).values
                ok = ok & (dmin >= EXCL)
            ho = ok.any(-1, keepdim=True)
            ok = (ok & ho) | (mask & ~ho)  # bool Where has no ORT CPU kernel
            i = lg.masked_fill(~ok, -1e9).argmax(-1)  # (1,)
            refs.append(lab[0, i])  # (1, 3)
            picks.append(i)
        idx = torch.cat(picks)  # (3,) positions in the list
        # MonetGdtRule20261030's role step: tertiary is the larger step from
        # primary, by HCT hue when all three are colorful, else by ΔE76.
        c = codes[idx]
        hue, col = self.t_hue[c], self.t_col[c]
        lab3 = lab[0, idx] * 100  # the CIELAB the head already uses
        dh = 180.0 - ((hue[1:] - hue[0]).abs() - 180.0).abs()
        de = ((lab3[1:] - lab3[0]) ** 2).sum(-1).sqrt()
        by_hue = col.all().float()  # 1.0 if all colorful
        st = dh * by_hue + de * (1.0 - by_hue)
        sw = (st[0] > st[1]).long()
        idx = torch.stack([idx[0], idx[1] * (1 - sw) + idx[2] * sw, idx[2] * (1 - sw) + idx[1] * sw])
        return rgb[idx].round().long()  # (3, 3) RGB int64


def export_pick(path):
    m = Pick().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    torch.backends.mha.set_fastpath_enabled(False)
    hyp = torch.hypot
    torch.hypot = lambda a, b: torch.sqrt(a * a + b * b)
    try:
        n0 = 900
        codes, share = torch.randperm(32768)[:n0], torch.rand(n0)
        feat = torch.nn.functional.normalize(torch.randn(1, 192), dim=-1)
        nd = torch.export.Dim('n', min=64, max=OG.K)
        prog = torch.onnx.export(
            m,
            (codes, share, feat),
            input_names=['codes', 'share', 'feature'],
            output_names=['colors'],
            dynamic_shapes={'codes': {0: nd}, 'share': {0: nd}, 'feature': None},
            opset_version=OG.OPSET,
            dynamo=True,
            optimize=True,
        )
        prog.save(path, external_data=False)
    finally:
        torch.hypot = hyp
    p = onnx.load(path)
    p.ir_version = OG.IR
    for nd_ in p.graph.node:
        nd_.doc_string = ''
        del nd_.metadata_props[:]
    del p.graph.value_info[:]
    del p.metadata_props[:]
    return p


def prune(g):
    """Drop nodes and initializers that no graph output depends on."""
    need = {o.name for o in g.output}
    keep = []
    for nd in reversed(list(g.node)):
        if any(o in need for o in nd.output):
            keep.append(nd)
            need |= set(nd.input)
            for at in nd.attribute:
                if at.g.node:
                    for sn in at.g.node:
                        need |= set(sn.input)
    del g.node[:]
    g.node.extend(reversed(keep))
    inits = [i for i in g.initializer if i.name in need]
    del g.initializer[:]
    g.initializer.extend(inits)


def build():
    from onnx import compose

    t = time.time()
    base = onnx.load(BASE)
    # rename the old colors tensor; expose codes/share/feature for the merge
    for nd in base.graph.node:
        nd.output[:] = ['colors_rule_v1' if x == 'colors' else x for x in nd.output]
        nd.input[:] = ['colors_rule_v1' if x == 'colors' else x for x in nd.input]
    outs = [o for o in base.graph.output if o.name != 'colors']
    del base.graph.output[:]
    base.graph.output.extend(outs)
    for t_ in ('codes', 'share'):
        base.graph.output.append(onnx.helper.make_empty_tensor_value_info(t_))
    pick = export_pick(os.path.join(ROOT, 'out', 'v2', 'pick.onnx'))
    # head weights stored fp16, cast to fp32 at load (as build_graph.py's
    # fp16_store); the 32,768-entry hue table stays fp32
    hue = {i.name for i in pick.graph.initializer if 32768 in i.dims}
    keep = [i for i in pick.graph.initializer if i.name in hue]
    rest = [i for i in pick.graph.initializer if i.name not in hue]
    del pick.graph.initializer[:]
    pick.graph.initializer.extend(rest)
    pick = OG.fp16_store(pick)
    pick.graph.initializer.extend(keep)
    pick = compose.add_prefix(pick, 'pb_')
    log(f'pick step exported ({len(pick.graph.node)} nodes)')
    m = compose.merge_models(
        base, pick, io_map=[('codes', 'pb_codes'), ('share', 'pb_share'), ('feature', 'pb_feature')]
    )
    for nd in m.graph.node:
        nd.output[:] = ['colors' if x == 'pb_colors' else x for x in nd.output]
    feat_vi = next(o for o in base.graph.output if o.name == 'feature')
    if not any(o.name == 'feature' for o in m.graph.output):  # merge_models drops mapped outputs
        m.graph.output.append(feat_vi)
    order = {'scores': 0, 'feature': 1, 'colors': 2, 'shapes': 3}
    outs = [o for o in m.graph.output if o.name in order or o.name == 'pb_colors']
    for o in outs:
        if o.name == 'pb_colors':
            o.name = 'colors'
    outs.sort(key=lambda o: order[o.name])
    del m.graph.output[:]
    m.graph.output.extend(outs)
    prune(m.graph)
    m = OG.dedup(m)
    import graph_meta as GM

    shapes = {p_.key: p_.value for p_ in base.metadata_props}['monet_gdt.shapes'].split(',')
    GM.apply(m, shapes)  # this release's version; name, words and shapes from the base
    onnx.checker.check_model(m)
    onnx.save(m, OUT)
    os.remove(os.path.join(ROOT, 'out', 'v2', 'pick.onnx'))
    log(
        f'built {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB); '
        f'outputs {[o.name for o in m.graph.output]}; {time.time() - t:.0f}s'
    )


def check():
    import onnxruntime as ort
    from PIL import Image

    import graph_meta as GM

    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import train_colors_v2 as T

    n = int(os.environ.get('N', 300))
    ids = json.load(open(os.path.join(ROOT, 'out', 'v2', 'ids.json')))
    hold = [m for m in ids if m['src'] == 'matched' and T.holdout(m['id'])][:n]
    so = ort.SessionOptions()
    old = ort.InferenceSession(BASE, so, providers=['CPUExecutionProvider'])
    base = onnx.load(BASE)
    for t_ in ('codes', 'share'):
        base.graph.output.append(onnx.helper.make_empty_tensor_value_info(t_))
    oldx = ort.InferenceSession(base.SerializeToString(), so, providers=['CPUExecutionProvider'])
    new = ort.InferenceSession(OUT, so, providers=['CPUExecutionProvider'])
    pick = Pick().eval()
    torch.backends.mha.set_fastpath_enabled(False)
    same_pt = same_rest = 0
    t_old, t_new = [], []
    triads = open(os.path.join(ROOT, 'out', 'v2', 'check_triads.jsonl'), 'w')
    for k, m in enumerate(hold):
        a = np.asarray(Image.open(os.path.join(ROOT, m['path'])).convert('RGBA'))
        b = time.perf_counter()
        ro = old.run(None, {'image': a})
        t_old.append(time.perf_counter() - b)
        b = time.perf_counter()
        rn = new.run(None, {'image': a})
        t_new.append(time.perf_counter() - b)
        _, feat, _, _, codes, share = oldx.run(None, {'image': a})
        with torch.no_grad():
            pc = pick(torch.tensor(codes), torch.tensor(share), torch.tensor(feat)).numpy()
        same_pt += np.array_equal(rn[2], pc)
        same_rest += all(np.array_equal(ro[i], rn[i]) for i in (0, 1, 3))
        hexes = ['#%02x%02x%02x' % tuple(int(x) for x in c) for c in rn[2]]
        triads.write(json.dumps({'id': m['id'], 'triad': hexes}) + '\n')
    triads.close()
    rep = {
        'n': len(hold),
        'colors_equal_pytorch': f'{same_pt}/{len(hold)}',
        'scores_feature_shapes_identical_to_base': f'{same_rest}/{len(hold)}',
        'ms_median_base': round(float(np.median(t_old)) * 1000, 1),
        'ms_median_graph': round(float(np.median(t_new)) * 1000, 1),
        'mb': round(os.path.getsize(OUT) / 1e6, 2),
        'sha256': hashlib.sha256(open(OUT, 'rb').read()).hexdigest(),
        'head_sha256': hashlib.sha256(open(HEAD, 'rb').read()).hexdigest(),
        'version': GM.VERSION,
    }
    json.dump(rep, open(os.path.join(ROOT, 'out', 'v2', 'check.json'), 'w'), indent=1)
    log(json.dumps(rep))


if __name__ == '__main__':
    {'build': build, 'check': check}[sys.argv[1]]()
