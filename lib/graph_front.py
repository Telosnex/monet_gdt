"""The graph's front end and export helpers (dedup, fp16_store).

front(): image uint8 (H, W, C), C = 4 (RGBA, alpha ignored) or 3 (RGB), any size ->
  o_img (1, 3, 224, 224): short side 224, center crop (backbone input)
  codes, share (n,): the color list, 5-bit RGB codes and pixel shares, top K by count
  (ties: lower code first), cut to the colors that exist (at least 64).
Resize: ONNX opset OPSET, cubic, antialias, a = -0.5, exclude_outside, half_pixel,
rounded and clipped to 0-255.
Settings, read from the environment at import: OPSET (default 18), SINGLE=1 (the color
list's image is resized from the backbone's, not the full image), TWO=1 (average-pool by
a whole factor, short side stays >= 448, before Resize). The shipped graph uses
SINGLE=1 TWO=1 OPSET=20."""

import os
import numpy as np
import torch

OPSET = int(os.environ.get('OPSET', '18'))
IR = 9 if OPSET <= 20 else 10
SINGLE = os.environ.get('SINGLE') == '1'  # N resized from O's 398 x 224 image, not the full image
TWO = os.environ.get('TWO') == '1'  # average-pool by a whole factor (short side stays >= 448) before Resize
K = 2048


def front():
    from onnx import helper as H, TensorProto as TP

    n, inits = [], []

    def c(name, v, dt=np.int64):
        inits.append(
            H.make_tensor(
                name,
                TP.INT64 if dt == np.int64 else TP.FLOAT,
                [len(v)] if isinstance(v, list) else [],
                v if isinstance(v, list) else [v],
            )
        )
        return name

    def op(t, ins, out, **kw):
        n.append(H.make_node(t, ins, [out], **kw))
        return out

    c('i0', [0])
    c('i1', [1])
    c('i224', 224)
    c('one', 1)
    c('i2', 2)
    c('k13', [1, 3])
    c('f0', 0.0, np.float32)
    c('f255', 255.0, np.float32)
    c('f2', 2.0, np.float32)
    c('f224', 224.0, np.float32)
    c('c0', [0])
    c('c3', [3])
    # transpose the uint8 first: then alpha is one contiguous plane and dropping it is cheap
    op('Transpose', ['image'], 'pre_t', perm=[2, 0, 1])  # (C, H, W)
    op('Slice', ['pre_t', 'c0', 'c3', 'c0'], 'pre_rgb')  # RGBA or RGB -> RGB
    op('Cast', ['pre_rgb'], 'pre_chw', to=TP.FLOAT)
    op('Unsqueeze', ['pre_chw', 'i0'], 'pre_x')
    op('Shape', ['image'], 'pre_shape')
    op('Gather', ['pre_shape', 'i0'], 'pre_h1')
    op('Gather', ['pre_shape', 'i1'], 'pre_w1')
    op('Squeeze', ['pre_h1'], 'h')
    op('Squeeze', ['pre_w1'], 'w')
    op('Mul', ['i224', 'h'], 'h224')
    op('Mul', ['i224', 'w'], 'w224')
    op('Div', ['h224', 'w'], 'h_by_w')
    op('Div', ['w224', 'h'], 'w_by_h')  # int64 floor (all positive)
    op('Max', ['h_by_w', 'one'], 'h_by_w1')
    op('Max', ['w_by_h', 'one'], 'w_by_h1')
    op('GreaterOrEqual', ['w', 'h'], 'wide')  # w >= h
    op('LessOrEqual', ['w', 'h'], 'tall')  # w <= h
    # N: long side 224. (tw, th) = (224, max(1, 224h//w)) if w >= h else (max(1, 224w//h), 224)
    op('Where', ['wide', 'i224', 'w_by_h1'], 'n_tw')
    op('Where', ['wide', 'h_by_w1', 'i224'], 'n_th')
    # O: short side 224. (tw, th) = (224, 224h//w) if w <= h else (224w//h, 224)
    op('Where', ['tall', 'i224', 'w_by_h'], 'o_tw')
    op('Where', ['tall', 'h_by_w', 'i224'], 'o_th')
    src_full = 'pre_x'
    if TWO:
        # f = max(1, min(h, w) // 448); crop to multiples of f (top-left); mean over f x f blocks
        c('i448', 448)
        c('ax35', [3, 5])
        c('i00', [0, 0])
        c('k1', [1])
        c('k3', [3])
        op('Min', ['h', 'w'], 'short')
        op('Div', ['short', 'i448'], 'pf0')
        op('Max', ['pf0', 'one'], 'f')
        op('Div', ['h', 'f'], 'h2')
        op('Div', ['w', 'f'], 'w2')
        op('Mul', ['h2', 'f'], 'hc')
        op('Mul', ['w2', 'f'], 'wc')
        op('Unsqueeze', ['hc', 'i0'], 'hc1')
        op('Unsqueeze', ['wc', 'i0'], 'wc1')
        op('Concat', ['hc1', 'wc1'], 'cend', axis=0)
        for v in ('h2', 'w2', 'f'):
            op('Unsqueeze', [v, 'i0'], f'{v}_1')
        op('Concat', ['k1', 'k3', 'h2_1', 'f_1', 'w2_1', 'f_1'], 'shape6', axis=0)
        # Slice/Reshape/ReduceMean inside an If: no full-image copy when f == 1
        vi = lambda nm: H.make_tensor_value_info(nm, TP.FLOAT, None)
        then_g = H.make_graph(
            [
                H.make_node('Slice', ['pre_x', 'i00', 'cend', 'ax23_'], ['x_crop']),
                H.make_node('Reshape', ['x_crop', 'shape6'], ['x6']),
                H.make_node('ReduceMean', ['x6', 'ax35'], ['x_pooled'], keepdims=0),
            ],
            'pool',
            [],
            [vi('x_pooled')],
        )
        else_g = H.make_graph([H.make_node('Identity', ['pre_x'], ['x_same'])], 'nopool', [], [vi('x_same')])
        op('Greater', ['f', 'one'], 'do_pool')
        n.append(H.make_node('If', ['do_pool'], ['x_pool'], then_branch=then_g, else_branch=else_g))
        src_full = 'x_pool'
        c('ax23_', [2, 3])
    for p in ('o', 'n'):  # O first: SINGLE=1 makes N from it
        op('Unsqueeze', [f'{p}_th', 'i0'], f'{p}_th1')
        op('Unsqueeze', [f'{p}_tw', 'i0'], f'{p}_tw1')
        op('Concat', ['k13', f'{p}_th1', f'{p}_tw1'], f'{p}_sizes', axis=0)
        src = 'o_full' if (p == 'n' and SINGLE) else src_full
        op(
            'Resize',
            [src, '', '', f'{p}_sizes'],
            f'{p}_rs',
            mode='cubic',
            antialias=1,
            cubic_coeff_a=-0.5,
            exclude_outside=1,
            coordinate_transformation_mode='half_pixel',
        )
        op('Round', [f'{p}_rs'], f'{p}_rd')
        op('Clip', [f'{p}_rd', 'f0', 'f255'], 'n_img' if p == 'n' else 'o_full')
    # center crop, torchvision offsets: round((th - 224) / 2) with round half to even
    for a, s in (('top', 'o_th'), ('left', 'o_tw')):
        op('Sub', [s, 'i224'], f'{a}_d')
        op('Cast', [f'{a}_d'], f'{a}_df', to=TP.FLOAT)
        op('Div', [f'{a}_df', 'f2'], f'{a}_h')
        op('Round', [f'{a}_h'], f'{a}_r')
        op('Cast', [f'{a}_r'], f'{a}_i', to=TP.INT64)
        op('Unsqueeze', [f'{a}_i', 'i0'], f'{a}_1')
    op('Concat', ['top_1', 'left_1'], 'starts', axis=0)
    c('c224', [224, 224])
    c('ax23', [2, 3])
    op('Add', ['starts', 'c224'], 'ends')
    op('Slice', ['o_full', 'starts', 'ends', 'ax23'], 'o_img')
    # N -> 5-bit codes -> count map -> top 2,048 -> keep the first max(real, 64)
    c('f8', 8.0, np.float32)
    c('i1024', 1024)
    c('i32', 32)
    c('m1', [-1])
    c('k2048', [K])
    c('i64', [64])
    c('i0s', [0])
    c('z32768', [32768])
    c('ch0', [0])
    c('ch1', [1])
    c('ch2', [2])
    c('i_1', [1])
    op('Div', ['n_img', 'f8'], 'q_f')
    op('Floor', ['q_f'], 'q_fl')
    op('Cast', ['q_fl'], 'q', to=TP.INT64)
    for k in range(3):
        op('Gather', ['q', f'ch{k}'], f'q{k}', axis=1)
    op('Mul', ['q0', 'i1024'], 'q0m')
    op('Mul', ['q1', 'i32'], 'q1m')
    op('Add', ['q0m', 'q1m'], 'q01')
    op('Add', ['q01', 'q2'], 'code4')
    op('Reshape', ['code4', 'm1'], 'code')
    op('Shape', ['code'], 'npix')
    op('ConstantOfShape', ['npix'], 'ones', value=H.make_tensor('v', TP.FLOAT, [1], [1.0]))
    op('ConstantOfShape', ['z32768'], 'zeros', value=H.make_tensor('v', TP.FLOAT, [1], [0.0]))
    op('ScatterElements', ['zeros', 'code', 'ones'], 'counts', axis=0, reduction='add')
    n.append(H.make_node('TopK', ['counts', 'k2048'], ['tvals', 'tidx'], axis=0, largest=1, sorted=1))
    op('Greater', ['tvals', 'f0'], 'real')
    op('Cast', ['real'], 'real_i', to=TP.INT64)
    op('ReduceSum', ['real_i', 'i0s'], 'nreal', keepdims=1)
    op('Max', ['nreal', 'i64'], 'nkeep')
    op('Slice', ['tidx', 'i0s', 'nkeep', 'i0s'], 'codes')
    op('Slice', ['tvals', 'i0s', 'nkeep', 'i0s'], 'cnt')
    op('Cast', ['npix'], 'npix_f', to=TP.FLOAT)
    op('Div', ['cnt', 'npix_f'], 'share')
    g = H.make_graph(
        n,
        'front',
        [H.make_tensor_value_info('image', TP.UINT8, ['H', 'W', 'C'])],
        [
            H.make_tensor_value_info('o_img', TP.FLOAT, [1, 3, 224, 224]),
            H.make_tensor_value_info('codes', TP.INT64, ['n']),
            H.make_tensor_value_info('share', TP.FLOAT, ['n']),
        ],
        inits,
    )
    return H.make_model(g, opset_imports=[H.make_opsetid('', OPSET)], ir_version=IR)


def rgb2lab_t(rgb):
    """direct_colors.rgb2lab in torch (float32)."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / torch.tensor([0.95047, 1.0, 1.08883])
    f = torch.where(xyz > 216 / 24389, xyz.clamp(min=1e-12) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def fast_rope(bb):
    """Export-only rewrite of the student's attention RoPE, same math: rot(x) as x @ R
    (R a fixed +-1 permutation) instead of strided slices, and the class token handled
    by padding the RoPE table with cos = 1, sin = 0 instead of slice + concat."""
    import types
    import torch.nn.functional as F

    hd = bb.blocks[0].attn.head_dim
    R = torch.zeros(hd, hd)
    for i in range(0, hd, 2):  # out[i] = -x[i+1], out[i+1] = x[i]
        R[i + 1, i] = -1.0
        R[i, i + 1] = 1.0

    def fwd(self, x, rope=None, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            sin, cos = rope.chunk(2, -1)
            npt = self.num_prefix_tokens
            sin = torch.cat([torch.zeros(npt, sin.shape[-1]), sin], 0)
            cos = torch.cat([torch.ones(npt, cos.shape[-1]), cos], 0)
            q = q * cos + (q @ R) * sin
            k = k * cos + (k @ R) * sin
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        x = self.norm(x.transpose(1, 2).reshape(B, N, self.attn_dim))
        return self.proj_drop(self.proj(x))

    for b in bb.blocks:
        a = b.attn
        assert type(a).__name__ == 'AttentionRope' and not getattr(a, 'rotate_half', False) and a.fused_attn
        assert a.qkv is not None and a.gate is None and a.num_kv_groups == 1
        a.forward = types.MethodType(fwd, a)


def dedup(m):
    """Constant nodes -> initializers, then identical initializers stored once."""
    import onnx
    import hashlib
    from onnx import numpy_helper as nh

    g = m.graph
    keep = []
    for n in g.node:
        if n.op_type == 'Constant' and len(n.attribute) == 1 and n.attribute[0].name == 'value':
            t = onnx.TensorProto()
            t.CopyFrom(n.attribute[0].t)
            t.name = n.output[0]
            g.initializer.append(t)
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    seen, alias, uniq = {}, {}, []
    for t in g.initializer:
        a = nh.to_array(t)
        k = (a.dtype.str, a.shape, hashlib.sha1(a.tobytes()).hexdigest())
        if k in seen:
            alias[t.name] = seen[k]
        else:
            seen[k] = t.name
            uniq.append(t)
    del g.initializer[:]
    g.initializer.extend(uniq)
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in alias:
                n.input[i] = alias[x]
    return m


def fp16_store(m, min_elems=4096):
    """Large fp32 initializers stored as fp16 with a Cast back to fp32 (size only; math stays fp32)."""
    import onnx
    from onnx import numpy_helper as nh, helper

    g = m.graph
    new_nodes = []
    inits = []
    for t in g.initializer:
        a = nh.to_array(t)
        if a.dtype == np.float32 and a.size >= min_elems:
            h = nh.from_array(a.astype(np.float16), t.name + '_f16')
            inits.append(h)
            new_nodes.append(helper.make_node('Cast', [h.name], [t.name], to=onnx.TensorProto.FLOAT))
        else:
            inits.append(t)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    return m
