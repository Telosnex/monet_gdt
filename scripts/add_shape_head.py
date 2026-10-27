# ADR D5 step 6: adds the shape head to the app graph: a `shapes` output
# after `feature`.
#   .venv/bin/python -u scripts/add_shape_head.py <base.onnx> <head_dir> <out.onnx>
# Head (scripts/train_shape_head.py, trained on the base graph's `feature`):
#   x = (feature - mean) / std; h = GELU(x W1^T + b1) (exact, Erf);
#   o = sigmoid(h W2^T + b2), 87 = echoes, domain clash, tone clash per shape;
#   shapes = e * (1 - d) * (1 - clip((t - floor) / (1 - floor), 0, 1)), [1, 29].
# Shape order: metadata_props `monet_gdt.shapes` (comma separated); all
# metadata from lib/graph_meta.py.
# Checks, on the first N (default 313) test images of data/manifest.jsonl by
# ID (Pillow RGBA, long side <= 2048, as the app decodes): scores, colors and
# feature are bit-identical to the base graph; shapes equal the PyTorch head
# on the base feature within 1e-5.
import hashlib
import json
import os
import sys
import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper, TensorProto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
import graph_meta  # noqa: E402
import shape_head as T  # noqa: E402

base_path, head_dir, out_path = sys.argv[1:4]
c = torch.load(f'{head_dir}/shape_head.pt', map_location='cpu', weights_only=False)
base_sha = hashlib.sha256(open(base_path, 'rb').read()).hexdigest()
assert c['embedding'] == f'ship_{base_sha[:12]}', (c['embedding'], base_sha[:12])
sd = {k: v.numpy().astype(np.float32) for k, v in c['state_dict'].items()}
n = len(c['shapes'])
floor = float(c['tone_floor'])
m = onnx.load(base_path)
g = m.graph
P = 'shape_head/'


def init(name, a):
    g.initializer.append(
        numpy_helper.from_array(
            np.asarray(a, np.float32) if np.asarray(a).dtype.kind == 'f' else np.asarray(a), P + name
        )
    )
    return P + name


def node(op, ins, out, **kw):
    g.node.append(helper.make_node(op, ins, [P + out], name=P + out, **kw))
    return P + out


x = node('Div', [node('Sub', ['feature', init('mean', c['feat_mean'])], 'centered'), init('std', c['feat_std'])], 'x')
a = node('Add', [node('MatMul', [x, init('w1t', sd['f.1.weight'].T)], 'xw1'), init('b1', sd['f.1.bias'])], 'a')
erf = node('Erf', [node('Mul', [a, init('inv_sqrt2', np.float32(1 / np.sqrt(2)))], 'a_scaled')], 'erf')
h = node(
    'Mul',
    [node('Mul', [a, init('half', np.float32(0.5))], 'a_half'), node('Add', [erf, init('one', np.float32(1))], 'erf1')],
    'h',
)
o = node(
    'Sigmoid',
    [node('Add', [node('MatMul', [h, init('w2t', sd['f.4.weight'].T)], 'hw2'), init('b2', sd['f.4.bias'])], 'logits')],
    'o',
)
ax = init('axis', np.array([1], np.int64))


def part(k, name):
    return node(
        'Slice',
        [
            o,
            init(f'{name}_start', np.array([k * n], np.int64)),
            init(f'{name}_end', np.array([(k + 1) * n], np.int64)),
            ax,
        ],
        name,
    )


e, d, t = part(0, 'echoes'), part(1, 'domain'), part(2, 'tone')
one = P + 'one'
tc = node(
    'Clip',
    [
        node(
            'Div',
            [node('Sub', [t, init('floor', np.float32(floor))], 't_off'), init('span', np.float32(1 - floor))],
            't_scaled',
        ),
        init('zero', np.float32(0)),
        one,
    ],
    't_clip',
)
g.node.append(
    helper.make_node(
        'Mul',
        [node('Mul', [e, node('Sub', [one, d], 'keep_d')], 'ed'), node('Sub', [one, tc], 'keep_t')],
        ['shapes'],
        name=P + 'shapes',
    )
)
g.output.append(helper.make_tensor_value_info('shapes', TensorProto.FLOAT, [1, n]))
graph_meta.apply(m, c['shapes'])
onnx.checker.check_model(m)
onnx.save(m, out_path)

import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

sb = ort.InferenceSession(base_path, providers=['CPUExecutionProvider'])
sn = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
head = T.Head(192, 3 * n, c['hidden'], c['dropout'])
head.load_state_dict(c['state_dict'])
head.eval()
man = [json.loads(line) for line in open(os.path.join(ROOT, 'data', 'manifest.jsonl'))]
files = [
    os.path.join(ROOT, r['path']) for r in sorted((r for r in man if r['split'] == 'test'), key=lambda r: r['id'])
][: int(os.environ.get('N', '313'))]
worst = 0.0
for f in files:
    im = Image.open(f).convert('RGBA')
    im.thumbnail((2048, 2048), Image.LANCZOS)
    arr = np.asarray(im)
    bs, bf, bc = sb.run(['scores', 'feature', 'colors'], {'image': arr})
    ns, nf, nc, sh = sn.run(['scores', 'feature', 'colors', 'shapes'], {'image': arr})
    assert np.array_equal(bs, ns) and np.array_equal(bf, nf) and np.array_equal(bc, nc), f
    want = T.combine(T.predict((head, c['feat_mean'], c['feat_std']), bf))
    worst = max(worst, float(np.abs(sh - want).max()))
assert worst < 1e-5, worst
print(f'{len(files)} images: scores/feature/colors identical; shapes max |diff| vs PyTorch {worst:.2e}')
print(
    f'{os.path.getsize(base_path)} -> {os.path.getsize(out_path)} bytes; sha256 {hashlib.sha256(open(out_path, "rb").read()).hexdigest()}'
)
