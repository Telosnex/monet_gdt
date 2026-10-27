"""Color model and color lists.

HistNet: the color model; context() reads the image's color list (Lab / 100 + share)
  and the L2-normalized image feature, logits() scores each color for role P, S or T.
hist(path): the color list of an image file: 5-bit RGB colors (code * 8 + 4), shares,
  number of distinct colors. LISTS (environment, at import) picks the image:
  '' = 256 px thumbnail, 'crop224' / 'fit224' = ONNX Resize as the graph,
  'graph' = graph_front.front() itself (needs SINGLE=1 TWO=1 OPSET=20).
loss_fn: training loss. pick: indices (B, 3) for P, S, T; AREA = True for the area pick."""

import math
import os
import numpy as np
import torch
from PIL import Image

MAXK = 2048
LISTS = os.environ.get('LISTS', '')  # '' = 256 px thumbnail; 'crop224' = the font model's 224 x 224 input
_RS = None


def fit224(im):
    """Whole image, long side 224 (aspect kept, no crop): ONNX Resize as crop224, round."""
    crop224(Image.new('RGB', (224, 224)))  # builds the session
    w, h = im.size
    tw, th = (224, max(1, int(224 * h / w))) if w >= h else (max(1, int(224 * w / h)), 224)
    a = np.asarray(im, np.float32).transpose(2, 0, 1)[None]
    r = np.clip(np.round(_RS.run(None, {'x': a, 'sizes': np.array([1, 3, th, tw], np.int64)})[0][0]), 0, 255)
    return r.transpose(1, 2, 0).astype(np.uint8)


_FS = None


def graph_list(path):
    """The final graph's own color list: graph_front.front() (single resize + pre-pool,
    opset 20), outputs codes + share. Needs SINGLE=1 TWO=1 OPSET=20 in the environment."""
    global _FS
    if _FS is None:
        assert os.environ.get('SINGLE') == '1' and os.environ.get('TWO') == '1' and os.environ.get('OPSET') == '20'
        import onnxruntime as ort
        import graph_front as G

        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        _FS = ort.InferenceSession(G.front().SerializeToString(), o, providers=['CPUExecutionProvider'])
    codes, share = _FS.run(['codes', 'share'], {'image': np.asarray(Image.open(path).convert('RGB'))})
    keep = share > 0
    codes, share = codes[keep], share[keep]
    rgb = (np.stack([codes // 1024, (codes // 32) % 32, codes % 32], -1) * 8 + 4).astype(np.uint8)
    return rgb, share.astype(np.float32), len(codes)


def crop224(im):
    """The one-input graph's image: ONNX Resize (opset 18, cubic, antialias, a = -0.5,
    exclude_outside, half_pixel) of the full image to short side 224, round, center crop
    224 x 224 (torchvision offsets). uint8 (224, 224, 3)."""
    global _RS
    import onnxruntime as ort
    from onnx import helper, TensorProto

    if _RS is None:
        x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, None, None])
        sz = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, None, None])
        n = helper.make_node(
            'Resize',
            ['x', '', '', 'sizes'],
            ['y'],
            mode='cubic',
            antialias=1,
            cubic_coeff_a=-0.5,
            exclude_outside=1,
            coordinate_transformation_mode='half_pixel',
        )
        m = helper.make_model(
            helper.make_graph([n], 'r', [x, sz], [y]), opset_imports=[helper.make_opsetid('', 18)], ir_version=9
        )
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        _RS = ort.InferenceSession(m.SerializeToString(), o, providers=['CPUExecutionProvider'])
    w, h = im.size
    tw, th = (224, int(224 * h / w)) if w <= h else (int(224 * w / h), 224)
    a = np.asarray(im, np.float32).transpose(2, 0, 1)[None]
    r = np.clip(np.round(_RS.run(None, {'x': a, 'sizes': np.array([1, 3, th, tw], np.int64)})[0][0]), 0, 255)
    t, l = int(round((th - 224) / 2)), int(round((tw - 224) / 2))
    return r[:, t : t + 224, l : l + 224].transpose(1, 2, 0).astype(np.uint8)


def hist(path):
    if LISTS == 'graph':
        try:
            return graph_list(path)
        except Exception:  # unreadable file
            q = np.zeros((1, 3), np.int32)
            return (q * 8 + 4).astype(np.uint8), np.ones(1, np.float32), 1
    try:
        if LISTS in ('crop224', 'fit224'):
            q = (
                (crop224 if LISTS == 'crop224' else fit224)(Image.open(path).convert('RGB')).reshape(-1, 3) >> 3
            ).astype(np.int32)
        else:
            im = Image.open(path)
            im.draft('RGB', (512, 512))
            im = im.convert('RGB')
            im.thumbnail((256, 256))
            q = (np.asarray(im).reshape(-1, 3) >> 3).astype(np.int32)
    except Exception:  # unreadable file
        q = np.zeros((1, 3), np.int32)
    u, n = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)
    o = np.argsort(-n, kind='stable')
    u, n = u[o], n[o]
    k = min(len(u), MAXK)
    rgb = (np.stack([u // 1024, (u // 32) % 32, u % 32], -1)[:k] * 8 + 4).astype(np.uint8)
    return rgb, n[:k] / n.sum(), len(u)


def relf(c, ref):
    """Each color's relation to a reference color; all Lab / 100. (B, K, 6)"""
    d = c - ref[:, None]
    hop = (
        torch.abs(
            torch.remainder(
                torch.atan2(c[..., 2], c[..., 1]) - torch.atan2(ref[:, 2], ref[:, 1])[:, None] + math.pi, 2 * math.pi
            )
            - math.pi
        )
        / math.pi
    )
    cr = torch.hypot(ref[:, 1], ref[:, 2])[:, None].expand_as(hop)
    return torch.cat([d, d.norm(dim=-1, keepdim=True), hop[..., None], cr[..., None]], -1)


class HistNet(torch.nn.Module):
    def __init__(self, emb_dim, d=96, latents=16):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(5, d), torch.nn.GELU(), torch.nn.Linear(d, d))
        self.lat = torch.nn.Parameter(torch.randn(latents, d) * 0.02)
        self.read = torch.nn.MultiheadAttention(d, 4, batch_first=True)
        self.img = torch.nn.Linear(emb_dim, d) if emb_dim else None
        layer = torch.nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, 2)
        self.write = torch.nn.MultiheadAttention(d, 4, batch_first=True)
        self.q = torch.nn.Parameter(torch.randn(3, d) * 0.02)
        self.qatt = torch.nn.MultiheadAttention(d, 4, batch_first=True)
        self.score = torch.nn.Sequential(torch.nn.Linear(2 * d + 12, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))

    def context(self, lab, sh, mask, emb):
        x = torch.cat(
            [lab, torch.hypot(lab[..., 1], lab[..., 2])[..., None], (torch.log(sh + 1e-6) / 14)[..., None]], -1
        )
        h = self.inp(x)
        z, _ = self.read(self.lat.expand(len(h), -1, -1), h, h, key_padding_mask=~mask)
        if self.img is not None:
            z = torch.cat([z, self.img(emb)[:, None]], 1)
        z = self.enc(z)
        w, _ = self.write(h, z, z)
        q, _ = self.qatt(self.q.expand(len(h), -1, -1), z, z)
        return h + w, q

    def logits(self, h, q, r, lab, mask, P=None, S=None):
        zero = torch.zeros(*lab.shape[:2], 6, device=lab.device)
        rel = torch.cat([relf(lab, P) if P is not None else zero, relf(lab, S) if S is not None else zero], -1)
        s = self.score(torch.cat([h, q[:, r : r + 1].expand(-1, lab.shape[1], -1), rel], -1)).squeeze(-1)
        return s.masked_fill(~mask, -1e4)


SEP_CAP = 20.0  # ΔE; no extra credit beyond this
SEP_TOP = 32  # expected distance over each role's 32 most likely colors (a full K x K is too big)
TAKEN_DE = 20.0
CE_TAU = 5.0  # ΔE; soft target width
AREA_DE = 10.0  # ΔE; area readout radius
AREA = False  # set per arm in run()
PICK = os.environ.get('PICK', 'chroma1_x10')
PICK_MIN_FRAC = None if PICK == 'share' else 0.01  # most chromatic color with 1% of the area
PICK_EXCL_DE = 0.0 if PICK == 'share' else 10.0  # no S or T within ΔE 10 of an earlier pick
PICK_FALLBACK = 'far'


def near(lab, c):
    return ((lab - c[:, None]).norm(dim=-1) * 100 < TAKEN_DE).float()


def taken(lab, probs, picks):
    """Probability S and T put on colors already taken (within TAKEN_DE of the actual
    argmax picks, no gradient through the picks). 0-1. If every color is near, the
    penalty is 1 whatever the model does, so it has no pull."""
    s = (probs[1] * near(lab, picks[0])).sum(-1)
    t = (probs[2] * torch.maximum(near(lab, picks[0]), near(lab, picks[1]))).sum(-1)
    return ((s + t) / 2).mean()


def sep(lab, probs):
    """Mean over pairs (P-S, P-T, S-T) of the expected capped distance, 0-1. probs: 3 x (B, K)."""
    tops = [p.topk(min(SEP_TOP, p.shape[-1]), -1) for p in probs]
    cols = [torch.gather(lab, 1, t.indices[..., None].expand(-1, -1, 3)) for t in tops]
    out = 0
    for a, b in ((0, 1), (0, 2), (1, 2)):
        d = (cols[a][:, :, None] - cols[b][:, None]).norm(dim=-1) * 100
        w = tops[a].values[:, :, None] * tops[b].values[:, None]
        out = out + (w * (d.clamp(max=SEP_CAP) / SEP_CAP)).sum((1, 2))
    return out.mean() / 3


def loss_fn(net, lab, sh, mask, emb, y, own=0.0, sep_w=0.0, taken_w=0.0, ce=False):
    """own: per wallpaper, chance that S and T get the model's own earlier picks
    (argmax, no gradient) as references instead of Jev's colors."""
    h, q = net.context(lab, sh, mask, emb)
    total, refs, probs, picks, ar = 0, [], [], [], torch.arange(len(lab), device=lab.device)
    for r in range(3):
        logits = net.logits(h, q, r, lab, mask, *(refs + [None, None])[:2])
        prob = torch.softmax(logits, -1)
        probs.append(prob)
        picks.append(lab[ar, logits.argmax(-1)].detach())
        if ce:
            d = (lab - y[:, r][:, None]).norm(dim=-1) * 100
            t = torch.softmax((-d / CE_TAU).masked_fill(~mask, -1e4), -1)
            total = total - (t * torch.log_softmax(logits, -1)).sum(-1).mean()
        else:
            total = total + (prob * (lab - y[:, r][:, None]).norm(dim=-1)).sum(-1).mean()
        if r < 2:
            if own > 0:
                mine = lab[ar, logits.argmax(-1)].detach()
                use = torch.rand(len(lab), 1, device=lab.device) < own
                refs.append(torch.where(use, mine, y[:, r]))
            else:
                refs.append(y[:, r])
    if ce:
        return total / 3 + taken_w * taken(lab, probs, picks)
    loss = total / 3 * 100
    if taken_w:
        loss = loss + taken_w * 100 * taken(lab, probs, picks)
    return loss - sep_w * 100 * sep(lab, probs) if sep_w else loss


_G = object()  # "use the module setting"


def allowed(lab, sh, mask, refs, excl, fallback='far', flags=None):
    """Colors S or T may take: none within excl ΔE of an earlier pick (refs). If none are left:
    'far' keeps the colors at least as far from the earlier picks as the farthest color with 0.1%
    of the image (or the farthest color, if none has); 'all' keeps every color."""
    if not refs or not excl:
        return mask
    dmin = torch.stack([(lab - ref[:, None]).norm(dim=-1) * 100 for ref in refs]).min(0).values
    m = mask & (dmin >= excl)
    empty = ~m.any(-1)
    if flags is not None:
        flags.append(empty)
    if fallback == 'far':
        # and/or, not torch.where: ONNX Runtime has no bool Where
        big = mask & (sh >= 1e-3)
        has_big = big.any(-1, keepdim=True)
        pool = (has_big & big) | (~has_big & mask)
        t = dmin.masked_fill(~pool, -1).max(-1, keepdim=True).values
        return (empty[:, None] & mask & (dmin >= t)) | (~empty[:, None] & m)
    return (empty[:, None] & mask) | (~empty[:, None] & m)


def pick(net, lab, sh, mask, emb, min_frac=_G, excl=_G, fallback=_G, flags=None):
    """Indices (B, 3) into the list, chosen in order P, S, T.

    Without AREA: each role's top color. With AREA: S and T may only take allowed() colors.
    The best area is the allowed color among the top 64 whose allowed colors within AREA_DE carry
    the most probability. In that area: the most chromatic color with at least min_frac of the
    area's pixels (the area's most common color always qualifies); min_frac None: the most common
    color. Defaults: PICK_MIN_FRAC, PICK_EXCL_DE, PICK_FALLBACK."""
    min_frac = PICK_MIN_FRAC if min_frac is _G else min_frac
    excl = PICK_EXCL_DE if excl is _G else excl
    fallback = PICK_FALLBACK if fallback is _G else fallback
    h, q = net.context(lab, sh, mask, emb)
    idx, refs = [], []
    ar = torch.arange(len(lab), device=lab.device)
    chroma = torch.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    for r in range(3):
        lg = net.logits(h, q, r, lab, mask, *(refs + [None, None])[:2])
        i = lg.argmax(-1)
        if AREA:
            m = allowed(lab, sh, mask, refs, excl, fallback, flags)
            p = torch.softmax(lg.masked_fill(~m, -1e9), -1)
            top = p.topk(min(64, p.shape[-1]), -1).indices
            cen = torch.gather(lab, 1, top[..., None].expand(-1, -1, 3))
            near = (torch.cdist(cen, lab) * 100 < AREA_DE) & m[:, None]
            score = (near.float() * p[:, None]).sum(-1).masked_fill(~m.gather(1, top), -1.0)
            c = top.gather(1, score.argmax(-1, keepdim=True)).squeeze(1)
            nc = (torch.cdist(lab[ar, c][:, None], lab)[:, 0] * 100 < AREA_DE) & m
            if min_frac is None:
                i = sh.masked_fill(~nc, -1).argmax(-1)
            else:
                area = (sh * nc).sum(-1, keepdim=True)
                top_sh = sh.masked_fill(~nc, -1).max(-1, keepdim=True).values
                ok = nc & (sh >= torch.minimum(min_frac * area, top_sh))
                i = chroma.masked_fill(~ok, -1).argmax(-1)
        idx.append(i)
        refs.append(lab[ar, i])
    return torch.stack(idx, 1)
