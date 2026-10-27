# rule_v1 (ADR D10) for the on-device graph, in PyTorch: exported into the
# ONNX graph by build_graph.py, and run directly by the Python checks, so
# both use one implementation.
#
# Input: the front end's color list (codes: 5-bit colors, sorted by pixel
# count, at least 64 entries; share: pixel share, 0 for padding) and the
# color model's per-role probabilities over that list (3, n).
# 1. Options, standing in for the app's merge (mergeWallpaperColorOptions):
#    up to G = 12 groups. Greedy: among the 64 most common colors, the
#    center with the most share within ΔE R (10) of colors not yet covered;
#    when every color is covered, the color farthest from the centers (so
#    images with few colors still get distinct options). Every color joins
#    its nearest center. A group's share and role probabilities are its
#    members' sums; it shows its most chromatic member with >= 1% of the
#    group's share (vivid merge).
# 2. rule_v1 over the groups, as libmonet monetGdtRuleV1Triad: Jev's
#    role probabilities -> the student's; HCT from a lookup table built by
#    libmonet's Hct (tools/libmonet_tools, scripts/libmonet_tools.sh), rounded.
# Output: (3, 3) int64 RGB of P, S, T.
import numpy as np
import torch

G, R_DE, CAND = 12, 10.0, 64
PLAUSIBLE, COLORFUL, HUE_GAP, W, MIN_DE, MIN_FRAC = 0.25, 16.0, 30.0, 0.5, 10.0, 0.01
BANDS = (30.0, 60.0, 90.0, 120.0)  # then 180: every hue is within 180


def rgb2lab(rgb):  # rgb 0-1 -> CIELAB (L 0-100); libmonet labFromArgb constants (the app's ΔE)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = torch.tensor(
        [[0.41233895, 0.35762064, 0.18051042], [0.2126, 0.7152, 0.0722], [0.01932141, 0.11916382, 0.95034478]]
    )
    xyz = lin @ m.T / torch.tensor([0.95047, 1.0, 1.08883])
    f = torch.where(xyz > 216 / 24389, xyz.clamp(min=1e-12) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def dist(a, b):  # (A, 3), (B, 3) -> (A, B)
    return ((a[:, None] - b[None]) ** 2).sum(-1).sqrt()


def code_rgb(codes):
    return torch.stack([codes // 1024, (codes // 32) % 32, codes % 32], -1).float() * 8 + 4


def hue_gap(a, b):
    x = a - b + 180
    return (x - 360 * torch.floor(x / 360) - 180).abs()


def first_max(v, ok):
    """Index of the largest v where ok (ties: the first), as a (1,) tensor."""
    return torch.where(ok, v, torch.full_like(v, -1e9)).argmax(-1, keepdim=True)


class Rule(torch.nn.Module):
    def __init__(self, lut_path):
        super().__init__()
        lut = np.fromfile(lut_path, '<f4').reshape(32768, 3)
        # rounded as the rule rounds; int16 hue and int16 chroma x 256 + tone (4 bytes a color, 131 KB)
        r = np.floor(lut + 0.5).astype(np.int32)
        assert r[:, 1].max() < 128 and r[:, 2].max() < 256
        self.register_buffer('lut_h', torch.tensor(r[:, 0].astype(np.int16)))
        self.register_buffer('lut_ct', torch.tensor((r[:, 1] * 256 + r[:, 2]).astype(np.int16)))

    def hct(self, codes):
        ct = self.lut_ct[codes].float()
        c = torch.floor(ct / 256)
        return self.lut_h[codes].float(), c, ct - 256 * c

    def groups(self, codes, share, probs):
        real = share > 0
        lab = rgb2lab(code_rgb(codes) / 255)
        _, chroma, _ = self.hct(codes)
        dc = dist(lab[:CAND], lab)  # (64, n)
        near = dc < R_DE
        d64 = dc[:, :CAND]  # (64, 64)
        avail = real.clone()
        taken = torch.zeros(CAND, dtype=torch.bool)
        centers, valid, dmin = [], [], torch.full((CAND,), 1e9)
        for _ in range(G):
            mass = (near & avail[None]).float() @ share  # (64,)
            ok = avail[:CAND]
            m_ok = torch.where(ok, mass, torch.full_like(mass, -1.0))
            far = torch.where(real[:CAND] & ~taken, dmin, torch.full_like(dmin, -1.0))
            has = m_ok.max() > 0
            score = torch.where(has, m_ok, far)
            j = score.argmax()
            v = score[j] > 0
            centers.append(j)
            valid.append(v)
            hit = torch.zeros(CAND, dtype=torch.bool).scatter(0, j[None], torch.ones(1, dtype=torch.bool)) & v
            taken = taken | hit
            avail = avail & ~(near[j] & v)
            dmin = torch.minimum(dmin, torch.where(v, d64[j], torch.full_like(dmin, 1e9)))
        cidx = torch.stack(centers)
        valid = torch.stack(valid)  # (G,)
        dg = torch.where(valid[:, None], dc[cidx], torch.full((G, dc.shape[1]), 1e9))
        a = (dg.argmin(0)[None] == torch.arange(G)[:, None]) & real[None]  # (G, n)
        af = a.float()
        g_share = af @ share  # (G,)
        g_p = probs @ af.T  # (3, G)
        ok = a & (share[None] >= MIN_FRAC * g_share[:, None])
        disp = torch.where(ok, chroma[None].expand(G, -1), torch.full((G, a.shape[1]), -1e9)).argmax(-1)  # (G,)
        valid = valid & (g_share > 0)
        return disp, valid, g_p

    def forward(self, codes, share, probs):
        disp, valid, p = self.groups(codes, share, probs)
        dc = codes[disp]
        hue, chroma, tone = self.hct(dc)
        sel = disp[pick(hue, chroma, tone, rgb2lab(code_rgb(dc) / 255), valid, p)]
        return code_rgb(codes[sel]).long()


def pick(hue, chroma, tone, lab, valid, p):
    """rule_v1 on options (rounded HCT, CIELAB, valid mask, (3, k) role
    probabilities) -> (3,) option indices of P, S, T."""
    if True:
        k = hue.shape[0]
        dd = dist(lab, lab)
        idx = torch.arange(k)

        def plaus(r, pool):
            top = torch.where(pool, p[r], torch.full_like(p[r], -1.0)).max()
            return pool & (p[r] >= PLAUSIBLE * top)

        def apart(picked):
            free = valid.clone()
            farm = valid.clone()
            for k in picked:
                free = free & (idx != k)
                farm = farm & (idx != k) & (dd[k] >= MIN_DE)
            # boolean select without Where (ONNX Runtime has no bool Where)
            fa, fr = farm.any(), free.any()
            return (farm & fa) | (free & ~fa & fr) | (valid & ~fa & ~fr)

        P = first_max(chroma, plaus(0, valid))[0]
        p_gray = chroma[P] < COLORFUL
        pt = plaus(2, apart([P]))
        colorful = pt & (chroma >= COLORFUL)
        gap_c = hue_gap(hue, (hue[P] + 180) % 360)
        band = sum((gap_c > b).long() for b in BANDS)
        minb = torch.where(colorful, band, torch.full_like(band, 99)).min()
        in_band = colorful & (band == minb)
        T = torch.where(colorful.any() & ~p_gray, first_max(chroma, in_band)[0], first_max(chroma, pt)[0])
        ps = plaus(1, apart([P, T]))
        distinct = ps & (chroma >= COLORFUL) & (hue_gap(hue, hue[P]) >= HUE_GAP) & (hue_gap(hue, hue[T]) >= HUE_GAP)
        mc = torch.where(ps, chroma, torch.zeros_like(chroma)).max()
        mc = torch.where(mc == 0, torch.ones_like(mc), mc)
        gap = torch.where(p_gray, (tone - tone[P]).abs() / 100, hue_gap(hue, hue[P]) / 180)
        S = torch.where(
            distinct.any(), first_max(chroma, distinct)[0], first_max(W * chroma / mc + (1 - W) * gap, ps)[0]
        )
        return torch.stack([P, S, T])
