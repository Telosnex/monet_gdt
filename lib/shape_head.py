"""Shape head training and metrics.

The head predicts 87 = 29 shapes x (echoes, domain clash, tone clash); combine() turns
them into one score per shape: echoes x (1 - domain) x (1 - tone above TONE_FLOOR).
TARGET (environment, at import): 'split' (default, rank loss) or the plain BCE target."""

import os
import numpy as np
import torch

TARGET = os.environ.get('TARGET', 'split')
TONE_FLOOR = 0.7
MAX_EPOCHS = int(os.environ.get('MAX_EPOCHS', '200'))
RANK, TAU = (0.1, 0.05) if TARGET == 'split' else (0.0, None)


def combine(A):
    """echoes x (1 - domain) x (1 - tone above 0.7, scaled to 0..1)."""
    if A.shape[1] == 29:
        return A
    e, d, t = A[:, :29], A[:, 29:58], A[:, 58:87]
    return e * (1 - d) * (1 - np.clip((t - TONE_FLOOR) / (1 - TONE_FLOOR), 0, 1))


class Head(torch.nn.Module):
    def __init__(s, d, k, hidden, drop):
        super().__init__()
        s.f = (
            torch.nn.Sequential(
                torch.nn.Dropout(drop / 2),
                torch.nn.Linear(d, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(drop),
                torch.nn.Linear(hidden, k),
            )
            if hidden
            else torch.nn.Sequential(torch.nn.Dropout(drop / 2), torch.nn.Linear(d, k))
        )

    def forward(s, x):
        return s.f(x)


def _tcombine(P):
    e, d, t = P[:, :29], P[:, 29:58], P[:, 58:87]
    return e * (1 - d) * (1 - ((t - TONE_FLOOR) / (1 - TONE_FLOOR)).clamp(0, 1))


def winner_weights(Y):
    win = combine(Y).argmax(1)
    fr = np.bincount(win, minlength=29) / len(win)
    w = 1 / np.sqrt(fr[win])
    return (w / w.mean()).astype(np.float32)


def fit_rank(Xtr, Ytr, hidden, drop, wd, epochs=60, seed=0, Xva=None, Yva=None):
    F = torch.nn.functional
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = Head(Xtr.shape[1], Ytr.shape[1], hidden, drop)
    with torch.no_grad():
        m.f[-1].bias.copy_(torch.logit(torch.tensor(Ytr.mean(0)).clamp(1e-4, 1 - 1e-4)))
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=wd)
    x = torch.tensor((Xtr - mu) / sd)
    y = torch.tensor(Ytr)
    wt = torch.tensor(winner_weights(Ytr))
    tgt = F.softmax(torch.log(_tcombine(y) + 1e-4) / TAU, 1)
    best = (-1e9, None, epochs)
    for ep in range(epochs):
        m.train()
        for idx in torch.randperm(len(x)).split(256):
            opt.zero_grad()
            o = m(x[idx])
            w = wt[idx]
            l = (F.binary_cross_entropy_with_logits(o, y[idx], reduction='none').mean(1) * w).sum() / w.sum()
            lp = torch.log(_tcombine(torch.sigmoid(o)) + 1e-4) / TAU
            l = l + RANK * (-(tgt[idx] * F.log_softmax(lp, 1)).sum(1) * w).sum() / w.sum()
            l.backward()
            opt.step()
        if Xva is not None:
            c = metrics(predict((m.eval(), mu, sd), Xva), Yva)['corr']
            if c > best[0]:
                best = (c, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1)
    if best[1] is not None:
        m.load_state_dict(best[1])
    return (m.eval(), mu, sd), best[2]


def predict(model, X):
    m, mu, sd = model
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor((X - mu) / sd))).numpy()


def bce(P, Y):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return float(-(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean())


def metrics(P, Y):
    P, Y = combine(P), combine(Y)
    tp, ty = P.argmax(1), Y.argmax(1)
    rank = np.argsort(-P, 1)
    top3 = np.mean([ty[i] in rank[i, :3] for i in range(len(Y))])
    # Jev's yes-probability for the student's pick, as a share of Jev's best.
    regret = np.mean(Y[np.arange(len(Y)), tp] / Y.max(1))
    corr = np.mean([np.corrcoef(P[i], Y[i])[0, 1] for i in range(len(Y))])
    return dict(
        bce=round(bce(P, Y), 4),
        top1=round(float(np.mean(tp == ty)), 3),
        jev_top1_in_top3=round(float(top3), 3),
        pick_value=round(float(regret), 3),
        corr=round(float(corr), 3),
    )
