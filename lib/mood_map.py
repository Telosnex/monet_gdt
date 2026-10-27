"""The app's mood map: per word, raw score -> wallpaper percentile as a step table {t: thresholds, v: values}, the font teacher's
quantile map composed with the wallpaper population."""

import numpy as np


def tables(P, Y, pop, words):
    out = {}
    for w, word in enumerate(words):
        s = np.sort(P[:, w])
        n = len(s)
        qt = np.quantile(Y[:, w], np.clip(np.arange(n + 1) / n, 0, 1))
        p = np.array(pop[word])
        perc = [float((np.sum(p < v) + 0.5 * np.sum(p == v) + 0.5) / (len(p) + 1)) for v in qt.astype(np.float64)]
        t, v = [], [perc[0]]
        for k in range(1, n + 1):
            if perc[k] != v[-1]:
                t.append(float(s[k - 1]))
                v.append(perc[k])
        out[word] = {'t': t, 'v': v}
    return out


def apply(tab, words, raw):
    r = []
    for w, word in enumerate(words):
        t, v = tab[word]['t'], tab[word]['v']
        x = float(np.float64(raw[w]))
        i = int(np.searchsorted(np.array(t), x, 'left'))  # thresholds below x
        r.append(v[i])
    return np.array(r)
