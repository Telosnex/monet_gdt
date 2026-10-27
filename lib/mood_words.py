"""Mood words and font matching."""

import numpy as np

WORDS = [
    'artistic',
    'attention-grabbing',
    'calm',
    'complex',
    'disorderly',
    'dramatic',
    'formal',
    'playful',
    'strong',
    'wide',
]


def wall_percentiles(raw, population):
    """raw: {word: value}; population: {word: [values]}. Mid-rank percentile, 0..1."""
    return np.array(
        [
            (np.sum(np.array(population[w]) < raw[w]) + 0.5 * np.sum(np.array(population[w]) == raw[w]) + 0.5)
            / (len(population[w]) + 1)
            for w in WORDS
        ]
    )


LOUD = ['attention-grabbing', 'dramatic', 'calm']  # calm counts reversed
REST = [w for w in WORDS if w not in LOUD]


def loudness_raw(X):
    """X: rows of percentiles over WORDS. Mean of attention-grabbing, dramatic and 1 - calm."""
    X = np.atleast_2d(X)
    j = [WORDS.index(w) for w in LOUD]
    return (X[:, j[0]] + X[:, j[1]] + (1 - X[:, j[2]])) / 3


def merge(X, loud_population):
    """Replace the three loudness words with one loudness percentile against loud_population."""
    X = np.atleast_2d(X)
    lr = loudness_raw(X)
    pop = np.asarray(loud_population)
    loud = np.array([(np.sum(pop < v) + 0.5 * np.sum(pop == v) + 0.5) / (len(pop) + 1) for v in lr])
    return np.column_stack([loud] + [X[:, WORDS.index(w)] for w in REST])


def scores(font_vectors, wall_vector):
    return -np.mean((np.asarray(font_vectors) - wall_vector) ** 2, axis=1)


def headline(model, raw_wall, *, wallpaper_key=None):
    """Deterministic: the closest font. Ties break by font name."""
    w = wall_percentiles(raw_wall, model['wall_population'])
    if model.get('merged'):
        w = merge(w, model['wall_loudness_population'])[0]
    s = scores(model['font_vectors'], w)
    return min(zip(model['fonts'], s), key=lambda t: (-t[1], t[0]))[0]
