"""Shared fixtures. Graph tests need a built graph (see README, Building);
they are skipped when it is missing. MONET_GDT_GRAPH / MONET_GDT_MOOD
override the default output paths."""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lib'))

GRAPH = os.environ.get('MONET_GDT_GRAPH', os.path.join(ROOT, 'out', 'graph', 'monet_gdt_w8_shapes.onnx'))
MOOD = os.environ.get('MONET_GDT_MOOD', os.path.join(ROOT, 'out', 'app', 'monet_gdt_mood.json'))


@pytest.fixture(scope='session')
def session():
    if not os.path.exists(GRAPH):
        pytest.skip(f'no graph at {GRAPH}; build it first')
    import onnxruntime as ort

    return ort.InferenceSession(GRAPH, providers=['CPUExecutionProvider'])


@pytest.fixture(scope='session')
def mood_map():
    if not os.path.exists(MOOD):
        pytest.skip(f'no mood map at {MOOD}; build it first')
    with open(MOOD) as f:
        return json.load(f)


def wallpaper(h=720, w=1280, channels=3, seed=0):
    """A deterministic synthetic wallpaper: two color gradients plus noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([255 * x / w, 255 * y / h, 128 + 64 * np.sin(x / 40)], -1)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    if channels == 4:
        img = np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], -1)
    return img
