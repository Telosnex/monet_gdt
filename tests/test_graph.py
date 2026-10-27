"""The ONNX graph's contract: inputs, outputs, ranges, determinism."""

import hashlib
import itertools

import numpy as np
import pytest
from conftest import GRAPH, wallpaper
from mood_words import WORDS


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def run(session, img):
    return dict(zip([o.name for o in session.get_outputs()], session.run(None, {'image': img})))


def test_signature(session):
    assert [(i.name, i.type) for i in session.get_inputs()] == [('image', 'tensor(uint8)')]
    assert [o.name for o in session.get_outputs()] == ['scores', 'feature', 'colors', 'shapes']


@pytest.mark.parametrize('h,w', [(720, 1280), (2400, 1080), (224, 224), (100, 60)])
def test_output_shapes_any_size(session, h, w):
    out = run(session, wallpaper(h, w))
    assert out['scores'].shape == (1, len(WORDS))
    assert out['feature'].shape == (1, 192)
    assert out['colors'].shape == (3, 3)
    assert out['shapes'].shape == (1, 29)
    for k in ('scores', 'feature', 'shapes'):
        assert np.isfinite(out[k]).all(), k


def test_colors_are_rgb(session):
    c = run(session, wallpaper())['colors']
    assert c.min() >= 0 and c.max() <= 255


def test_rgba_with_opaque_alpha_matches_rgb(session):
    a, b = run(session, wallpaper(channels=3)), run(session, wallpaper(channels=4))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-5, err_msg=k)


def test_deterministic(session):
    img = wallpaper(seed=3)
    a, b = run(session, img), run(session, img)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_different_images_differ(session):
    a, b = run(session, wallpaper(seed=1)), run(session, wallpaper()[::-1, ::-1].copy())
    assert not np.allclose(a['feature'], b['feature'])


def test_shape_names_metadata(session):
    names = session.get_modelmeta().custom_metadata_map['monet_gdt.shapes'].split(',')
    assert len(names) == 29 and len(set(names)) == 29 and all(names)


def test_mood_map_matches_graph(session, mood_map):
    assert mood_map['words'] == WORDS
    assert mood_map['model_sha256'] == sha256(GRAPH)


def test_mood_map_tables_monotonic(mood_map):
    for w in WORDS:
        t = mood_map['map'][w]['t']
        assert len(t) > 1 and all(a <= b for a, b in itertools.pairwise(t)), w


def test_release_metadata(session):
    import graph_meta

    meta = session.get_modelmeta()
    m = meta.custom_metadata_map
    assert meta.producer_name == 'monet_gdt' and m['version'] == graph_meta.VERSION
    assert m['monet_gdt.mood_words'].split(',') == WORDS
    assert 'libmonet_gdt.shapes' not in m
