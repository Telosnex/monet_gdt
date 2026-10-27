# ADR D5 step 7: app assets for a graph: the mood map (lib/mood_map.py,
# from libmonet_gdt export_app.py) and reference fixtures, with this build's teacher.
#   mood <graph.onnx>   out/app/monet_gdt_mood.json: per word, raw score
#       -> wallpaper percentile (the new teacher's quantile map composed
#       with the wallpaper population, fonts/wall_population.json;
#       WALL_POP=legacy: mood_font_model.json's 272 wallpapers, the map
#       shipped until 2026-10-26), checked against the Python path.
#       APP_OUT=<dir> writes elsewhere (default out/app).
#   fixtures <graph.onnx> <rgba dir> <reference.json>   as libmonet's.
import hashlib
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('APP_OUT') or os.path.join(ROOT, 'out', 'app')
sys.path[:0] = [os.path.join(ROOT, 'lib'), os.path.join(ROOT, 'scripts')]
import mood_map as XA  # noqa: E402


def _ref():
    """words, and raw scores -> wallpaper percentiles on the Python path (this build's teacher)."""
    import train_teacher as TT
    import mood_words as VM

    ck = torch.load(os.path.join(ROOT, 'out', 'teacher', 'teacher.pt'), weights_only=False)
    P, Y = np.asarray(ck['qmap_train_pred']), np.asarray(ck['qmap_train_y'])
    if os.environ.get('WALL_POP') == 'legacy':
        pop = json.load(open(os.path.join(ROOT, 'fonts', 'mood_font_model.json')))['wall_population']
    else:
        pop = json.load(open(os.path.join(ROOT, 'fonts', 'wall_population.json')))['population']
    assert list(VM.WORDS) == TT.WORDS
    q = TT.qmap_fit(P, Y)
    return (
        P,
        Y,
        pop,
        VM.WORDS,
        lambda raw: VM.wall_percentiles(dict(zip(VM.WORDS, q(np.asarray(raw, np.float64)[None])[0])), pop),
    )


def mood(model_path):
    P, Y, pop, words, ref = _ref()
    tab = XA.tables(P, Y, pop, words)
    rng = np.random.default_rng(0)
    test = np.concatenate(
        [P.astype(np.float64), rng.uniform(0, 1, (3000, 10)), np.unique(P)[:, None].repeat(10, 1)[:2000]]
    )
    bad = sum(1 for x in test if not np.array_equal(XA.apply(tab, words, x), ref(x)))
    print(f'composed map vs Python on {len(test)} score vectors: {bad} differ')
    assert bad == 0
    os.makedirs(OUT, exist_ok=True)
    d = {
        'version': 1,
        'model_sha256': hashlib.sha256(open(model_path, 'rb').read()).hexdigest(),
        'words': list(words),
        'map': tab,
    }
    p = os.path.join(OUT, 'monet_gdt_mood.json')
    json.dump(d, open(p, 'w'), separators=(',', ':'))
    print(p, os.path.getsize(p), 'bytes;', {w: len(tab[w]['v']) for w in words})


def fixtures(model_path, rgba_dir, out_path):
    """Reference for the app test. rgba_dir has <name>.rgba (packed RGBA, as the app decodes
    the wallpaper) and <name>.json ({width, height})."""
    import onnxruntime as ort

    _, _, _, words, ref = _ref()
    s = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    refj = {
        'model_sha256': hashlib.sha256(open(model_path, 'rb').read()).hexdigest(),
        'words': list(words),
        'images': {},
    }
    # Graphs with the shape head (add_shape_head.py) also give `shapes`.
    shape_names = s.get_modelmeta().custom_metadata_map.get('monet_gdt.shapes')
    if shape_names:
        refj['shapes'] = shape_names.split(',')
    for fn in sorted(f for f in os.listdir(rgba_dir) if f.endswith('.rgba')):
        name = fn[: -len('.rgba')]
        meta = json.load(open(os.path.join(rgba_dir, name + '.json')))
        w, h = meta['width'], meta['height']
        raw = open(os.path.join(rgba_dir, fn), 'rb').read()
        arr = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
        sc, col = s.run(['scores', 'colors'], {'image': arr})
        refj['images'][name] = {
            'width': w,
            'height': h,
            'rgba_sha256': hashlib.sha256(raw).hexdigest(),
            'scores': [float(x) for x in sc[0]],
            'colors': [[int(v) for v in r] for r in col],
            'percentiles': [float(x) for x in ref(sc[0])],
        }
        if shape_names:
            refj['images'][name]['shapes'] = [float(x) for x in s.run(['shapes'], {'image': arr})[0][0]]
        print(name, (w, h), ['#%02x%02x%02x' % tuple(r) for r in col])
    json.dump(refj, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    {'mood': lambda: mood(sys.argv[2]), 'fixtures': lambda: fixtures(*sys.argv[2:5])}[sys.argv[1]]()
