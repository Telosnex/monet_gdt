# Jev calls for monet_gdt (ADR D3, D7). Requests come only from prompts/.
# Each call returns one JSONL record: {id, call, prompt_sha256, model,
# tokens, answers, ...derived} or {id, call, error}.
#   mood    prompts/mood.json
#   shapes  prompts/shapes.json + prompts/shapes/*
#   colors  bodies from telosnex dev/design/monet_gdt/color_requests_test.dart
#           (app code); this file only attaches the JPEG and sends it.
#           prompts/colors.json is the template those bodies must match.
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'prompts')
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def post(body):
    key = os.environ['USER_API_KEY_JEV']
    err = None
    for attempt in range(6):
        try:
            r = urllib.request.Request(
                ENDPOINT,
                data=json.dumps(body).encode(),
                headers={
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json',
                    'TypeSafe-Danger-Zone-Images': 'true',
                },
            )
            with urllib.request.urlopen(r, timeout=240) as resp:
                return json.loads(resp.read()), None
        except urllib.error.HTTPError as e:
            err = f'HTTP {e.code}: {e.read()[:300]!r}'
            if e.code < 500 and e.code != 429:
                break
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
        time.sleep(min(60, 2 * 2**attempt))
    return None, err


# ── mood ─────────────────────────────────────────────────────────────────
MOOD = json.load(open(os.path.join(P, 'mood.json')))
MOOD_SHA = _sha(os.path.join(P, 'mood.json'))


def mood(id_, path):
    im = Image.open(path).convert('RGB')
    im.thumbnail((1024, 1024), Image.LANCZOS)
    if min(im.size) < 320:
        k = 320 / min(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=88)
    ref = f'wallpaper-{uuid.uuid4()}'
    out, err = post(
        {
            'model': MOOD['model'],
            'state': {'about': MOOD['about'], 'wallpaper': ref},
            'assets': {ref: {'type': 'image', 'format': 'jpg', 'data_base64': base64.b64encode(b.getvalue()).decode()}},
            'questions': MOOD['questions'],
        }
    )
    rec = {'id': id_, 'call': 'mood', 'prompt_sha256': MOOD_SHA}
    if err:
        return {**rec, 'error': err}
    raw = {}
    for w in MOOD['words']:
        pr = out['answers'][w]['probabilities']
        pr = pr if isinstance(pr, dict) else {x['value']: x['probability'] for x in pr}
        d = np.array([pr.get(f'option_{i}', 0) for i in range(5)])
        raw[w] = float(d / max(d.sum(), 1e-9) @ np.linspace(0, 1, 5))
    return {
        **rec,
        'model': out.get('model'),
        'tokens': out['usage']['input_tokens'],
        'raw': raw,
        'answers': out['answers'],
    }


# ── shapes ───────────────────────────────────────────────────────────────
SH = os.path.join(P, 'shapes')
SHAPES = json.load(open(os.path.join(P, 'shapes.json')))
SHEETS = {k: json.load(open(os.path.join(SH, f'options_{k}.json')))['cells'] for k in ('o0', 'o1')}  # o1: D4 repeat
CELLS = SHEETS['o0']
ECHOES = open(os.path.join(SH, 'echoes.txt')).read()
EXTRA = {
    'd': json.load(open(os.path.join(SH, 'clash_domain.json'))),
    't': json.load(open(os.path.join(SH, 'clash_tone.json'))),
}
SHAPE_PNG = {
    v: base64.b64encode(open(os.path.join(SH, 'shape_images', f'{v}.png'), 'rb').read()).decode()
    for v in CELLS.values()
}
SHAPES_SHA = hashlib.sha256(
    json.dumps({**SHAPES['files_sha256'], 'spec': _sha(os.path.join(P, 'shapes.json'))}).encode()
).hexdigest()


def fit_wallpaper(w, h):
    # wallpaper_shapes/lib/image_fit.dart
    s = min(1.0, 1024 / w)
    s = min(s, 1568 / max(w, h))
    if min(w, h) * s < 280:
        s = 280 / min(w, h)
    W, H = round(w * s), round(h * s)
    if min(W, H) < 280 or max(W, H) > 1568:
        raise ValueError(f'aspect too extreme: {w}x{h}')
    return W, H


def _fill(v, img):
    if isinstance(v, str):
        return v.replace('{img}', img)
    if isinstance(v, list):
        return [_fill(x, img) for x in v]
    if isinstance(v, dict):
        return {k: _fill(x, img) for k, x in v.items()}
    return v


def shape_score(q, d, t):
    return q * (1 - d) * (1 - min(1.0, max(0.0, (t - 0.7) / 0.3)))


def shapes(id_, path, sheet='o0'):
    CELLS = SHEETS[sheet]
    im = Image.open(path).convert('RGB')
    W, H = fit_wallpaper(*im.size)
    if (W, H) != im.size:
        im = im.resize((W, H), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, 'PNG')
    ref = uuid.uuid4().hex[:16]
    qs = {f'q_{k}': {'type': 'noul', 'instructions': ECHOES.replace('{img}', f'shape-{ref}-{k}')} for k in CELLS}
    for x, spec in EXTRA.items():
        for k in CELLS:
            qs[f'{x}_{k}'] = {'type': 'noul', **_fill(spec, f'shape-{ref}-{k}')}
    assets = {f'wall-{ref}': {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(b.getvalue()).decode()}}
    for k, v in CELLS.items():
        assets[f'shape-{ref}-{k}'] = {'type': 'image', 'format': 'png', 'data_base64': SHAPE_PNG[v]}
    out, err = post(
        {'model': SHAPES['model'], 'state': {'wallpaper': f'wall-{ref}'}, 'assets': assets, 'questions': qs}
    )
    rec = {'id': id_, 'call': 'shapes', 'sheet': sheet, 'prompt_sha256': SHAPES_SHA}
    if err:
        return {**rec, 'error': err}
    a = out['answers']
    n = lambda key: float(a[key]['noul'])
    try:
        parts = {CELLS[k]: {'echoes': n(f'q_{k}'), 'domain': n(f'd_{k}'), 'tone': n(f't_{k}')} for k in CELLS}
    except Exception as e:
        return {**rec, 'error': f'bad answers: {e}', 'answers': a}
    return {
        **rec,
        'model': out.get('model'),
        'tokens': out['usage']['input_tokens'],
        'parts': parts,
        'scores': {s: shape_score(v['echoes'], v['domain'], v['tone']) for s, v in parts.items()},
    }


# ── colors ───────────────────────────────────────────────────────────────
def colors(req):
    """[req]: one line of colors_requests_*.jsonl (app-built body)."""
    rec = {'id': req['id'], 'call': 'colors'}
    if 'skip' in req:
        return {**rec, 'error': f"skip: {req['skip']}"}
    body = json.loads(json.dumps(req['body']))
    jpeg = open(req['jpeg'], 'rb').read()
    if hashlib.sha256(jpeg).hexdigest() != req['jpeg_sha256']:
        return {**rec, 'error': 'jpeg changed'}
    for name, asset in body['assets'].items():  # multipart part -> inline base64
        body['assets'][name] = {'type': 'image', 'format': 'jpg', 'data_base64': base64.b64encode(jpeg).decode()}
    rec['prompt_sha256'] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != 'assets'}, sort_keys=True).encode()
    ).hexdigest()
    out, err = post(body)
    if err:
        return {**rec, 'error': err}
    # Labels are the per-role probabilities. The triad is app code
    # (libmonet monetGdtRuleV1Triad, rule_v1); it is not computed here.
    probs = {}
    for r in ('primary', 'secondary', 'tertiary'):
        pr = out['answers'][r]['probabilities']
        probs[r] = pr if isinstance(pr, dict) else {x['value']: x['probability'] for x in pr}
    return {
        **rec,
        'model': out.get('model'),
        'tokens': out['usage']['input_tokens'],
        'options': req['options'],
        'probabilities': probs,
    }
