# CONCLUDED 2026-10-25. Real + MONET synthetic beats real only on wallpaper
# test sets (mood); became ADR D1. Results: results/.
#
# Jev mood labels for the synthetic A/B (request: prompts/mood.json).
#   <venv>/bin/python -u experiments/synthetic_ab/label_mood.py
# Split by sha256(id): real (megalith10m) 2,000 train / 500 test; synthetic
# (3 generators) 2,000 train / 300 test; the rest unused. Thumbnail sent as
# JPEG q88 (as mood_fresh.ask), short side scaled up to 320 px if smaller. Appends to data/jev_mood.jsonl; resumable.
import base64
import hashlib
import io
import json
import os
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
PROMPT_PATH = os.path.join(HERE, '..', '..', 'prompts', 'mood.json')
PROMPT = json.load(open(PROMPT_PATH))
PSHA = hashlib.sha256(open(PROMPT_PATH, 'rb').read()).hexdigest()
KEY = os.environ['USER_API_KEY_JEV']
MIN_SIDE = 320


def plan():
    M = [json.loads(l) for l in open(os.path.join(DATA, 'manifest.jsonl'))]
    h = lambda m: hashlib.sha256(m['id'].encode()).hexdigest()
    real = sorted((m for m in M if m['source'] == 'megalith10m'), key=h)
    syn = sorted((m for m in M if m['source'].startswith('synthetic')), key=h)
    out = []
    for m in real[:2000]:
        out.append({**m, 'arm': 'real', 'split': 'train'})
    for m in real[2000:2500]:
        out.append({**m, 'arm': 'real', 'split': 'test'})
    for m in syn[:2000]:
        out.append({**m, 'arm': 'synthetic', 'split': 'train'})
    for m in syn[2000:2300]:
        out.append({**m, 'arm': 'synthetic', 'split': 'test'})
    return out


def post(req):
    err = None
    for attempt in range(5):
        try:
            r = urllib.request.Request(
                'https://api.typesafe.ai/v1/systemone',
                data=json.dumps(req).encode(),
                headers={
                    'Authorization': f'Bearer {KEY}',
                    'Content-Type': 'application/json',
                    'TypeSafe-Danger-Zone-Images': 'true',
                },
            )
            with urllib.request.urlopen(r, timeout=180) as resp:
                return json.loads(resp.read()), None
        except urllib.error.HTTPError as e:
            err = f'HTTP {e.code}: {e.read()[:300]!r}'
            if e.code < 500 and e.code != 429:
                break
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
        time.sleep(2 * 2**attempt)
    return None, err


def ask(m):
    im = Image.open(os.path.join(DATA, 'thumbs', m['source'], f"{m['id']}.webp")).convert('RGB')
    # Jev rejects a short side under ~280 px; MONET thumbnails are < 384 px.
    if min(im.size) < MIN_SIDE:
        k = MIN_SIDE / min(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=88)
    ref = f'wallpaper-{uuid.uuid4()}'
    out, err = post(
        {
            'model': PROMPT['model'],
            'state': {'about': PROMPT['about'], 'wallpaper': ref},
            'assets': {ref: {'type': 'image', 'format': 'jpg', 'data_base64': base64.b64encode(b.getvalue()).decode()}},
            'questions': PROMPT['questions'],
        }
    )
    r = {'id': m['id'], 'source': m['source'], 'arm': m['arm'], 'split': m['split'], 'prompt_sha256': PSHA}
    if err:
        return {**r, 'error': err}
    raw = {}
    for w in PROMPT['words']:
        pr = out['answers'][w]['probabilities']
        pr = pr if isinstance(pr, dict) else {x['value']: x['probability'] for x in pr}
        d = np.array([pr.get(f'option_{i}', 0) for i in range(5)])
        raw[w] = float(d / max(d.sum(), 1e-9) @ np.linspace(0, 1, 5))
    return {**r, 'tokens': out['usage']['input_tokens'], 'raw': raw, 'answers': out['answers']}


def main():
    path = os.path.join(DATA, 'jev_mood.jsonl')
    done = {json.loads(l)['id'] for l in open(path) if 'error' not in json.loads(l)} if os.path.exists(path) else set()
    todo = [m for m in plan() if m['id'] not in done]
    print(f'{len(todo)} to ask, {len(done)} done', flush=True)
    n = bad = tok = 0
    t = time.time()
    with open(path, 'a') as f, ThreadPoolExecutor(16) as ex:
        for r in ex.map(ask, todo):
            f.write(json.dumps(r) + '\n')
            f.flush()
            n += 1
            if 'error' in r:
                bad += 1
            else:
                tok += r['tokens']
            if n % 500 == 0 or n == len(todo):
                print(f'{n}/{len(todo)}, {bad} failed, {tok} tokens, {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
