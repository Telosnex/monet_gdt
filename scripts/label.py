# Step 4 of docs/ADR_train_monet_gdt.md: Jev labels (D3, D7).
#   <venv>/bin/python -u scripts/label.py <mood|shapes|colors> <ids.json> <out.jsonl> [--parallel N] [--colors-dir DIR] [--sheet o0|o1]
# <ids.json>: list of selection IDs (data/selection.jsonl). Images:
# data/images/<part>/<id>.jpg. colors reads the app-built requests from
# --colors-dir (telosnex dev/design/monet_gdt/color_requests_test.dart output).
# Appends to <out.jsonl>; skips IDs already answered without error.
# Logs every 10 s: calls done / total, failures, tokens, elapsed, remaining.
import argparse
import glob
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from progress import Progress  # noqa: E402
import jev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('call')
    ap.add_argument('ids')
    ap.add_argument('out')
    ap.add_argument('--parallel', type=int, default=16)
    ap.add_argument('--colors-dir')
    ap.add_argument('--sheet', default='o0', help='shapes sheet; o1 for the D4 repeat')
    a = ap.parse_args()
    sel = {r['id']: r for r in map(json.loads, open(os.path.join(ROOT, 'data', 'selection.jsonl')))}
    mp = os.path.join(ROOT, 'data', 'selection_matched.jsonl')  # wallpaper-matched images (ADR D11), part 'matched'
    if os.path.exists(mp):
        sel.update({r['id']: r for r in map(json.loads, open(mp)) if r['id'] not in sel})
    if os.environ.get('IMAGES_DIR'):  # e.g. a resized copy of the images (same file names)
        img = lambda i: os.path.join(os.environ['IMAGES_DIR'], f'{i}.jpg')
    ids = json.load(open(a.ids))
    done = set()
    if os.path.exists(a.out):
        done = {r['id'] for r in map(json.loads, open(a.out)) if 'error' not in r}
    todo = [i for i in ids if i not in done]
    if not os.environ.get('IMAGES_DIR'):
        img = lambda i: os.path.join(ROOT, 'data', 'images', sel[i]['part'], f'{i}.jpg')
    if a.call == 'colors':
        reqs = {
            r['id']: r
            for f in glob.glob(os.path.join(a.colors_dir, 'colors_requests_*.jsonl'))
            for r in map(json.loads, open(f))
        }
        missing = [i for i in todo if i not in reqs]
        if missing:
            sys.exit(f'{len(missing)} IDs have no color request in {a.colors_dir}; run color_requests_test.dart first')
        fn = lambda i: jev.colors(reqs[i])
    else:
        fn = (
            (lambda i: jev.shapes(i, img(i), a.sheet))
            if a.call == 'shapes'
            else (lambda i: getattr(jev, a.call)(i, img(i)))
        )
    stats = {'failed': 0, 'tokens': 0}
    lock = threading.Lock()
    p = Progress(
        f'Jev {a.call} labels -> {os.path.relpath(a.out, ROOT)} ({len(done)} already done, {a.parallel} parallel)',
        len(ids),
        start_done=len(ids) - len(todo),
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'a') as f, ThreadPoolExecutor(a.parallel) as ex:
        for r in ex.map(fn, todo):
            r['part'] = sel[r['id']]['part']
            with lock:
                f.write(json.dumps(r) + '\n')
                f.flush()
                p.add()
                if 'error' in r:
                    stats['failed'] += 1
                else:
                    stats['tokens'] += r.get('tokens', 0)
    p.done()
    print(
        f"{a.call}: {stats['failed']} failed, {stats['tokens']} tokens, ${stats['tokens'] * 0.042 / 1e6:.2f}",
        flush=True,
    )


if __name__ == '__main__':
    main()
