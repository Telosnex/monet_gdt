# Step 2b of docs/ADR_train_monet_gdt.md: stream MONET tars and keep only the
# selected images (data/selection.jsonl). Full-size JPGs (MONET's 1024 px
# copies) go to data/images/<part>/<id>.jpg with the MONET JSON sidecar.
#   <venv>/bin/python -u scripts/download_corpus.py [--workers N] [--limit-shards K]
# Resumable: a shard is skipped once data/download_done/<shard>.json exists.
# Progress: every 10 s, MB streamed / total MB of the needed tars.
import argparse
import hashlib
import json
import os
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, HERE)
from progress import Progress  # noqa: E402
import select_corpus as S  # noqa: E402  (MONET path at the pinned revision)


class Counted:
    def __init__(self, f, p):
        self.f, self.p = f, p

    def read(self, n=-1):
        b = self.f.read(n)
        self.p.add(len(b) / 1e6)
        return b


def shard_job(fs, shard, rows, p, lock, stats):
    done = os.path.join(DATA, 'download_done', shard.replace('/', '_') + '.json')
    if os.path.exists(done):
        return
    want = {r['id']: r for r in rows}
    got = {}
    sha_ok = 0
    with fs.open(f'{S.MONET}/{shard}.tar', block_size=2**24) as raw:
        tf = tarfile.open(fileobj=Counted(raw, p), mode='r|')
        for m in tf:
            key, _, ext = m.name.partition('.')
            if key not in want or ext not in ('jpg', 'png', 'webp', 'json'):
                continue
            b = tf.extractfile(m).read()
            r = want[key]
            d = os.path.join(DATA, 'images', r['part'])
            os.makedirs(d, exist_ok=True)
            if ext == 'json':
                open(os.path.join(d, f'{key}.json'), 'wb').write(b)
            else:
                open(os.path.join(d, f'{key}.{ext}'), 'wb').write(b)
                got[key] = ext
                sha_ok += hashlib.sha256(b).hexdigest() == r.get('monet_sha256')
            if len(got) == len(want) and all(
                os.path.exists(os.path.join(DATA, 'images', want[k]['part'], f'{k}.json')) for k in got
            ):
                break  # all selected rows of this shard found; stop streaming
    os.makedirs(os.path.dirname(done), exist_ok=True)
    json.dump({'wanted': len(want), 'got': len(got), 'sha256_match': sha_ok}, open(done, 'w'))
    with lock:
        stats['wanted'] += len(want)
        stats['got'] += len(got)
        stats['sha'] += sha_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit-shards', type=int)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(DATA, 'selection.jsonl'))]
    by = {}
    for r in rows:
        by.setdefault(r['shard'], []).append(r)
    shards = sorted(by)[: a.limit_shards] if a.limit_shards else sorted(by)
    fs = HfFileSystem()
    todo = [s for s in shards if not os.path.exists(os.path.join(DATA, 'download_done', s.replace('/', '_') + '.json'))]
    total_mb = sum(fs.info(f'{S.MONET}/{s}.tar')['size'] for s in todo) / 1e6
    print(
        f'{len(shards)} shards with selected rows, {len(todo)} to stream, {total_mb / 1e3:.0f} GB at most '
        f'(a shard stops early once all its rows are found)',
        flush=True,
    )
    p = Progress(f'step 2b: streaming MONET tars ({a.workers} in parallel), MB read', int(total_mb))
    lock = threading.Lock()
    stats = {'wanted': 0, 'got': 0, 'sha': 0}
    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(shard_job, fs, s, by[s], p, lock, stats) for s in todo]:
            f.result()
    p.done()
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
