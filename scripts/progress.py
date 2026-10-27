# Progress log for long steps: every 10 s, one line with what is running,
# items done / total, elapsed time and estimated time remaining.
#   p = Progress('downloading thumbnails', total=13000); p.add(); p.done()
import sys
import threading
import time


def _fmt(s):
    s = int(s)
    return f'{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s' if s >= 3600 else f'{s // 60}m{s % 60:02d}s'


class Progress:
    def __init__(self, what, total, every=10.0, start_done=0):
        self.what, self.total, self.n, self.skip = what, total, start_done, start_done
        self.t0 = time.time()
        self._stop = threading.Event()
        self._log('start')
        self._th = threading.Thread(target=self._beat, args=(every,), daemon=True)
        self._th.start()

    def add(self, k=1):
        self.n += k

    def _line(self):
        el = time.time() - self.t0
        new = self.n - self.skip
        left = self.total - self.n
        eta = _fmt(left * el / new) if new > 0 else 'unknown (no items done yet)'
        return f'{self.what}: {self.n:.0f}/{self.total:.0f} ({100 * self.n / max(self.total, 1):.0f}%), elapsed {_fmt(el)}, remaining ~{eta}'

    def _log(self, tag):
        print(f'[{time.strftime("%H:%M:%S")}] {tag} {self._line()}', flush=True, file=sys.stdout)

    def _beat(self, every):
        while not self._stop.wait(every):
            self._log('..')

    def done(self):
        self._stop.set()
        self._log('done')
