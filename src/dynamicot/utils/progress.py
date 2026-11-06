from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, total: int, *, desc: str, unit: str = "iter"):
        self.total = max(int(total), 1)
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.postfix = ""
        self.start_time = time.time()
        self.last_render = 0.0

    def __iter__(self):
        for index in range(self.total):
            self.current = index
            self._render(force=index == 0)
            yield index
        self.current = self.total

    def set_postfix(self, **items):
        self.postfix = " ".join(f"{key}={value}" for key, value in items.items())
        self._render()

    def write(self, message: str):
        sys.stdout.write("\n" + message + "\n")
        sys.stdout.flush()
        self._render(force=True)

    def close(self):
        self.current = self.total
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, *, force: bool = False):
        now = time.time()
        if not force and now - self.last_render < 0.1 and self.current < self.total:
            return
        self.last_render = now

        completed = self.current if self.current < self.total else self.total
        width = 30
        filled = int(width * completed / self.total)
        bar = "=" * filled + "." * (width - filled)
        percent = 100.0 * completed / self.total
        elapsed = now - self.start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        line = (
            f"\r{self.desc} [{bar}] {completed}/{self.total} "
            f"{percent:6.2f}% {rate:5.2f} {self.unit}/s"
        )
        if self.postfix:
            line += f" {self.postfix}"
        sys.stdout.write(line)
        sys.stdout.flush()
