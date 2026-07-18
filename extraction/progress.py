from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, label: str, total: int, width: int = 28) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.width = max(8, int(width))
        self.current = 0
        self.started_at = time.monotonic()
        self._last_draw = 0.0

    def update(self, current: int | None = None, step: int = 1, force: bool = False) -> None:
        if current is None:
            self.current += step
        else:
            self.current = int(current)
        self.current = max(0, min(self.current, self.total))

        now = time.monotonic()
        if not force and self.current < self.total and now - self._last_draw < 0.2:
            return
        self._last_draw = now

        fraction = self.current / self.total
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.started_at
        sys.stdout.write(
            f"\r[{bar}] {self.current}/{self.total} "
            f"{fraction * 100:5.1f}% {self.label} ({elapsed:0.1f}s)"
        )
        sys.stdout.flush()

    def finish(self) -> None:
        self.update(self.total, force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
