"""Wall-clock timing utilities."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator


@dataclass
class TimerResult:
    """Holds the elapsed time measured by :func:`timer`.

    Attributes:
        elapsed_seconds: Wall-clock seconds elapsed inside the ``with`` block.
    """

    elapsed_seconds: float

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds.

        Returns:
            ``elapsed_seconds * 1000``.
        """
        return self.elapsed_seconds * 1000


@contextmanager
def timer() -> Generator[TimerResult, None, None]:
    """Context manager that measures wall-clock elapsed time.

    Yields a :class:`TimerResult` whose ``elapsed_seconds`` attribute is
    populated *after* the ``with`` block exits.

    Usage::

        with timer() as t:
            do_work()
        print(t.elapsed_seconds)   # populated after the block
        print(t.elapsed_ms)        # milliseconds convenience property

    Yields:
        :class:`TimerResult` (``elapsed_seconds`` is 0.0 while inside the block).
    """
    result = TimerResult(elapsed_seconds=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start
