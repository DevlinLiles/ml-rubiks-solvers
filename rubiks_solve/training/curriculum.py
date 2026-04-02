"""Curriculum scheduler for progressive scramble-depth training."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CurriculumConfig:
    """Configuration for :class:`ScrambleCurriculum`.

    Attributes:
        min_depth:           Starting scramble depth; also the floor of the
                             depth range.
        max_depth:           Maximum scramble depth the curriculum will ever
                             reach.  Typically set to ``puzzle.move_limit()``.
        increase_threshold:  Fraction of recent attempts that must be solved
                             before the current depth is incremented.
        increase_step:       How many depth levels to advance when the threshold
                             is met.
        eval_window:         Number of recent solve attempts to include when
                             computing the rolling solve rate.
    """

    min_depth: int = 1
    max_depth: int = 20
    increase_threshold: float = 0.8
    increase_step: int = 1
    eval_window: int = 100


class ScrambleCurriculum:
    """Manage scramble depth scheduling during training.

    Starts at ``config.min_depth``.  After each call to
    :meth:`maybe_increase_depth`, if the rolling solve rate over the last
    ``config.eval_window`` attempts exceeds ``config.increase_threshold`` **and**
    the current depth is below ``config.max_depth``, the depth is incremented by
    ``config.increase_step`` (clamped to ``max_depth``).

    Args:
        config: A :class:`CurriculumConfig` instance.

    Example::

        cfg = CurriculumConfig(min_depth=1, max_depth=20, increase_threshold=0.8)
        curriculum = ScrambleCurriculum(cfg)

        for episode in range(10_000):
            depth = curriculum.sample_depth(rng)
            solved = run_episode(depth)
            curriculum.record_attempt(solved)
            curriculum.maybe_increase_depth()
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self._config = config
        self._current_depth: int = config.min_depth
        # Fixed-length window of bool outcomes (True = solved).
        self._window: deque[bool] = deque(maxlen=config.eval_window)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_depth(self) -> int:
        """The current maximum scramble depth used for depth sampling."""
        return self._current_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_attempt(self, solved: bool) -> None:
        """Record the outcome of a single solve attempt.

        Args:
            solved: ``True`` if the agent solved the scrambled puzzle,
                    ``False`` otherwise.
        """
        self._window.append(solved)

    def current_solve_rate(self) -> float:
        """Return the rolling solve rate over the last *eval_window* attempts.

        Returns:
            A float in ``[0.0, 1.0]``.  Returns ``0.0`` when no attempts have
            been recorded yet.
        """
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def maybe_increase_depth(self) -> bool:
        """Increase the current depth if the solve-rate threshold is met.

        The depth is only increased when:

        * The rolling window has at least ``eval_window`` entries.
        * :meth:`current_solve_rate` exceeds ``increase_threshold``.
        * ``current_depth`` is strictly below ``max_depth``.

        When those conditions are met, ``current_depth`` is incremented by
        ``increase_step`` (clamped to ``max_depth``) and the rolling window is
        **cleared** so the next evaluation starts fresh at the new difficulty.

        Returns:
            ``True`` if the depth was increased, ``False`` otherwise.
        """
        cfg = self._config
        if (
            len(self._window) >= cfg.eval_window
            and self.current_solve_rate() > cfg.increase_threshold
            and self._current_depth < cfg.max_depth
        ):
            self._current_depth = min(
                self._current_depth + cfg.increase_step, cfg.max_depth
            )
            self._window.clear()
            return True
        return False

    def sample_depth(self, rng: np.random.Generator) -> int:
        """Sample a scramble depth uniformly from ``[min_depth, current_depth]``.

        Args:
            rng: NumPy random generator used for sampling.

        Returns:
            An integer depth in ``[config.min_depth, current_depth]``.
        """
        return int(rng.integers(self._config.min_depth, self._current_depth + 1))
