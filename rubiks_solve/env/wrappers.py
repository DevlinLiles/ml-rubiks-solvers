"""Gymnasium environment wrappers for curriculum learning and other augmentations."""
from __future__ import annotations

import gymnasium as gym

from rubiks_solve.env.base_env import AbstractPuzzleEnv


class ScrambleDepthCurriculum(gym.Wrapper):
    """Curriculum wrapper that gradually increases scramble depth as the agent improves.

    Starts training with ``min_depth`` scramble moves. After each epoch the
    caller should invoke :meth:`update_curriculum` with the agent's current
    solve rate. When the rate exceeds ``threshold`` the scramble depth
    increases by one (up to ``max_depth``).

    The wrapped environment must be an instance of
    :class:`~rubiks_solve.env.base_env.AbstractPuzzleEnv` so that
    ``_scramble_depth`` can be updated in place.

    Parameters
    ----------
    env:
        An :class:`~rubiks_solve.env.base_env.AbstractPuzzleEnv` instance to
        wrap.
    min_depth:
        Starting scramble depth (must be >= 1).
    max_depth:
        Maximum scramble depth the curriculum will advance to.
    threshold:
        Solve-rate threshold in [0, 1] above which the depth is increased.
        Default is 0.8 (80 % solve rate).

    Raises
    ------
    TypeError
        If ``env`` is not an :class:`~rubiks_solve.env.base_env.AbstractPuzzleEnv`.
    ValueError
        If ``min_depth`` > ``max_depth`` or ``threshold`` is outside [0, 1].
    """

    def __init__(
        self,
        env: gym.Env,
        min_depth: int,
        max_depth: int,
        threshold: float = 0.8,
    ) -> None:
        if not isinstance(env, AbstractPuzzleEnv):
            raise TypeError(
                f"ScrambleDepthCurriculum requires an AbstractPuzzleEnv, "
                f"got {type(env).__name__!r}."
            )
        if min_depth > max_depth:
            raise ValueError(
                f"min_depth ({min_depth}) must be <= max_depth ({max_depth})."
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1]. Got {threshold!r}."
            )

        super().__init__(env)
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._threshold = threshold

        # Set the initial scramble depth.
        self.env._scramble_depth = min_depth  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_depth(self) -> int:
        """Current scramble depth being used by the wrapped environment.

        Returns
        -------
        int
            The scramble depth currently configured on the inner environment.
        """
        return self.env._scramble_depth  # type: ignore[attr-defined]

    def update_curriculum(self, solve_rate: float) -> None:
        """Advance the scramble depth if the agent's solve rate is high enough.

        Call this once per training epoch with the fraction of episodes solved
        during that epoch. If ``solve_rate > threshold`` and the current depth
        is below ``max_depth``, the depth is incremented by one.

        Parameters
        ----------
        solve_rate:
            Fraction of episodes solved in the most recent epoch, in [0, 1].
        """
        if solve_rate > self._threshold and self.current_depth < self._max_depth:
            self.env._scramble_depth += 1  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return (
            f"ScrambleDepthCurriculum("
            f"depth={self.current_depth}, "
            f"min={self._min_depth}, "
            f"max={self._max_depth}, "
            f"threshold={self._threshold!r})"
        )
