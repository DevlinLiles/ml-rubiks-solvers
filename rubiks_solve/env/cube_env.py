"""Concrete Gymnasium environment for NxN cube puzzles."""
from __future__ import annotations

from typing import Any

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.env.base_env import AbstractPuzzleEnv, RewardFn


class CubeEnv(AbstractPuzzleEnv):
    """Gymnasium environment for NxN Rubik's cube puzzles.

    Inherits the full Gymnasium loop (reset/step/render) from
    :class:`~rubiks_solve.env.base_env.AbstractPuzzleEnv` and adds a richer
    ``info`` dictionary that includes scramble depth, moves taken so far, and
    whether the current state is solved.

    Parameters
    ----------
    puzzle_type:
        The AbstractPuzzle subclass representing the cube variant (e.g. a 3x3
        or 2x2 implementation).
    encoder:
        State encoder that converts puzzle states to float32 numpy arrays.
    reward_fn:
        Callable ``(prev, action, next) -> float`` that computes step rewards.
    scramble_depth:
        Number of random moves applied to produce the initial scrambled state.
    max_steps:
        Maximum number of steps per episode before truncation.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        encoder: AbstractStateEncoder,
        reward_fn: RewardFn,
        scramble_depth: int,
        max_steps: int,
    ) -> None:
        super().__init__(
            puzzle_type=puzzle_type,
            encoder=encoder,
            reward_fn=reward_fn,
            scramble_depth=scramble_depth,
            max_steps=max_steps,
        )

    def _build_info(self) -> dict[str, Any]:
        """Return cube-specific diagnostic information.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            ``"scramble_depth"``
                The number of random moves used to scramble the puzzle at
                episode start.
            ``"moves_taken"``
                The number of actions taken in the current episode so far.
            ``"is_solved"``
                Whether the current puzzle state is the solved state.
        """
        return {
            "scramble_depth": self._scramble_depth,
            "moves_taken": self._steps_taken,
            "is_solved": bool(self._puzzle.is_solved),
        }
