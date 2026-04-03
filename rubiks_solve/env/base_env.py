"""Abstract Gymnasium environment for puzzle solving."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.encoding.base import AbstractStateEncoder


# Type alias for reward functions.
RewardFn = Callable[[AbstractPuzzle, Move, AbstractPuzzle], float]


class AbstractPuzzleEnv(gym.Env, ABC):
    """Abstract Gymnasium environment for a puzzle-solving task.

    Sub-classes must implement :meth:`_build_info`, which provides
    environment-specific entries for the ``info`` dict returned by
    :meth:`step` and :meth:`reset`.

    Parameters
    ----------
    puzzle_type:
        The AbstractPuzzle subclass this environment operates on.
    encoder:
        State encoder that converts puzzle states to float32 numpy arrays.
    reward_fn:
        Callable ``(prev, action, next) -> float`` that computes step rewards.
    scramble_depth:
        Number of random moves applied to produce the initial scrambled state.
    max_steps:
        Maximum number of steps per episode before truncation.
    """

    metadata: dict[str, Any] = {"render_modes": ["ansi"]}
    render_mode: str = "ansi"

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        encoder: AbstractStateEncoder,
        reward_fn: RewardFn,
        scramble_depth: int,
        max_steps: int,
    ) -> None:
        super().__init__()
        self._puzzle_type = puzzle_type
        self._encoder = encoder
        self._reward_fn = reward_fn
        self._scramble_depth = scramble_depth
        self._max_steps = max_steps

        # Derive action space from the legal moves of the solved puzzle.
        solved = puzzle_type.solved_state()
        self._legal_moves: list[Move] = solved.legal_moves()
        self.action_space: spaces.Discrete = spaces.Discrete(len(self._legal_moves))

        # Observation space is a flat float32 box in [0, 1].
        low = np.zeros(encoder.output_shape, dtype=np.float32)
        high = np.ones(encoder.output_shape, dtype=np.float32)
        self.observation_space: spaces.Box = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        # Episode state (initialised properly in reset()).
        self._puzzle: AbstractPuzzle = solved
        self._steps_taken: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Abstract hooks for sub-classes
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_info(self) -> dict[str, Any]:
        """Return environment-specific entries for the info dictionary.

        Called by :meth:`step` and :meth:`reset` to populate the ``info``
        dict. Sub-classes should include at minimum:
        ``scramble_depth``, ``moves_taken``, ``is_solved``.
        """

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        _options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a freshly scrambled puzzle.

        Parameters
        ----------
        seed:
            Optional RNG seed for reproducibility.
        _options:
            Unused; accepted for Gymnasium API compatibility.

        Returns
        -------
        observation : np.ndarray
            Encoded state of the scrambled puzzle.
        info : dict[str, Any]
            Environment-specific diagnostic info.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        solved = self._puzzle_type.solved_state()
        self._puzzle = solved.scramble(self._scramble_depth, self._rng)
        self._steps_taken = 0

        obs = self._encoder.encode(self._puzzle)
        return obs, self._build_info()

    def step(
        self, action_idx: int | np.intp
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply one action and advance the environment by one step.

        Parameters
        ----------
        action_idx:
            Index into ``self._legal_moves`` selecting the move to apply.

        Returns
        -------
        observation : np.ndarray
            Encoded state after the move.
        reward : float
            Reward signal from the configured reward function.
        terminated : bool
            True when the puzzle is solved.
        truncated : bool
            True when ``max_steps`` has been reached without solving.
        info : dict[str, Any]
            Environment-specific diagnostic info.
        """
        action_idx = int(action_idx)
        move = self._legal_moves[action_idx]

        prev_puzzle = self._puzzle
        self._puzzle = self._puzzle.apply_move(move)
        self._steps_taken += 1

        reward = self._reward_fn(prev_puzzle, move, self._puzzle)
        terminated = bool(self._puzzle.is_solved)
        truncated = (not terminated) and (self._steps_taken >= self._max_steps)

        obs = self._encoder.encode(self._puzzle)
        return obs, reward, terminated, truncated, self._build_info()

    def render(self) -> str:
        """Return a human-readable string representation of the current state.

        Returns
        -------
        str
            Multi-line text describing the puzzle's current state.
        """
        state = self._puzzle.state
        lines: list[str] = [
            f"Puzzle: {self._puzzle_type.puzzle_name()}",
            f"Steps: {self._steps_taken}/{self._max_steps}",
            f"Solved: {self._puzzle.is_solved}",
            f"State:\n{state}",
        ]
        return "\n".join(lines)
