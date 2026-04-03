"""Reward strategies for puzzle environments.

All reward callables share the signature::

    (prev_puzzle: AbstractPuzzle, action: Move, next_puzzle: AbstractPuzzle) -> float

This allows them to be swapped via dependency injection without changing
environment code.
"""
from __future__ import annotations

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle, Move


def _misplaced_facelets(puzzle: AbstractPuzzle) -> int:
    """Return the number of facelets not in their solved-state position.

    Computes this by comparing the puzzle state to the solved state. For NxN
    cubes the state shape is (6, n, n): face ``f`` should have all values equal
    to ``f`` in the solved state. For Megaminx the shape is (12, 11): face
    ``f`` should have all values equal to ``f``.

    Parameters
    ----------
    puzzle:
        Any AbstractPuzzle instance.

    Returns
    -------
    int
        Number of facelets that differ from the solved configuration.
    """
    state = puzzle.state
    _total = state.size
    if state.ndim == 3:
        # NxN cube: (6, n, n), solved face f has all values == f
        n = state.shape[1]
        solved = np.repeat(
            np.arange(6, dtype=state.dtype)[:, None, None],
            n * n,
            axis=0,
        ).reshape(6, n, n)
    else:
        # Megaminx: (12, 11), solved face f has all values == f
        solved = np.repeat(
            np.arange(12, dtype=state.dtype)[:, None],
            11,
            axis=1,
        )
    return int(np.sum(state != solved))


def _total_facelets(puzzle: AbstractPuzzle) -> int:
    """Return the total number of facelets in the puzzle state array.

    Parameters
    ----------
    puzzle:
        Any AbstractPuzzle instance.

    Returns
    -------
    int
        Total element count of the state array.
    """
    return puzzle.state.size


class SparseReward:
    """Binary reward: +1.0 when the puzzle is solved, 0.0 otherwise.

    This is the simplest reward signal and is suitable for short scramble
    depths where the agent can discover the solved state by chance.
    """

    def __call__(
        self,
        prev_puzzle: AbstractPuzzle,
        action: Move,
        next_puzzle: AbstractPuzzle,
    ) -> float:
        """Compute sparse reward.

        Parameters
        ----------
        prev_puzzle:
            Puzzle state before the action was applied.
        action:
            The move that was applied.
        next_puzzle:
            Puzzle state after the action was applied.

        Returns
        -------
        float
            1.0 if ``next_puzzle.is_solved``, else 0.0.
        """
        return 1.0 if next_puzzle.is_solved else 0.0

    def __repr__(self) -> str:
        return "SparseReward()"


class DenseReward:
    """Dense reward based on fraction of misplaced facelets.

    Returns a value in (-1, 0] where 0 means solved. The reward is:

    .. math::

        r = -\\frac{\\text{misplaced\\_facelets}}{\\text{total\\_facelets}}

    This gives the agent a continuous signal proportional to how far the
    puzzle is from solved, but it can encourage local optima (e.g. the agent
    learns to oscillate between states with slightly fewer misplaced tiles).
    """

    def __call__(
        self,
        prev_puzzle: AbstractPuzzle,
        action: Move,
        next_puzzle: AbstractPuzzle,
    ) -> float:
        """Compute dense reward.

        Parameters
        ----------
        prev_puzzle:
            Puzzle state before the action was applied.
        action:
            The move that was applied.
        next_puzzle:
            Puzzle state after the action was applied.

        Returns
        -------
        float
            Value in (-1.0, 0.0] where 0.0 means solved.
        """
        total = _total_facelets(next_puzzle)
        misplaced = _misplaced_facelets(next_puzzle)
        return -misplaced / total

    def __repr__(self) -> str:
        return "DenseReward()"


class PDTReward:
    """Potential-based dense reward with tolerance (PDT).

    Uses potential-based reward shaping to produce a dense signal that is
    theoretically guaranteed not to change the optimal policy (unlike naive
    dense reward, which can introduce reward-hacking local optima).

    The shaping follows Ng et al. (1999):

    .. math::

        r = \\gamma \\cdot V(s') - V(s)

    where :math:`V(s) = -\\text{misplaced\\_facelets}(s)` is the state-value
    potential. A solved state has potential 0 (the maximum). The ``gamma``
    parameter is the discount factor used during training; it must match the
    one used in the RL algorithm to preserve policy invariance.

    Parameters
    ----------
    gamma:
        Discount factor applied to the next-state potential. Default 0.99.
    """

    def __init__(self, gamma: float = 0.99) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1]. Got {gamma!r}.")
        self.gamma = gamma

    def _potential(self, puzzle: AbstractPuzzle) -> float:
        """Compute state potential V(s) = -misplaced_facelets(s).

        Parameters
        ----------
        puzzle:
            Any AbstractPuzzle instance.

        Returns
        -------
        float
            Non-positive scalar; 0.0 for a solved puzzle.
        """
        return -float(_misplaced_facelets(puzzle))

    def __call__(
        self,
        prev_puzzle: AbstractPuzzle,
        action: Move,
        next_puzzle: AbstractPuzzle,
    ) -> float:
        """Compute potential-based shaped reward.

        Parameters
        ----------
        prev_puzzle:
            Puzzle state before the action was applied.
        action:
            The move that was applied.
        next_puzzle:
            Puzzle state after the action was applied.

        Returns
        -------
        float
            Shaped reward :math:`\\gamma V(s') - V(s)`.
        """
        return self.gamma * self._potential(next_puzzle) - self._potential(prev_puzzle)

    def __repr__(self) -> str:
        return f"PDTReward(gamma={self.gamma!r})"
