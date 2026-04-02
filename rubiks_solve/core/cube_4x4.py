"""
4x4 Rubik's cube (Revenge Cube) implementation.

Subclasses CubeNNN with n=4.  45 moves (18 outer + 18 wide-slice + 9 rotations).
Even-n cubes can exhibit OLL/PLL parity — detected via `has_parity_error`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rubiks_solve.core.cube_nnn import CubeNNN


class Cube4x4(CubeNNN):
    """4x4x4 Rubik's Revenge Cube.

    State: numpy array shape (6, 4, 4) dtype uint8.
    Legal moves: 45 (outer + inner slice + whole-cube rotations).
    God's number: ~40 moves in HTM (practical limit used here).

    Even-n cubes may reach states that look one-move-away from solved on the
    outer layer but require extra moves to resolve — parity errors.
    """

    _DEFAULT_N: int = 4

    def __init__(self, state: Optional[np.ndarray] = None) -> None:
        """Initialise the 4x4 cube.

        Args:
            state: Optional (6, 4, 4) uint8 state array.  Defaults to solved.
        """
        super().__init__(4, state)

    def _new_instance(self, new_state: np.ndarray) -> "Cube4x4":
        """Create a new Cube4x4 with the given state array.

        Args:
            new_state: The (6, 4, 4) uint8 array for the new instance.

        Returns:
            New Cube4x4 with the provided state.
        """
        return Cube4x4(new_state)

    @classmethod
    def solved_state(cls) -> "Cube4x4":
        """Return a new Cube4x4 in the canonical solved state.

        Returns:
            Cube4x4 instance with all stickers in their home positions.
        """
        return cls()

    @classmethod
    def move_limit(cls) -> int:
        """Maximum move budget for solving a 4x4 in HTM.

        Returns:
            40
        """
        return 40

    @classmethod
    def puzzle_name(cls) -> str:
        """Human-readable puzzle name.

        Returns:
            '4x4'
        """
        return "4x4"

    @property
    def has_parity_error(self) -> bool:
        """Detect whether this cube state contains an OLL or PLL parity error.

        Even-n cubes (4x4, 6x6, …) can reach states that are unreachable on
        a 3x3 and require special parity algorithms.  True parity detection
        requires tracking the permutation parity of the inner-slice edges,
        which is non-trivial from sticker colours alone.

        This implementation uses a heuristic: it checks whether any pair of
        identically-coloured edge stickers on the inner rings are swapped in a
        way that would be impossible on an odd-n cube.  A full, exact
        implementation would use cycle-counting on the permutation group.

        Returns:
            True if a parity error is detected, False otherwise.  May return
            False negatives for complex scrambles (simplified heuristic).
        """
        from rubiks_solve.core.validator import has_parity_error as _has_parity
        return _has_parity(self)
