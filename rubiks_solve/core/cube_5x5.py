"""
5x5 Rubik's cube (Professor's Cube) implementation.

Subclasses CubeNNN with n=5.  72 moves (outer + wide-slice + middle-slice
+ whole-cube rotations + three-layer-wide moves).
Even-n cubes can have parity errors; the 5x5 can also have wing-edge parity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rubiks_solve.core.cube_nnn import CubeNNN
from rubiks_solve.core.validator import has_parity_error as _has_parity


class Cube5x5(CubeNNN):
    """5x5x5 Rubik's Professor Cube.

    State: numpy array shape (6, 5, 5) dtype uint8.
    Legal moves: 72 (outer + inner slices + rotations + wide moves).
    God's number: ~60 moves in HTM (practical limit used here).
    """

    _DEFAULT_N: int = 5

    def __init__(self, state: Optional[np.ndarray] = None) -> None:
        """Initialise the 5x5 cube.

        Args:
            state: Optional (6, 5, 5) uint8 state array.  Defaults to solved.
        """
        super().__init__(5, state)

    def _new_instance(self, new_state: np.ndarray) -> "Cube5x5":
        """Create a new Cube5x5 with the given state array.

        Args:
            new_state: The (6, 5, 5) uint8 array for the new instance.

        Returns:
            New Cube5x5 with the provided state.
        """
        return Cube5x5(new_state)

    @classmethod
    def solved_state(cls) -> "Cube5x5":
        """Return a new Cube5x5 in the canonical solved state.

        Returns:
            Cube5x5 instance with all stickers in their home positions.
        """
        return cls()

    @classmethod
    def move_limit(cls) -> int:
        """Maximum move budget for solving a 5x5 in HTM.

        Returns:
            60
        """
        return 60

    @classmethod
    def puzzle_name(cls) -> str:
        """Human-readable puzzle name.

        Returns:
            '5x5'
        """
        return "5x5"

    @property
    def has_parity_error(self) -> bool:
        """Detect whether this cube state contains a parity error.

        The 5x5 is an odd-n cube, so it does not exhibit the same OLL/PLL
        parity as even-n cubes.  However, the inner-wing edges can still end
        up in a parity state relative to the centres.  This property delegates
        to the validator module for a consistent implementation.

        Returns:
            True if a parity error is detected, False otherwise.
        """
        return _has_parity(self)
