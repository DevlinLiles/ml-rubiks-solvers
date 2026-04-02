"""
2x2 Rubik's cube (Pocket Cube) implementation.

Subclasses CubeNNN with n=2.  Only 9 canonical moves are legal (U, R, F in
CW / CCW / 180 variants) because the remaining moves are equivalent by
whole-cube symmetry when one corner is held fixed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rubiks_solve.core.cube_nnn import CubeNNN


class Cube2x2(CubeNNN):
    """2x2x2 Rubik's Pocket Cube.

    State: numpy array shape (6, 2, 2) dtype uint8.
    Legal moves: 9 (U/R/F in CW, CCW, 180 variants).
    God's number: 11 moves in HTM.
    """

    _DEFAULT_N: int = 2

    def __init__(self, state: Optional[np.ndarray] = None) -> None:
        """Initialise the 2x2 cube.

        Args:
            state: Optional (6, 2, 2) uint8 state array.  Defaults to solved.
        """
        super().__init__(2, state)

    def _new_instance(self, new_state: np.ndarray) -> "Cube2x2":
        """Create a new Cube2x2 with the given state array.

        Args:
            new_state: The (6, 2, 2) uint8 array for the new instance.

        Returns:
            New Cube2x2 with the provided state.
        """
        return Cube2x2(new_state)

    @classmethod
    def solved_state(cls) -> "Cube2x2":
        """Return a new Cube2x2 in the canonical solved state.

        Returns:
            Cube2x2 instance with all stickers in their home positions.
        """
        return cls()

    @classmethod
    def move_limit(cls) -> int:
        """Maximum move budget for solving a 2x2 (God's number in HTM).

        Returns:
            11
        """
        return 11

    @classmethod
    def puzzle_name(cls) -> str:
        """Human-readable puzzle name.

        Returns:
            '2x2'
        """
        return "2x2"
