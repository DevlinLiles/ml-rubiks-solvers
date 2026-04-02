"""
3x3 Rubik's cube (Standard Cube) implementation.

Subclasses CubeNNN with n=3.  18 standard HTM moves (6 faces × CW/CCW/180).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rubiks_solve.core.cube_nnn import CubeNNN


class Cube3x3(CubeNNN):
    """3x3x3 Standard Rubik's Cube.

    State: numpy array shape (6, 3, 3) dtype uint8.
    Legal moves: 18 (U/D/L/R/F/B each in CW, CCW, 180 variants).
    God's number: 20 moves in HTM.
    """

    _DEFAULT_N: int = 3

    def __init__(self, state: Optional[np.ndarray] = None) -> None:
        """Initialise the 3x3 cube.

        Args:
            state: Optional (6, 3, 3) uint8 state array.  Defaults to solved.
        """
        super().__init__(3, state)

    def _new_instance(self, new_state: np.ndarray) -> "Cube3x3":
        """Create a new Cube3x3 with the given state array.

        Args:
            new_state: The (6, 3, 3) uint8 array for the new instance.

        Returns:
            New Cube3x3 with the provided state.
        """
        return Cube3x3(new_state)

    @classmethod
    def solved_state(cls) -> "Cube3x3":
        """Return a new Cube3x3 in the canonical solved state.

        Returns:
            Cube3x3 instance with all faces solved.
        """
        return cls()

    @classmethod
    def move_limit(cls) -> int:
        """Maximum move budget for solving a 3x3 (God's number in HTM).

        Returns:
            20
        """
        return 20

    @classmethod
    def puzzle_name(cls) -> str:
        """Human-readable puzzle name.

        Returns:
            '3x3'
        """
        return "3x3"
