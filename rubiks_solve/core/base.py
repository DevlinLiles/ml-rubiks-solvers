"""
Core abstract interfaces for all puzzle types and state encoders.
All puzzles are IMMUTABLE — apply_move returns a new instance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Move:
    """Represents a single puzzle move.

    For NxN cubes:
        face: one of "U","D","L","R","F","B"
        layer: 0 = outermost face, 1 = first inner slice, etc.
        direction: +1 clockwise, -1 counter-clockwise (viewed from outside)
        double: True for 180-degree turns (counts as 1 in HTM)

    For Megaminx:
        face: one of "U","BL","BR","L","R","DL","DR","DBL","DBR","DB","F","DF"
        layer: always 0 (Megaminx has no inner slices)
        direction: +1 clockwise, -1 counter-clockwise
        double: False (Megaminx uses 72-degree turns, no concept of double)
    """

    name: str          # Human-readable label, e.g. "R", "U'", "Fw2", "U++"
    face: str          # Face identifier
    layer: int         # 0-indexed from outer face
    direction: int     # +1 CW, -1 CCW
    double: bool = False  # 180-degree turn (HTM counts as 1 move)

    def inverse(self) -> Move:
        """Return the inverse of this move."""
        if self.double:
            return self  # 180-degree is its own inverse
        inv_name = self.name.rstrip("'") + ("" if self.name.endswith("'") else "'")
        return Move(
            name=inv_name,
            face=self.face,
            layer=self.layer,
            direction=-self.direction,
            double=False,
        )

    def __repr__(self) -> str:
        return f"Move({self.name!r})"


class AbstractPuzzle(ABC):
    """Base class for all puzzle types.

    Contract:
    - Puzzles are IMMUTABLE. apply_move() always returns a new instance.
    - state property returns a numpy array suitable for encoding.
    - legal_moves() returns the full set of valid moves for this puzzle type.
    - scramble() returns a new instance reached by applying n_moves random moves.
    """

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """Canonical state array. Shape and dtype are puzzle-specific.

        NxN cubes: shape (6, n, n) uint8, values 0-5 (color index).
        Megaminx:  shape (12, 11) uint8, values 0-11 (color index).
        """

    @property
    @abstractmethod
    def is_solved(self) -> bool:
        """True iff the puzzle is in the solved (goal) state."""

    @abstractmethod
    def apply_move(self, move: Move) -> AbstractPuzzle:
        """Return a NEW puzzle with the given move applied. Do not mutate self."""

    @abstractmethod
    def legal_moves(self) -> list[Move]:
        """Return all legal moves for this puzzle type (constant, not state-dependent)."""

    @abstractmethod
    def scramble(self, n_moves: int, rng: np.random.Generator) -> AbstractPuzzle:
        """Return a new instance reached by applying n_moves random moves.

        Avoids immediately inverting the previous move to prevent trivial sequences.
        """

    @abstractmethod
    def copy(self) -> AbstractPuzzle:
        """Return a deep copy of this puzzle instance."""

    @classmethod
    @abstractmethod
    def solved_state(cls) -> AbstractPuzzle:
        """Return a new instance in the canonical solved state."""

    @classmethod
    @abstractmethod
    def move_limit(cls) -> int:
        """Maximum move budget for solving (Half Turn Metric).

        Limits per puzzle type:
          2x2  -> 11
          3x3  -> 20
          4x4  -> 40
          5x5  -> 60
          Megaminx -> 70
        """

    @classmethod
    @abstractmethod
    def puzzle_name(cls) -> str:
        """Short human-readable name, e.g. '3x3', 'megaminx'."""

    def apply_moves(self, moves: list[Move]) -> AbstractPuzzle:
        """Apply a sequence of moves, returning the final state."""
        result = self
        for move in moves:
            result = result.apply_move(move)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractPuzzle):
            return NotImplemented
        return np.array_equal(self.state, other.state)

    def __hash__(self) -> int:
        return hash(self.state.tobytes())
