"""Abstract solver interface and result types."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rubiks_solve.core.base import AbstractPuzzle, Move


@dataclass
class SolveResult:
    """Result returned by any solver.

    Fields:
        solved:               Whether the puzzle was solved.
        moves:                The solution move sequence (may be empty if not solved).
        solve_time_seconds:   Wall-clock time spent in solve().
        iterations:           Algorithm-specific iteration count (generations,
                              episodes, MCTS rollouts, etc.).
        metadata:             Algorithm-specific extra info (fitness history,
                              loss values, etc.).
    """

    solved: bool
    moves: list[Move]
    solve_time_seconds: float
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def move_count(self) -> int:
        """Return the number of moves in the solution sequence."""
        return len(self.moves)

    def verify(self, initial_puzzle: AbstractPuzzle) -> bool:
        """Independently verify that replaying moves from initial_puzzle yields solved state.

        Always use this in tests — do not trust solved=True alone.
        """
        result = initial_puzzle.apply_moves(self.moves)
        return result.is_solved


class AbstractSolver(ABC):
    """Base class for all solver algorithms.

    All solvers share this interface so they can be composed in SolverChain
    and EnsembleSolver without knowing each other's internals.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        config: Any,
    ) -> None:
        self.puzzle_type = puzzle_type
        self.config = config
        self._logger = logging.getLogger(self.__class__.__qualname__)

    @abstractmethod
    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Attempt to solve the puzzle. Returns a SolveResult whether or not solved."""

    def supports_puzzle(self, puzzle: AbstractPuzzle) -> bool:
        """Return True if this solver can handle the given puzzle instance."""
        return isinstance(puzzle, self.puzzle_type)

    @property
    def solver_name(self) -> str:
        """Return the human-readable name of this solver class."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.solver_name}(puzzle={self.puzzle_type.__name__})"
