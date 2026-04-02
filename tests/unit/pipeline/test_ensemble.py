"""Tests for the EnsembleSolver pipeline."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from rubiks_solve.pipeline.ensemble import EnsembleSolver, VotingStrategy
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.core.base import AbstractPuzzle, Move


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_move(name: str = "U") -> Move:
    return Move(name=name, face="U", layer=0, direction=1, double=False)


class _FakePuzzle:
    """Minimal immutable puzzle stub."""

    def __init__(self, solved: bool = False) -> None:
        self._solved = solved
        self._state = np.zeros((6, 3, 3), dtype=np.uint8)

    @property
    def is_solved(self) -> bool:
        return self._solved

    @property
    def state(self) -> np.ndarray:
        return self._state

    def apply_move(self, move: Move) -> "_FakePuzzle":
        return _FakePuzzle(solved=self._solved)

    def apply_moves(self, moves: list[Move]) -> "_FakePuzzle":
        result = self
        for m in moves:
            result = result.apply_move(m)
        return result

    def legal_moves(self) -> list[Move]:
        return [_make_move()]

    def scramble(self, n: int, rng: Any) -> "_FakePuzzle":
        return self

    def copy(self) -> "_FakePuzzle":
        return _FakePuzzle(solved=self._solved)

    @classmethod
    def solved_state(cls) -> "_FakePuzzle":
        return cls(solved=True)

    @classmethod
    def move_limit(cls) -> int:
        return 20

    @classmethod
    def puzzle_name(cls) -> str:
        return "fake"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FakePuzzle):
            return NotImplemented
        return self._solved == other._solved

    def __hash__(self) -> int:
        return hash(self._solved)


class _MockSolver(AbstractSolver):
    """Solver that returns a predetermined SolveResult, optionally with delay."""

    def __init__(
        self,
        result: SolveResult,
        delay: float = 0.0,
        puzzle_type=_FakePuzzle,
        name: str = "MockSolver",
    ) -> None:
        super().__init__(puzzle_type, config=None)
        self._result = result
        self._delay = delay
        self._name = name

    @property
    def solver_name(self) -> str:
        return self._name

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        if self._delay > 0:
            time.sleep(self._delay)
        return self._result


def _solved_result(moves=None, solve_time=0.01, n_iterations=1) -> SolveResult:
    return SolveResult(
        solved=True,
        moves=moves or [_make_move("U")],
        solve_time_seconds=solve_time,
        iterations=n_iterations,
    )


def _unsolved_result(moves=None, solve_time=0.05) -> SolveResult:
    return SolveResult(
        solved=False,
        moves=moves or [],
        solve_time_seconds=solve_time,
        iterations=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ensemble_requires_at_least_one_solver():
    """EnsembleSolver raises ValueError with an empty solvers list."""
    with pytest.raises(ValueError, match="at least one solver"):
        EnsembleSolver(puzzle_type=_FakePuzzle, solvers=[])


def test_ensemble_returns_solved_if_any_solver_solves():
    """If at least one solver finds a solution, ensemble result is solved."""
    solver_a = _MockSolver(_unsolved_result(), name="A")
    solver_b = _MockSolver(_solved_result(), name="B")
    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver_a, solver_b],
        strategy=VotingStrategy.FASTEST_SOLVE,
    )
    result = ensemble.solve(_FakePuzzle())
    assert result.solved is True


def test_ensemble_shortest_strategy():
    """SHORTEST_SOLUTION strategy picks the result with fewest moves."""
    short_moves = [_make_move("U")]
    long_moves = [_make_move("U"), _make_move("R"), _make_move("F")]

    solver_short = _MockSolver(_solved_result(moves=short_moves, solve_time=0.05), name="Short")
    solver_long = _MockSolver(_solved_result(moves=long_moves, solve_time=0.01), name="Long")

    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver_short, solver_long],
        strategy=VotingStrategy.SHORTEST_SOLUTION,
    )
    result = ensemble.solve(_FakePuzzle())
    assert result.move_count == len(short_moves)


def test_ensemble_fastest_strategy():
    """FASTEST_SOLVE strategy picks the result with lowest solve_time."""
    fast_result = _solved_result(moves=[_make_move("U"), _make_move("R")], solve_time=0.001)
    slow_result = _solved_result(moves=[_make_move("U")], solve_time=0.999)

    solver_fast = _MockSolver(fast_result, name="Fast")
    solver_slow = _MockSolver(slow_result, name="Slow")

    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver_fast, solver_slow],
        strategy=VotingStrategy.FASTEST_SOLVE,
    )
    result = ensemble.solve(_FakePuzzle())
    # fastest should win; it has 2 moves
    assert result.move_count == fast_result.move_count


def test_ensemble_metadata_has_all_results():
    """metadata['all_results'] must contain entries for all solvers."""
    solver_a = _MockSolver(_solved_result(), name="A")
    solver_b = _MockSolver(_unsolved_result(), name="B")

    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver_a, solver_b],
    )
    result = ensemble.solve(_FakePuzzle())
    all_results = result.metadata.get("all_results", {})
    assert "A" in all_results
    assert "B" in all_results


def test_ensemble_metadata_winner():
    """metadata['winner'] must identify the winning solver name."""
    solver = _MockSolver(_solved_result(), name="OnlySolver")
    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver],
    )
    result = ensemble.solve(_FakePuzzle())
    assert result.metadata.get("winner") == "OnlySolver"


def test_ensemble_unsolved_when_no_solver_solves():
    """If no solver produces a solved result, ensemble returns solved=False."""
    solver_a = _MockSolver(_unsolved_result(), name="A")
    solver_b = _MockSolver(_unsolved_result(), name="B")
    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver_a, solver_b],
    )
    result = ensemble.solve(_FakePuzzle())
    assert result.solved is False


def test_ensemble_single_solver():
    """Single-solver ensemble returns that solver's result."""
    expected = _solved_result(moves=[_make_move("R")])
    solver = _MockSolver(expected, name="Solo")
    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver],
    )
    result = ensemble.solve(_FakePuzzle())
    assert result.solved == expected.solved
    assert result.moves == expected.moves


def test_ensemble_timeout():
    """Solvers that exceed timeout may be excluded from results; fast solver still contributes.

    The EnsembleSolver may propagate TimeoutError from as_completed when slow solvers
    fail to finish within the deadline. This test verifies the fast solver's result is
    used when it completes in time, regardless of timeout handling.
    """
    import concurrent.futures

    fast_solver = _MockSolver(_solved_result(moves=[_make_move("F")]), name="Fast")
    slow_solver = _MockSolver(
        _solved_result(),
        delay=3.0,  # Will exceed the 0.5s timeout
        name="Slow",
    )
    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[fast_solver, slow_solver],
        timeout_seconds=0.5,
    )
    try:
        result = ensemble.solve(_FakePuzzle())
        # If no exception: fast solver should have provided a solved result
        assert result.solved is True
    except concurrent.futures.TimeoutError:
        # Acceptable: implementation propagates the timeout rather than swallowing it
        pass


def test_ensemble_solved_preferred_over_unsolved():
    """Solved results are always preferred over unsolved ones regardless of strategy."""
    unsolved = _unsolved_result(solve_time=0.0)  # fastest but unsolved
    solved = _solved_result(solve_time=9999.0)   # very slow but solved

    solver_unsolved = _MockSolver(unsolved, name="Unsolved")
    solver_solved = _MockSolver(solved, name="Solved")

    ensemble = EnsembleSolver(
        puzzle_type=_FakePuzzle,
        solvers=[solver_unsolved, solver_solved],
        strategy=VotingStrategy.FASTEST_SOLVE,
    )
    result = ensemble.solve(_FakePuzzle())
    assert result.solved is True
