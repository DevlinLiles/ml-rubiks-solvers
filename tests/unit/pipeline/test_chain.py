"""Tests for the SolverChain pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import numpy as np
import pytest

from rubiks_solve.pipeline.chain import SolverChain, StageConfig
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.core.base import AbstractPuzzle, Move


# ---------------------------------------------------------------------------
# Helpers: minimal mock puzzle and solver
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
    """Solver that returns a predetermined SolveResult."""

    def __init__(self, result: SolveResult, puzzle_type=_FakePuzzle) -> None:
        super().__init__(puzzle_type, config=None)
        self._result = result
        self.call_count = 0

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        self.call_count += 1
        return self._result


def _solved_result(moves=None) -> SolveResult:
    return SolveResult(
        solved=True,
        moves=moves or [_make_move("U")],
        solve_time_seconds=0.01,
        iterations=1,
    )


def _unsolved_result(moves=None) -> SolveResult:
    return SolveResult(
        solved=False,
        moves=moves or [_make_move("R")],
        solve_time_seconds=0.02,
        iterations=5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chain_requires_at_least_one_stage():
    """SolverChain raises ValueError with an empty stages list."""
    with pytest.raises(ValueError, match="at least one stage"):
        SolverChain(puzzle_type=_FakePuzzle, stages=[], config=None)


def test_chain_with_single_solver_solved():
    """A single-stage chain behaves like the underlying solver when it solves."""
    result = _solved_result()
    solver = _MockSolver(result)

    # The solver returns solved=True with one move.
    # The chain checks by re-applying moves; _FakePuzzle.apply_move always returns
    # a non-solved puzzle — so we need a puzzle that is already solved to start.
    solved_puzzle = _FakePuzzle(solved=True)
    # The result moves being applied to a solved puzzle should leave it solved.
    # Override apply_move to preserve solved status:
    stage = StageConfig(solver=solver, move_budget=10)
    chain = SolverChain(puzzle_type=_FakePuzzle, stages=[stage])

    # Use a puzzle whose apply_moves always returns solved
    class _AlwaysSolvedPuzzle(_FakePuzzle):
        def apply_move(self, move):
            return _AlwaysSolvedPuzzle(solved=True)
        def apply_moves(self, moves):
            return _AlwaysSolvedPuzzle(solved=True)

    chain_result = chain.solve(_AlwaysSolvedPuzzle(solved=False))
    assert chain_result.solved is True
    assert chain_result.moves == result.moves


def test_chain_aggregates_moves():
    """Moves from all stages are concatenated in the final result."""
    move_a = _make_move("U")
    move_b = _make_move("R")

    class _ProgressivePuzzle(_FakePuzzle):
        """Puzzle that becomes solved after applying move_b."""
        def __init__(self, moves_applied=0):
            super().__init__(solved=False)
            self._moves_applied = moves_applied

        @property
        def is_solved(self):
            return self._moves_applied >= 2

        def apply_move(self, move):
            return _ProgressivePuzzle(self._moves_applied + 1)

        def apply_moves(self, moves):
            result = self
            for m in moves:
                result = result.apply_move(m)
            return result

    unsolved_res = SolveResult(solved=False, moves=[move_a], solve_time_seconds=0.01, iterations=1)
    solved_res = SolveResult(solved=True, moves=[move_b], solve_time_seconds=0.01, iterations=1)

    solver1 = _MockSolver(unsolved_res)
    solver2 = _MockSolver(solved_res)

    stage1 = StageConfig(solver=solver1, move_budget=10, pass_partial=True)
    stage2 = StageConfig(solver=solver2, move_budget=10)
    chain = SolverChain(puzzle_type=_FakePuzzle, stages=[stage1, stage2])

    chain_result = chain.solve(_ProgressivePuzzle(0))
    assert move_a in chain_result.moves
    assert move_b in chain_result.moves


def test_chain_stops_on_solved():
    """If stage 1 solves the puzzle, stage 2 solver is never called."""
    class _ImmediatelySolvedPuzzle(_FakePuzzle):
        @property
        def is_solved(self):
            return True
        def apply_move(self, move):
            return _ImmediatelySolvedPuzzle()
        def apply_moves(self, moves):
            return _ImmediatelySolvedPuzzle()

    solved_res = _solved_result()
    dummy_res = _solved_result(moves=[_make_move("B")])

    solver1 = _MockSolver(solved_res)
    solver2 = _MockSolver(dummy_res)

    stage1 = StageConfig(solver=solver1, move_budget=10)
    stage2 = StageConfig(solver=solver2, move_budget=10)
    chain = SolverChain(puzzle_type=_FakePuzzle, stages=[stage1, stage2])

    chain.solve(_ImmediatelySolvedPuzzle())
    assert solver2.call_count == 0, "Stage 2 solver should not have been called"


def test_chain_metadata_has_stage_results():
    """Metadata must contain 'stage_results' list."""
    class _AlwaysSolvedPuzzle(_FakePuzzle):
        @property
        def is_solved(self):
            return True
        def apply_move(self, move):
            return _AlwaysSolvedPuzzle()
        def apply_moves(self, moves):
            return _AlwaysSolvedPuzzle()

    solver = _MockSolver(_solved_result())
    stage = StageConfig(solver=solver, move_budget=10)
    chain = SolverChain(puzzle_type=_FakePuzzle, stages=[stage])

    result = chain.solve(_AlwaysSolvedPuzzle())
    assert "stage_results" in result.metadata
    assert isinstance(result.metadata["stage_results"], list)
    assert len(result.metadata["stage_results"]) >= 1


def test_chain_metadata_stage_that_solved():
    """Metadata 'stage_that_solved' identifies the index of the solving stage."""
    class _AlwaysSolvedPuzzle(_FakePuzzle):
        @property
        def is_solved(self):
            return True
        def apply_move(self, move):
            return _AlwaysSolvedPuzzle()
        def apply_moves(self, moves):
            return _AlwaysSolvedPuzzle()

    solver = _MockSolver(_solved_result())
    stage = StageConfig(solver=solver, move_budget=10)
    chain = SolverChain(puzzle_type=_FakePuzzle, stages=[stage])

    result = chain.solve(_AlwaysSolvedPuzzle())
    assert result.metadata["stage_that_solved"] == 0


def test_chain_unsolved_result_when_no_stage_solves():
    """If no stage solves, the chain result has solved=False."""
    class _NeverSolvedPuzzle(_FakePuzzle):
        @property
        def is_solved(self):
            return False
        def apply_move(self, move):
            return _NeverSolvedPuzzle()
        def apply_moves(self, moves):
            return _NeverSolvedPuzzle()

    solver = _MockSolver(_unsolved_result())
    stage = StageConfig(solver=solver, move_budget=10)
    chain = SolverChain(puzzle_type=_FakePuzzle, stages=[stage])

    result = chain.solve(_NeverSolvedPuzzle())
    assert result.solved is False
    assert result.metadata["stage_that_solved"] is None


def test_chain_move_budget_truncation():
    """Moves beyond move_budget are ignored."""
    many_moves = [_make_move("U")] * 20

    class _AlwaysSolvedPuzzle(_FakePuzzle):
        @property
        def is_solved(self):
            return True
        def apply_move(self, move):
            return _AlwaysSolvedPuzzle()
        def apply_moves(self, moves):
            return _AlwaysSolvedPuzzle()

    solver = _MockSolver(SolveResult(solved=True, moves=many_moves,
                                     solve_time_seconds=0.01, iterations=1))
    stage = StageConfig(solver=solver, move_budget=5)
    chain = SolverChain(puzzle_type=_FakePuzzle, stages=[stage])

    result = chain.solve(_AlwaysSolvedPuzzle())
    assert len(result.moves) <= 5
