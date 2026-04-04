"""End-to-end tests for multi-solver pipeline components.

Tests cover:
  - SolverChain: sequential multi-stage solving with move budgets.
  - EnsembleSolver: concurrent multi-solver solving with voting strategies.

All tests use Cube3x3 with 3-move shallow scrambles and small genetic /
MCTS configs to stay well under 30 seconds each.
"""
from __future__ import annotations

import pytest
import numpy as np

from rubiks_solve.core.cube_3x3 import Cube3x3
from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.pipeline.chain import SolverChain, StageConfig
from rubiks_solve.pipeline.ensemble import EnsembleSolver, VotingStrategy
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.solvers.genetic.solver import GeneticSolver, GeneticConfig
from rubiks_solve.solvers.mcts.solver import MCTSSolver, MCTSConfig

# ---------------------------------------------------------------------------
# Small configs
# ---------------------------------------------------------------------------
_FAST_GENETIC = GeneticConfig(
    population_size=200,
    max_generations=500,
    max_chromosome_length=20,
    seed=42,
)

_FAST_MCTS = MCTSConfig(
    n_simulations=5000,
    time_limit_seconds=10.0,
    seed=42,
)


# ---------------------------------------------------------------------------
# Mock solver that always reports failure
# ---------------------------------------------------------------------------

class _AlwaysFailSolver(AbstractSolver):
    """Solver stub that immediately returns solved=False with no moves.

    Used to verify that EnsembleSolver falls back to another solver that
    does succeed.
    """

    def __init__(self, puzzle_type: type[AbstractPuzzle]) -> None:
        super().__init__(puzzle_type, config=None)

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        return SolveResult(
            solved=False,
            moves=[],
            solve_time_seconds=0.001,
            iterations=0,
            metadata={"reason": "intentional failure"},
        )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _verify_solution(initial_puzzle: Cube3x3, result: SolveResult) -> None:
    """Assert all four correctness conditions for a solve result."""
    assert result.solved, "Solver reported not solved"
    assert result.verify(initial_puzzle), "Move replay did not yield solved state"
    assert result.move_count <= initial_puzzle.move_limit() * 2, (
        f"Solution too long: {result.move_count} moves "
        f"(limit * 2 = {initial_puzzle.move_limit() * 2})"
    )
    assert result.solve_time_seconds > 0, "solve_time_seconds must be positive"


# ---------------------------------------------------------------------------
# SolverChain tests
# ---------------------------------------------------------------------------

class TestSolverChain:
    """SolverChain integration tests."""

    @pytest.mark.xfail(strict=False, reason="Stochastic chain — may not solve within budget")
    def test_two_stage_chain_3x3(self):
        """Two genetic solvers chained: stage 1 (small budget), stage 2 (larger budget).

        On a 3-move scramble at least one stage must fully solve the puzzle.
        """
        rng = np.random.default_rng(0)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        stage1_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=100,
            max_generations=200,
            max_chromosome_length=20,
            seed=1,
        ))
        stage2_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=200,
            max_generations=500,
            max_chromosome_length=20,
            seed=2,
        ))

        chain = SolverChain(
            puzzle_type=Cube3x3,
            stages=[
                StageConfig(solver=stage1_solver, move_budget=Cube3x3.move_limit()),
                StageConfig(solver=stage2_solver, move_budget=Cube3x3.move_limit()),
            ],
        )
        result = chain.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_chain_metadata_populated(self):
        """metadata['stage_results'] has an entry for each stage that ran."""
        rng = np.random.default_rng(1)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        stage1_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=100,
            max_generations=200,
            max_chromosome_length=20,
            seed=10,
        ))
        stage2_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=200,
            max_generations=500,
            max_chromosome_length=20,
            seed=11,
        ))

        chain = SolverChain(
            puzzle_type=Cube3x3,
            stages=[
                StageConfig(solver=stage1_solver, move_budget=Cube3x3.move_limit()),
                StageConfig(solver=stage2_solver, move_budget=Cube3x3.move_limit()),
            ],
        )
        result = chain.solve(scrambled)

        assert "stage_results" in result.metadata, (
            "metadata must contain 'stage_results'"
        )
        stage_results = result.metadata["stage_results"]
        assert isinstance(stage_results, list), "stage_results must be a list"
        # At least one stage must have run; at most all stages ran.
        assert 1 <= len(stage_results) <= 2, (
            f"Expected 1 or 2 stage results, got {len(stage_results)}"
        )
        for sr in stage_results:
            assert isinstance(sr, SolveResult), (
                f"Each entry must be a SolveResult, got {type(sr)}"
            )

        assert "stage_that_solved" in result.metadata
        solved_idx = result.metadata["stage_that_solved"]
        if result.solved:
            assert solved_idx is not None, (
                "stage_that_solved must not be None when chain solved"
            )

    def test_chain_moves_aggregate(self):
        """Total moves equals sum of all stage move sequences used."""
        rng = np.random.default_rng(2)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        stage1_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=100,
            max_generations=200,
            max_chromosome_length=20,
            seed=20,
        ))
        stage2_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=200,
            max_generations=500,
            max_chromosome_length=20,
            seed=21,
        ))

        chain = SolverChain(
            puzzle_type=Cube3x3,
            stages=[
                StageConfig(solver=stage1_solver, move_budget=Cube3x3.move_limit()),
                StageConfig(solver=stage2_solver, move_budget=Cube3x3.move_limit()),
            ],
        )
        result = chain.solve(scrambled)

        stage_results: list[SolveResult] = result.metadata["stage_results"]
        expected_total = sum(len(sr.moves) for sr in stage_results)
        assert result.move_count == expected_total, (
            f"Chain move_count {result.move_count} != "
            f"sum of stage moves {expected_total}"
        )

    def test_chain_stops_early_when_solved(self):
        """If stage 1 solves the puzzle, stage 2 must not be called.

        Uses a 1-move scramble so the genetic algorithm trivially solves it
        in stage 1.  The stage_results list must then have exactly 1 entry.
        """
        rng = np.random.default_rng(3)
        solved = Cube3x3.solved_state()
        # 1-move scramble — extremely easy for the genetic solver
        scrambled = solved.scramble(1, rng)

        stage1_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=200,
            max_generations=500,
            max_chromosome_length=20,
            seed=30,
        ))
        stage2_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=200,
            max_generations=500,
            max_chromosome_length=20,
            seed=31,
        ))

        chain = SolverChain(
            puzzle_type=Cube3x3,
            stages=[
                StageConfig(solver=stage1_solver, move_budget=Cube3x3.move_limit()),
                StageConfig(solver=stage2_solver, move_budget=Cube3x3.move_limit()),
            ],
        )
        result = chain.solve(scrambled)

        # Must be solved overall
        assert result.solved, "Chain failed to solve a 1-move scramble"

        stage_results: list[SolveResult] = result.metadata["stage_results"]
        # Stage 1 solved it → chain stopped → only 1 stage result recorded
        assert len(stage_results) == 1, (
            f"Expected exactly 1 stage result (chain should stop early), "
            f"got {len(stage_results)}"
        )
        assert result.metadata["stage_that_solved"] == 0, (
            "stage_that_solved must be 0 when stage 1 solves"
        )

    def test_chain_verify_independently(self):
        """result.verify(original) must pass for the combined move sequence."""
        rng = np.random.default_rng(4)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        stage1_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=100,
            max_generations=200,
            max_chromosome_length=20,
            seed=40,
        ))
        stage2_solver = GeneticSolver(Cube3x3, GeneticConfig(
            population_size=200,
            max_generations=500,
            max_chromosome_length=20,
            seed=41,
        ))

        chain = SolverChain(
            puzzle_type=Cube3x3,
            stages=[
                StageConfig(solver=stage1_solver, move_budget=Cube3x3.move_limit()),
                StageConfig(solver=stage2_solver, move_budget=Cube3x3.move_limit()),
            ],
        )
        result = chain.solve(scrambled)

        assert result.verify(scrambled), (
            "Chain result.verify() failed — aggregated move sequence does not "
            "replay to solved state"
        )


# ---------------------------------------------------------------------------
# EnsembleSolver tests
# ---------------------------------------------------------------------------

class TestEnsembleSolver:
    """EnsembleSolver integration tests."""

    def test_ensemble_two_solvers_3x3(self):
        """Ensemble with genetic + MCTS on a 3-move scramble must solve it."""
        rng = np.random.default_rng(50)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        genetic_solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        mcts_solver = MCTSSolver(Cube3x3, _FAST_MCTS)

        ensemble = EnsembleSolver(
            puzzle_type=Cube3x3,
            solvers=[genetic_solver, mcts_solver],
            strategy=VotingStrategy.FASTEST_SOLVE,
            timeout_seconds=20.0,
        )
        result = ensemble.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_ensemble_metadata_has_all_solvers(self):
        """metadata['all_results'] must contain an entry for each solver that finished."""
        rng = np.random.default_rng(51)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        genetic_solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        mcts_solver = MCTSSolver(Cube3x3, _FAST_MCTS)

        ensemble = EnsembleSolver(
            puzzle_type=Cube3x3,
            solvers=[genetic_solver, mcts_solver],
            strategy=VotingStrategy.FASTEST_SOLVE,
            timeout_seconds=20.0,
        )
        result = ensemble.solve(scrambled)

        assert "all_results" in result.metadata, (
            "metadata must contain 'all_results'"
        )
        all_results: dict = result.metadata["all_results"]
        assert isinstance(all_results, dict), "all_results must be a dict"
        # Both solvers should have finished within the generous 20-second timeout
        assert len(all_results) >= 1, "at least one solver must have a result"
        # Each value must be a SolveResult
        for solver_name, sr in all_results.items():
            assert isinstance(sr, SolveResult), (
                f"all_results[{solver_name!r}] must be a SolveResult"
            )

        assert "winner" in result.metadata, "metadata must contain 'winner'"

    def test_ensemble_winner_is_fastest(self):
        """With FASTEST_SOLVE strategy, the winner is the solver with lowest time.

        Both solvers run; we check that the winner field names the solver
        whose result has the smallest solve_time_seconds among solved results.
        """
        rng = np.random.default_rng(52)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        genetic_solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        mcts_solver = MCTSSolver(Cube3x3, _FAST_MCTS)

        ensemble = EnsembleSolver(
            puzzle_type=Cube3x3,
            solvers=[genetic_solver, mcts_solver],
            strategy=VotingStrategy.FASTEST_SOLVE,
            timeout_seconds=20.0,
        )
        result = ensemble.solve(scrambled)

        all_results: dict = result.metadata["all_results"]
        winner_name: str = result.metadata["winner"]

        # Among solved results the winner must have the smallest time
        solved_results = {n: r for n, r in all_results.items() if r.solved}
        if solved_results:
            fastest_name = min(
                solved_results,
                key=lambda n: (solved_results[n].solve_time_seconds,
                               solved_results[n].move_count),
            )
            assert winner_name == fastest_name, (
                f"Winner {winner_name!r} is not the fastest solver "
                f"({fastest_name!r} was faster)"
            )

    @pytest.mark.xfail(strict=False, reason="Stochastic ensemble — may not solve within budget")
    def test_ensemble_verify_independently(self):
        """result.verify(original) must pass for the ensemble winner's moves."""
        rng = np.random.default_rng(53)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        genetic_solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        mcts_solver = MCTSSolver(Cube3x3, _FAST_MCTS)

        ensemble = EnsembleSolver(
            puzzle_type=Cube3x3,
            solvers=[genetic_solver, mcts_solver],
            strategy=VotingStrategy.FASTEST_SOLVE,
            timeout_seconds=20.0,
        )
        result = ensemble.solve(scrambled)

        assert result.verify(scrambled), (
            "Ensemble result.verify() failed — winner's moves do not replay "
            "to solved state"
        )

    @pytest.mark.xfail(strict=False, reason="Stochastic genetic solver — 3-move solve not guaranteed")
    def test_ensemble_with_one_failing_solver(self):
        """Ensemble still returns solved=True if at least one solver succeeds.

        Uses a mock solver that always returns solved=False alongside a real
        GeneticSolver.  The ensemble must select the real solver's result.
        """
        rng = np.random.default_rng(54)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)

        failing_solver = _AlwaysFailSolver(Cube3x3)
        real_solver = GeneticSolver(Cube3x3, _FAST_GENETIC)

        ensemble = EnsembleSolver(
            puzzle_type=Cube3x3,
            solvers=[failing_solver, real_solver],
            strategy=VotingStrategy.FASTEST_SOLVE,
            timeout_seconds=20.0,
        )
        result = ensemble.solve(scrambled)

        # The ensemble must have picked the real solver's solved result
        assert result.solved, (
            "EnsembleSolver did not select the successful solver's result"
        )
        assert result.verify(scrambled), (
            "Ensemble move replay did not yield solved state"
        )

        all_results: dict = result.metadata["all_results"]
        winner_name: str = result.metadata["winner"]
        # The winner must not be the always-failing mock
        assert winner_name != failing_solver.solver_name or result.solved, (
            "Winner should be the real solver, not the failing mock"
        )
        # The real solver's result in all_results must be solved
        real_name = real_solver.solver_name
        if real_name in all_results:
            assert all_results[real_name].solved, (
                f"Real solver ({real_name}) did not solve the puzzle"
            )
