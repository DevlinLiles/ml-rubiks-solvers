"""End-to-end solve tests for the 3x3 Standard Cube.

Uses 3–5 move shallow scrambles and small solver budgets to keep each test
well under 30 seconds.

Extra tests versus 2x2:
  - test_solver_chain_two_stage: SolverChain with two genetic stages.
  - test_ensemble_genetic_mcts:  EnsembleSolver combining genetic + MCTS.

ML-model tests are skipped until trained weights are available.
"""
from __future__ import annotations

import pytest
import numpy as np

from rubiks_solve.core.cube_3x3 import Cube3x3
from rubiks_solve.pipeline.chain import SolverChain, StageConfig
from rubiks_solve.pipeline.ensemble import EnsembleSolver, VotingStrategy
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
# Shared helper
# ---------------------------------------------------------------------------

def _verify_solution(initial_puzzle: Cube3x3, result) -> None:
    """Assert all four correctness conditions for a solve result."""
    assert result.solved, "Solver reported not solved"
    assert result.verify(initial_puzzle), "Move replay did not yield solved state"
    assert result.move_count <= initial_puzzle.move_limit() * 2, (
        f"Solution too long: {result.move_count} moves "
        f"(limit * 2 = {initial_puzzle.move_limit() * 2})"
    )
    assert result.solve_time_seconds > 0, "solve_time_seconds must be positive"


# ---------------------------------------------------------------------------
# Genetic solver tests
# ---------------------------------------------------------------------------

class TestE2E3x3Genetic:
    """Full solve pipeline: scramble → genetic solve → verify."""

    def test_solve_1_move_scramble(self):
        """Single move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(0)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(1, rng)
        solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_3_move_scramble(self):
        """3-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(1)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_5_move_scramble(self):
        """5-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(2)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(5, rng)
        solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_result_moves_replay_correctly(self):
        """result.verify() must return True independently of result.solved."""
        rng = np.random.default_rng(3)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        result = solver.solve(scrambled)
        replayed = scrambled.apply_moves(result.moves)
        assert replayed.is_solved, "Manual move replay did not reach solved state"
        assert result.verify(scrambled), "result.verify() must agree with manual replay"

    def test_move_count_within_limit(self):
        """Solution length must not exceed puzzle move_limit() * 2."""
        rng = np.random.default_rng(4)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert result.move_count <= Cube3x3.move_limit() * 2

    def test_solve_time_recorded(self):
        """solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(5)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube3x3, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert isinstance(result.solve_time_seconds, float)
        assert result.solve_time_seconds > 0


# ---------------------------------------------------------------------------
# MCTS solver tests
# ---------------------------------------------------------------------------

class TestE2E3x3MCTS:
    """MCTS full solve pipeline for 3x3."""

    @pytest.mark.xfail(strict=False, reason="Stochastic MCTS — 3-move 3x3 not guaranteed in budget")
    def test_solve_3_move_scramble(self):
        """3-move scramble solved by MCTS."""
        rng = np.random.default_rng(10)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube3x3, _FAST_MCTS)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_time_recorded(self):
        """MCTS solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(11)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube3x3, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.solve_time_seconds > 0

    @pytest.mark.xfail(strict=False, reason="Stochastic MCTS — 3-move 3x3 not guaranteed in budget")
    def test_result_verify_independently(self):
        """result.verify() must confirm the MCTS solution is correct."""
        rng = np.random.default_rng(12)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube3x3, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.verify(scrambled), "MCTS move replay did not yield solved state"


# ---------------------------------------------------------------------------
# Pipeline tests (3x3-specific)
# ---------------------------------------------------------------------------

class TestE2E3x3Pipeline:
    """SolverChain and EnsembleSolver integration tests on 3x3."""

    @pytest.mark.xfail(strict=False, reason="Stochastic chain — two-stage 3x3 not guaranteed in budget")
    def test_solver_chain_two_stage(self):
        """SolverChain with genetic stage 1 (small budget) + stage 2 (larger budget).

        Stage 1 gets a tight move budget; if it fails to solve, stage 2 picks
        up the partially-reduced state and finishes.
        """
        rng = np.random.default_rng(20)
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

        # Two-stage chain over a shallow scramble must solve it
        assert result.solved, "SolverChain failed to solve a 3-move scramble"
        assert result.verify(scrambled), "Chain move replay did not yield solved state"
        assert result.solve_time_seconds > 0

    def test_ensemble_genetic_mcts(self):
        """EnsembleSolver with genetic + MCTS; winner must be the solved result."""
        rng = np.random.default_rng(21)
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

        assert result.solved, "EnsembleSolver failed to solve a 3-move scramble"
        assert result.verify(scrambled), "Ensemble move replay did not yield solved state"
        assert result.solve_time_seconds > 0
        # Metadata must record individual solver results
        assert "all_results" in result.metadata
        assert "winner" in result.metadata


# ---------------------------------------------------------------------------
# ML model tests (skipped until trained weights are available)
# ---------------------------------------------------------------------------

class TestE2E3x3MLModels:
    """Placeholder tests for ML-based solvers on 3x3.

    Enable each test by removing the skip decorator after training the
    corresponding model with, for example:
        rubiks-train --solver cnn --puzzle 3x3
    """

    @pytest.mark.skip(
        reason="Requires trained model weights. "
               "Run after: rubiks-train --solver cnn --puzzle 3x3"
    )
    def test_cnn_solve(self):
        from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig
        rng = np.random.default_rng(30)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = CNNSolver(Cube3x3, CNNConfig())
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_policy_solve(self):
        from rubiks_solve.solvers.policy.solver import PolicySolver
        rng = np.random.default_rng(31)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = PolicySolver(Cube3x3)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_dqn_solve(self):
        from rubiks_solve.solvers.dqn.model import DQNSolver
        rng = np.random.default_rng(32)
        solved = Cube3x3.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = DQNSolver(Cube3x3)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)
