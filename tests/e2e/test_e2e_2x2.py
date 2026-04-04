"""End-to-end solve tests for the 2x2 Pocket Cube.

Each test:
  1. Creates a solved Cube2x2.
  2. Scrambles it by N moves.
  3. Runs a solver.
  4. Verifies the result two ways: result.solved and result.verify(original).
  5. Checks move count is within puzzle.move_limit() * 2 (generous bound for
     algorithm overhead).
  6. Checks solve_time_seconds > 0.

ML-model tests are skipped until trained weights are available.
"""
from __future__ import annotations

import pytest
import numpy as np

from rubiks_solve.core.cube_2x2 import Cube2x2
from rubiks_solve.solvers.genetic.solver import GeneticSolver, GeneticConfig
from rubiks_solve.solvers.mcts.solver import MCTSSolver, MCTSConfig

# ---------------------------------------------------------------------------
# Shared RNG — fixed seed for reproducibility
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Small configs that keep individual tests well under 30 s
# ---------------------------------------------------------------------------
_FAST_GENETIC = GeneticConfig(
    population_size=50,
    max_generations=100,
    max_chromosome_length=20,
    seed=42,
)

_FAST_MCTS = MCTSConfig(
    n_simulations=500,
    time_limit_seconds=5.0,
    seed=42,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _verify_solution(initial_puzzle: Cube2x2, result) -> None:
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

class TestE2E2x2Genetic:
    """Full solve pipeline: scramble → genetic solve → verify."""

    def test_solve_1_move_scramble(self):
        """Single move scramble should be trivially solved."""
        rng = np.random.default_rng(0)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(1, rng)
        solver = GeneticSolver(Cube2x2, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_3_move_scramble(self):
        """3-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(1)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube2x2, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_5_move_scramble(self):
        """5-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(2)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(5, rng)
        solver = GeneticSolver(Cube2x2, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_result_moves_replay_correctly(self):
        """result.verify() must return True independently of result.solved."""
        rng = np.random.default_rng(3)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube2x2, _FAST_GENETIC)
        result = solver.solve(scrambled)
        # Independently replay without trusting result.solved
        replayed = scrambled.apply_moves(result.moves)
        assert replayed.is_solved, (
            "Manual move replay did not reach solved state, "
            "but result.verify() should catch this too"
        )
        assert result.verify(scrambled), "result.verify() must agree with manual replay"

    def test_move_count_within_limit(self):
        """Solution length must not exceed puzzle move_limit()."""
        rng = np.random.default_rng(4)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube2x2, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert result.move_count <= Cube2x2.move_limit() * 2, (
            f"move_count {result.move_count} exceeds generous bound "
            f"{Cube2x2.move_limit() * 2}"
        )

    def test_solve_time_recorded(self):
        """solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(5)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube2x2, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert isinstance(result.solve_time_seconds, float)
        assert result.solve_time_seconds > 0


# ---------------------------------------------------------------------------
# MCTS solver tests
# ---------------------------------------------------------------------------

class TestE2E2x2MCTS:
    """MCTS full solve pipeline for 2x2."""

    def test_solve_3_move_scramble(self):
        """3-move scramble solved by MCTS."""
        rng = np.random.default_rng(10)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube2x2, _FAST_MCTS)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_time_recorded(self):
        """MCTS solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(11)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube2x2, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.solve_time_seconds > 0

    def test_result_verify_independently(self):
        """result.verify() must confirm the MCTS solution is correct."""
        rng = np.random.default_rng(12)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube2x2, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.verify(scrambled), "MCTS move replay did not yield solved state"


# ---------------------------------------------------------------------------
# ML model tests (skipped until trained weights are available)
# ---------------------------------------------------------------------------

class TestE2E2x2MLModels:
    """Placeholder tests for ML-based solvers.

    Enable each test by removing the skip decorator after training the
    corresponding model with, for example:
        rubiks-train --solver cnn --puzzle 2x2
    """

    @pytest.mark.skip(
        reason="Requires trained model weights. "
               "Run after: rubiks-train --solver cnn --puzzle 2x2"
    )
    def test_cnn_solve(self):
        from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig
        rng = np.random.default_rng(20)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = CNNSolver(Cube2x2, CNNConfig())
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_policy_solve(self):
        from rubiks_solve.solvers.policy.solver import PolicyNetworkSolver as PolicySolver
        rng = np.random.default_rng(21)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        from rubiks_solve.encoding import get_encoder
        _encoder = get_encoder("one_hot", Cube2x2)
        solver = PolicySolver(Cube2x2, _encoder)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_dqn_solve(self):
        from rubiks_solve.solvers.dqn.solver import DQNSolver
        rng = np.random.default_rng(22)
        solved = Cube2x2.solved_state()
        scrambled = solved.scramble(3, rng)
        from rubiks_solve.encoding import get_encoder
        _encoder = get_encoder("one_hot", Cube2x2)
        solver = DQNSolver(Cube2x2, _encoder)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)
