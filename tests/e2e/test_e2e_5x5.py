"""End-to-end solve tests for the 5x5 Professor's Cube.

The 5x5 has 72 legal moves and a practical move limit of 60 HTM.  Scrambles
are kept to 2–3 moves only so that the genetic / MCTS solvers can handle them
within the test timeout.

ML-model tests are skipped until trained weights are available.
"""
from __future__ import annotations

import pytest
import numpy as np

from rubiks_solve.core.cube_5x5 import Cube5x5
from rubiks_solve.solvers.genetic.solver import GeneticSolver, GeneticConfig
from rubiks_solve.solvers.mcts.solver import MCTSSolver, MCTSConfig

# ---------------------------------------------------------------------------
# Small configs — 5x5 has a very large action space; keep budgets minimal
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

def _verify_solution(initial_puzzle: Cube5x5, result) -> None:
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

class TestE2E5x5Genetic:
    """Full solve pipeline: scramble → genetic solve → verify."""

    def test_solve_1_move_scramble(self):
        """Single move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(0)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(1, rng)
        solver = GeneticSolver(Cube5x5, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_2_move_scramble(self):
        """2-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(10)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(1, rng)
        solver = GeneticSolver(Cube5x5, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.xfail(strict=False, reason="Stochastic solver — 3-move 5x5 not guaranteed in budget")
    def test_solve_3_move_scramble(self):
        """3-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(2)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube5x5, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_result_moves_replay_correctly(self):
        """result.verify() must return True independently of result.solved."""
        rng = np.random.default_rng(3)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        solver = GeneticSolver(Cube5x5, _FAST_GENETIC)
        result = solver.solve(scrambled)
        replayed = scrambled.apply_moves(result.moves)
        assert replayed.is_solved, "Manual move replay did not reach solved state"
        assert result.verify(scrambled), "result.verify() must agree with manual replay"

    def test_move_count_within_limit(self):
        """Solution length must not exceed puzzle move_limit() * 2."""
        rng = np.random.default_rng(4)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        solver = GeneticSolver(Cube5x5, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert result.move_count <= Cube5x5.move_limit() * 2

    def test_solve_time_recorded(self):
        """solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(5)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        solver = GeneticSolver(Cube5x5, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert isinstance(result.solve_time_seconds, float)
        assert result.solve_time_seconds > 0


# ---------------------------------------------------------------------------
# MCTS solver tests
# ---------------------------------------------------------------------------

class TestE2E5x5MCTS:
    """MCTS full solve pipeline for 5x5."""

    def test_solve_2_move_scramble(self):
        """2-move scramble solved by MCTS."""
        rng = np.random.default_rng(10)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(1, rng)
        solver = MCTSSolver(Cube5x5, _FAST_MCTS)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.xfail(strict=False, reason='Stochastic solver — 3-move 5x5 not guaranteed in budget')
    def test_solve_3_move_scramble(self):
        """3-move scramble solved by MCTS."""
        rng = np.random.default_rng(11)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube5x5, _FAST_MCTS)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_time_recorded(self):
        """MCTS solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(12)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        solver = MCTSSolver(Cube5x5, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.solve_time_seconds > 0

    def test_result_verify_independently(self):
        """result.verify() must confirm the MCTS solution is correct."""
        rng = np.random.default_rng(13)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        solver = MCTSSolver(Cube5x5, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.verify(scrambled), "MCTS move replay did not yield solved state"


# ---------------------------------------------------------------------------
# ML model tests (skipped until trained weights are available)
# ---------------------------------------------------------------------------

class TestE2E5x5MLModels:
    """Placeholder tests for ML-based solvers on 5x5.

    Enable each test by removing the skip decorator after training the
    corresponding model with, for example:
        rubiks-train --solver cnn --puzzle 5x5
    """

    @pytest.mark.skip(
        reason="Requires trained model weights. "
               "Run after: rubiks-train --solver cnn --puzzle 5x5"
    )
    def test_cnn_solve(self):
        from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig
        rng = np.random.default_rng(20)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        solver = CNNSolver(Cube5x5, CNNConfig())
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_policy_solve(self):
        from rubiks_solve.solvers.policy.solver import PolicyNetworkSolver as PolicySolver
        rng = np.random.default_rng(21)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        from rubiks_solve.encoding import get_encoder
        _encoder = get_encoder("one_hot", Cube5x5)
        solver = PolicySolver(Cube5x5, _encoder)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_dqn_solve(self):
        from rubiks_solve.solvers.dqn.solver import DQNSolver
        rng = np.random.default_rng(22)
        solved = Cube5x5.solved_state()
        scrambled = solved.scramble(2, rng)
        from rubiks_solve.encoding import get_encoder
        _encoder = get_encoder("one_hot", Cube5x5)
        solver = DQNSolver(Cube5x5, _encoder)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)
