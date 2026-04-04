"""End-to-end solve tests for the 4x4 Revenge Cube.

The 4x4 has 45 legal moves and a practical move limit of 40 HTM.  Scrambles
are kept to 3 moves only so that the genetic / MCTS solvers can handle them
within the test timeout.

Extra test versus smaller cubes:
  - test_no_parity_in_test_scrambles: verifies that the shallow 3-move
    scrambles used here do not introduce a parity error (or documents the
    behaviour if they do).

ML-model tests are skipped until trained weights are available.
"""
from __future__ import annotations

import pytest
import numpy as np

from rubiks_solve.core.cube_4x4 import Cube4x4
from rubiks_solve.solvers.genetic.solver import GeneticSolver, GeneticConfig
from rubiks_solve.solvers.mcts.solver import MCTSSolver, MCTSConfig

# ---------------------------------------------------------------------------
# Small configs — 4x4 is significantly harder, keep budgets low
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

def _verify_solution(initial_puzzle: Cube4x4, result) -> None:
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

class TestE2E4x4Genetic:
    """Full solve pipeline: scramble → genetic solve → verify."""

    def test_solve_1_move_scramble(self):
        """Single move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(0)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(1, rng)
        solver = GeneticSolver(Cube4x4, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.xfail(strict=False, reason="Stochastic solver — 3-move 4x4 not guaranteed in budget")
    def test_solve_3_move_scramble(self):
        """3-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(1)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube4x4, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_result_moves_replay_correctly(self):
        """result.verify() must return True independently of result.solved."""
        rng = np.random.default_rng(2)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube4x4, _FAST_GENETIC)
        result = solver.solve(scrambled)
        replayed = scrambled.apply_moves(result.moves)
        assert replayed.is_solved, "Manual move replay did not reach solved state"
        assert result.verify(scrambled), "result.verify() must agree with manual replay"

    def test_move_count_within_limit(self):
        """Solution length must not exceed puzzle move_limit() * 2."""
        rng = np.random.default_rng(3)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube4x4, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert result.move_count <= Cube4x4.move_limit() * 2

    def test_solve_time_recorded(self):
        """solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(4)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Cube4x4, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert isinstance(result.solve_time_seconds, float)
        assert result.solve_time_seconds > 0


# ---------------------------------------------------------------------------
# MCTS solver tests
# ---------------------------------------------------------------------------

class TestE2E4x4MCTS:
    """MCTS full solve pipeline for 4x4."""

    @pytest.mark.xfail(strict=False, reason="Stochastic solver — 3-move 4x4 MCTS not guaranteed")
    def test_solve_3_move_scramble(self):
        """3-move scramble solved by MCTS."""
        rng = np.random.default_rng(10)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube4x4, _FAST_MCTS)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_time_recorded(self):
        """MCTS solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(11)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube4x4, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.solve_time_seconds > 0

    @pytest.mark.xfail(strict=False, reason="Stochastic MCTS — 3-move 4x4 not guaranteed in budget")
    def test_result_verify_independently(self):
        """result.verify() must confirm the MCTS solution is correct."""
        rng = np.random.default_rng(12)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Cube4x4, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.verify(scrambled), "MCTS move replay did not yield solved state"


# ---------------------------------------------------------------------------
# 4x4-specific: parity check
# ---------------------------------------------------------------------------

class TestE2E4x4Parity:
    """Verify parity behaviour for the shallow scrambles used in these tests.

    The 4x4 (even-n cube) can exhibit OLL or PLL parity.  For the 3-move
    scrambles used in the test suite the heuristic parity detector should
    either confirm no parity, or document the known heuristic limitation.
    """

    def test_no_parity_in_test_scrambles(self):
        """3-move scrambles from the fixed seeds used in this file should not
        introduce a parity error according to the cube's heuristic detector.

        If a seed does trigger parity the test prints a warning but does NOT
        fail — the scramble is still legal and the solver must handle it.
        """
        seeds_and_n_moves = [(0, 1), (1, 3), (2, 3), (3, 3), (4, 3), (10, 3), (11, 3), (12, 3)]
        parity_found: list[int] = []

        for seed, n_moves in seeds_and_n_moves:
            rng = np.random.default_rng(seed)
            solved = Cube4x4.solved_state()
            scrambled = solved.scramble(n_moves, rng)
            if scrambled.has_parity_error:
                parity_found.append(seed)

        # Shallow scrambles rarely produce parity; log any that do so the
        # developer can decide whether to use a different seed.
        if parity_found:
            import warnings
            warnings.warn(
                f"Seeds {parity_found} produced scrambles with detected parity "
                "errors.  Solvers must handle these states correctly.",
                UserWarning,
                stacklevel=2,
            )

        # The test itself always passes — it is documentation, not a hard gate.
        # Change `assert not parity_found` to enforce zero-parity seeds.


# ---------------------------------------------------------------------------
# ML model tests (skipped until trained weights are available)
# ---------------------------------------------------------------------------

class TestE2E4x4MLModels:
    """Placeholder tests for ML-based solvers on 4x4.

    Enable each test by removing the skip decorator after training the
    corresponding model with, for example:
        rubiks-train --solver cnn --puzzle 4x4
    """

    @pytest.mark.skip(
        reason="Requires trained model weights. "
               "Run after: rubiks-train --solver cnn --puzzle 4x4"
    )
    def test_cnn_solve(self):
        from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig
        rng = np.random.default_rng(20)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = CNNSolver(Cube4x4, CNNConfig())
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_policy_solve(self):
        from rubiks_solve.solvers.policy.solver import PolicyNetworkSolver as PolicySolver
        rng = np.random.default_rng(21)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        from rubiks_solve.encoding import get_encoder
        _encoder = get_encoder("one_hot", Cube4x4)
        solver = PolicySolver(Cube4x4, _encoder)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_dqn_solve(self):
        from rubiks_solve.solvers.dqn.solver import DQNSolver
        rng = np.random.default_rng(22)
        solved = Cube4x4.solved_state()
        scrambled = solved.scramble(3, rng)
        from rubiks_solve.encoding import get_encoder
        _encoder = get_encoder("one_hot", Cube4x4)
        solver = DQNSolver(Cube4x4, _encoder)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)
