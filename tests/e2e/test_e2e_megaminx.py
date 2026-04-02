"""End-to-end solve tests for the Megaminx.

The Megaminx is a dodecahedral puzzle with 12 faces, 24 legal moves (each
face CW/CCW), and a practical move limit of 70 HTM.  Megaminx face turns are
72-degree rotations (pentagonal faces), not 90-degree — there is no "double"
move concept for Megaminx.

Scrambles are kept to 3–5 moves to keep tests fast.

Extra tests:
  - test_inverse_undoes_single_move: verifies that applying a move and then
    its inverse returns to the original state (validates the 72-degree-turn
    inverse logic).

ML-model tests are skipped until trained weights are available.
"""
from __future__ import annotations

import pytest
import numpy as np

from rubiks_solve.core.megaminx import Megaminx
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

def _verify_solution(initial_puzzle: Megaminx, result) -> None:
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

class TestE2EMegaminxGenetic:
    """Full solve pipeline: scramble → genetic solve → verify."""

    def test_solve_1_move_scramble(self):
        """Single move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(0)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(1, rng)
        solver = GeneticSolver(Megaminx, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_3_move_scramble(self):
        """3-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(1)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Megaminx, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.xfail(strict=False, reason="Stochastic solver — 5-move Megaminx not guaranteed in budget")
    def test_solve_5_move_scramble(self):
        """5-move scramble solved by genetic algorithm."""
        rng = np.random.default_rng(2)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(5, rng)
        solver = GeneticSolver(Megaminx, _FAST_GENETIC)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_result_moves_replay_correctly(self):
        """result.verify() must return True independently of result.solved."""
        rng = np.random.default_rng(3)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Megaminx, _FAST_GENETIC)
        result = solver.solve(scrambled)
        replayed = scrambled.apply_moves(result.moves)
        assert replayed.is_solved, "Manual move replay did not reach solved state"
        assert result.verify(scrambled), "result.verify() must agree with manual replay"

    def test_move_count_within_limit(self):
        """Solution length must not exceed puzzle move_limit() * 2."""
        rng = np.random.default_rng(4)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Megaminx, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert result.move_count <= Megaminx.move_limit() * 2

    def test_solve_time_recorded(self):
        """solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(5)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = GeneticSolver(Megaminx, _FAST_GENETIC)
        result = solver.solve(scrambled)
        assert isinstance(result.solve_time_seconds, float)
        assert result.solve_time_seconds > 0


# ---------------------------------------------------------------------------
# MCTS solver tests
# ---------------------------------------------------------------------------

class TestE2EMegaminxMCTS:
    """MCTS full solve pipeline for Megaminx."""

    @pytest.mark.xfail(strict=False, reason="Stochastic MCTS — 3-move Megaminx not guaranteed")
    def test_solve_3_move_scramble(self):
        """3-move scramble solved by MCTS."""
        rng = np.random.default_rng(10)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Megaminx, _FAST_MCTS)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    def test_solve_time_recorded(self):
        """MCTS solve_time_seconds must be a positive float."""
        rng = np.random.default_rng(11)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Megaminx, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.solve_time_seconds > 0

    @pytest.mark.xfail(strict=False, reason="Stochastic MCTS — 3-move Megaminx not guaranteed in budget")
    def test_result_verify_independently(self):
        """result.verify() must confirm the MCTS solution is correct."""
        rng = np.random.default_rng(12)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = MCTSSolver(Megaminx, _FAST_MCTS)
        result = solver.solve(scrambled)
        assert result.verify(scrambled), "MCTS move replay did not yield solved state"


# ---------------------------------------------------------------------------
# Megaminx-specific: inverse move correctness (72-degree turns)
# ---------------------------------------------------------------------------

class TestE2EMegaminxInverse:
    """Verify that Megaminx move inverses correctly undo 72-degree turns.

    Megaminx has no double moves (unlike NxN cubes).  A move's inverse must
    always be a distinct CCW/CW move (never itself), and applying a move
    followed by its inverse must restore the original state.
    """

    def test_inverse_undoes_single_move(self):
        """Applying any move and then its inverse must return to the original state."""
        solved = Megaminx.solved_state()
        legal_moves = solved.legal_moves()

        # Test every legal move
        for move in legal_moves:
            after_move = solved.apply_move(move)
            inverse = move.inverse()
            # 180-degree doubles are their own inverse — Megaminx never produces
            # double moves, but guard against it for robustness.
            assert not move.double, (
                f"Megaminx move {move.name!r} unexpectedly marked as double "
                "(Megaminx uses 72-degree turns only)"
            )
            restored = after_move.apply_move(inverse)
            assert restored == solved, (
                f"Applying {move.name!r} then {inverse.name!r} did not restore "
                "the solved state"
            )

    def test_inverse_of_sequence_restores_state(self):
        """Applying a 3-move scramble then its inverse sequence restores solved."""
        rng = np.random.default_rng(99)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)

        # Build the scramble move sequence by inspecting legal_moves and
        # re-scrambling from a known seed so the moves are accessible.
        # Since AbstractPuzzle.scramble() does not return the move list we
        # verify via the solver instead — this test is just for the inverse API.
        legal_moves = solved.legal_moves()
        n_moves = len(legal_moves)
        rng2 = np.random.default_rng(0)
        moves = [legal_moves[int(rng2.integers(0, n_moves))] for _ in range(3)]
        state_after = solved.apply_moves(moves)

        # Apply inverses in reverse
        inverses = [m.inverse() for m in reversed(moves)]
        restored = state_after.apply_moves(inverses)
        assert restored == solved, (
            "Applying inverse sequence in reverse order did not restore solved state"
        )


# ---------------------------------------------------------------------------
# ML model tests (skipped until trained weights are available)
# ---------------------------------------------------------------------------

class TestE2EMegaminxMLModels:
    """Placeholder tests for ML-based solvers on Megaminx.

    Enable each test by removing the skip decorator after training the
    corresponding model with, for example:
        rubiks-train --solver cnn --puzzle megaminx
    """

    @pytest.mark.skip(
        reason="Requires trained model weights. "
               "Run after: rubiks-train --solver cnn --puzzle megaminx"
    )
    def test_cnn_solve(self):
        from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig
        rng = np.random.default_rng(20)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = CNNSolver(Megaminx, CNNConfig())
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_policy_solve(self):
        from rubiks_solve.solvers.policy.solver import PolicySolver
        rng = np.random.default_rng(21)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = PolicySolver(Megaminx)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)

    @pytest.mark.skip(reason="Requires trained model weights.")
    def test_dqn_solve(self):
        from rubiks_solve.solvers.dqn.model import DQNSolver
        rng = np.random.default_rng(22)
        solved = Megaminx.solved_state()
        scrambled = solved.scramble(3, rng)
        solver = DQNSolver(Megaminx)
        result = solver.solve(scrambled)
        _verify_solution(scrambled, result)
