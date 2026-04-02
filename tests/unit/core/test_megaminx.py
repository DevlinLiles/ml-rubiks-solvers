"""Tests for the Megaminx puzzle implementation."""
from __future__ import annotations

import numpy as np
import pytest

from rubiks_solve.core.megaminx import Megaminx


@pytest.fixture
def solved():
    return Megaminx.solved_state()


def test_solved_state_is_solved(solved):
    """solved_state().is_solved must be True."""
    assert solved.is_solved is True


def test_state_shape(solved):
    """Megaminx state must have shape (12, 11)."""
    assert solved.state.shape == (12, 11)


def test_state_dtype(solved):
    """Megaminx state must be uint8."""
    assert solved.state.dtype == np.uint8


def test_apply_move_returns_new_instance(solved):
    """apply_move must return a different object."""
    move = solved.legal_moves()[0]
    result = solved.apply_move(move)
    assert result is not solved


def test_apply_move_does_not_mutate(solved):
    """Original state must not be modified by apply_move."""
    original_state = solved.state.copy()
    move = solved.legal_moves()[0]
    solved.apply_move(move)
    assert np.array_equal(solved.state, original_state)


def test_inverse_move_identity(solved):
    """Applying F+ then F- returns to the original state."""
    # Find the CW move for face "F"
    f_cw = next(m for m in solved.legal_moves() if m.face == "F" and m.direction == +1)
    f_ccw = f_cw.inverse()
    after_cw = solved.apply_move(f_cw)
    after_round_trip = after_cw.apply_move(f_ccw)
    assert after_round_trip == solved


def test_all_24_moves_exist(solved):
    """There must be exactly 24 legal Megaminx moves."""
    moves = solved.legal_moves()
    assert len(moves) == 24


def test_scramble_not_solved(solved):
    """scramble(20) with a fixed seed produces a non-solved state."""
    rng = np.random.default_rng(42)
    scrambled = solved.scramble(20, rng)
    assert not scrambled.is_solved


def test_scramble_returns_megaminx(solved):
    """scramble() returns a Megaminx instance."""
    rng = np.random.default_rng(1)
    result = solved.scramble(5, rng)
    assert isinstance(result, Megaminx)


def test_puzzle_name():
    """puzzle_name() must return 'megaminx'."""
    assert Megaminx.puzzle_name() == "megaminx"


def test_move_limit():
    """move_limit() must return 70."""
    assert Megaminx.move_limit() == 70


def test_move_names_have_plus_minus_suffix(solved):
    """Megaminx move names end with '+' (CW) or '-' (CCW)."""
    for move in solved.legal_moves():
        assert move.name.endswith("+") or move.name.endswith("-"), (
            f"Move name {move.name!r} does not follow '+'/'-' convention"
        )


def test_copy_is_independent(solved):
    """copy() creates an independent deep copy."""
    copy = solved.copy()
    assert copy is not solved
    assert copy == solved
    # Applying a move to the copy must not affect the original
    move = solved.legal_moves()[0]
    copy.apply_move(move)
    assert solved.is_solved


def test_equality_solved_states():
    """Two independently created solved states compare as equal."""
    a = Megaminx.solved_state()
    b = Megaminx.solved_state()
    assert a == b


def test_equality_solved_vs_scrambled():
    """Solved state != scrambled state."""
    solved = Megaminx.solved_state()
    rng = np.random.default_rng(3)
    scrambled = solved.scramble(10, rng)
    assert solved != scrambled


def test_hash_equal_states():
    """Equal states must share the same hash."""
    a = Megaminx.solved_state()
    b = Megaminx.solved_state()
    assert hash(a) == hash(b)


def test_solved_state_sticker_values(solved):
    """In the solved state, all stickers on face i equal i."""
    for i in range(12):
        assert np.all(solved.state[i] == i), (
            f"Face {i} has unexpected sticker values"
        )


def test_each_face_move_cw_ccw(solved):
    """Applying CW then CCW for every face returns to the solved state."""
    for move in solved.legal_moves():
        if move.direction != +1:
            continue
        after_cw = solved.apply_move(move)
        after_ccw = after_cw.apply_move(move.inverse())
        assert after_ccw == solved, (
            f"Round-trip failed for face {move.face}"
        )


def test_invalid_state_shape_raises():
    """Megaminx rejects a state array with wrong shape."""
    bad = np.zeros((12, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        Megaminx(bad)


def test_invalid_state_dtype_raises():
    """Megaminx rejects a state array with wrong dtype."""
    bad = np.zeros((12, 11), dtype=np.int32)
    with pytest.raises(ValueError, match="dtype"):
        Megaminx(bad)
