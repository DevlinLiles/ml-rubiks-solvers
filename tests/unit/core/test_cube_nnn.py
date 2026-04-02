"""Tests for CubeNNN parameterized engine, covering 2x2 through 5x5."""
from __future__ import annotations

import numpy as np
import pytest


def _get_cube_class(n: int):
    """Return the cube class for the given n."""
    if n == 2:
        from rubiks_solve.core.cube_2x2 import Cube2x2
        return Cube2x2
    if n == 3:
        from rubiks_solve.core.cube_3x3 import Cube3x3
        return Cube3x3
    if n == 4:
        from rubiks_solve.core.cube_4x4 import Cube4x4
        return Cube4x4
    if n == 5:
        from rubiks_solve.core.cube_5x5 import Cube5x5
        return Cube5x5
    raise ValueError(f"Unsupported n={n}")


CUBE_SIZES = [2, 3, 4, 5]


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_solved_state_is_solved(n):
    """solved_state().is_solved must be True."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    assert cube.is_solved is True


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_initial_state_shape(n):
    """state.shape must be (6, n, n) for an NxN cube."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    assert cube.state.shape == (6, n, n)


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_initial_state_dtype(n):
    """state dtype must be uint8."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    assert cube.state.dtype == np.uint8


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_apply_move_returns_new_instance(n):
    """apply_move must return a different object (immutability)."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    move = cube.legal_moves()[0]
    result = cube.apply_move(move)
    assert result is not cube


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_apply_move_does_not_mutate(n):
    """Original state must remain unchanged after apply_move."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    original_state = cube.state.copy()
    move = cube.legal_moves()[0]
    cube.apply_move(move)
    assert np.array_equal(cube.state, original_state)


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_inverse_move_identity(n):
    """Applying a move then its inverse returns to the original state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    # Use a non-double move so inverse is different
    move = next(m for m in cube.legal_moves() if not m.double)
    after_move = cube.apply_move(move)
    after_inverse = after_move.apply_move(move.inverse())
    assert after_inverse == cube


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_double_move_identity(n):
    """Applying a double move twice returns to the original state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    double_moves = [m for m in cube.legal_moves() if m.double]
    if not double_moves:
        pytest.skip(f"No double moves for {n}x{n}")
    move = double_moves[0]
    after_once = cube.apply_move(move)
    after_twice = after_once.apply_move(move)
    assert after_twice == cube


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_all_legal_moves_return_valid_state(n):
    """Every legal move applied to the solved state produces a valid (6, n, n) state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    for move in cube.legal_moves():
        result = cube.apply_move(move)
        assert result.state.shape == (6, n, n)
        assert result.state.dtype == np.uint8


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_scramble_not_solved(n):
    """scramble(20) with a fixed seed produces a non-solved state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    rng = np.random.default_rng(42)
    scrambled = cube.scramble(20, rng)
    # With 20 moves from a fixed seed, extremely unlikely to be solved
    assert not scrambled.is_solved


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_scramble_returns_puzzle_instance(n):
    """scramble must return an AbstractPuzzle instance of the same type."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    rng = np.random.default_rng(0)
    result = cube.scramble(5, rng)
    assert isinstance(result, cls)


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_apply_moves_sequence_u_six_times(n):
    """Applying 4 U moves (if available) returns to original state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    # Find U CW move (non-double)
    u_moves = [m for m in cube.legal_moves() if m.face == "U" and m.direction == 1 and not m.double]
    if not u_moves:
        pytest.skip(f"No U CW move found for {n}x{n}")
    u_move = u_moves[0]
    # 4 CW quarter turns = identity
    result = cube.apply_moves([u_move] * 4)
    assert result == cube


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_equality(n):
    """Two solved_states are equal; solved != scrambled (with seed)."""
    cls = _get_cube_class(n)
    cube1 = cls.solved_state()
    cube2 = cls.solved_state()
    assert cube1 == cube2
    rng = np.random.default_rng(7)
    scrambled = cube1.scramble(10, rng)
    assert cube1 != scrambled


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_hash_consistency(n):
    """Equal puzzles must have equal hashes."""
    cls = _get_cube_class(n)
    cube1 = cls.solved_state()
    cube2 = cls.solved_state()
    assert hash(cube1) == hash(cube2)


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_copy_is_independent(n):
    """copy() returns a new object with no shared state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    copy = cube.copy()
    assert copy is not cube
    assert copy == cube
    # Apply a move to the original; copy should be unaffected
    move = cube.legal_moves()[0]
    _ = cube.apply_move(move)
    assert copy == cls.solved_state()


@pytest.mark.parametrize("n,expected_limit", [
    (2, 11),
    (3, 20),
    (4, 40),
    (5, 60),
])
def test_move_limit(n, expected_limit):
    """Each cube size has the correct move_limit()."""
    cls = _get_cube_class(n)
    assert cls.move_limit() == expected_limit


@pytest.mark.parametrize("n,expected_name", [
    (2, "2x2"),
    (3, "3x3"),
    (4, "4x4"),
    (5, "5x5"),
])
def test_puzzle_name(n, expected_name):
    """puzzle_name() returns the expected short name."""
    cls = _get_cube_class(n)
    assert cls.puzzle_name() == expected_name


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_solved_state_color_values(n):
    """Each face of the solved state has all stickers equal to the face index."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    for face_idx in range(6):
        face_stickers = cube.state[face_idx]
        assert np.all(face_stickers == face_idx), (
            f"Face {face_idx} has unexpected sticker values: {face_stickers}"
        )


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_legal_moves_returns_list(n):
    """legal_moves() must return a non-empty list."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    moves = cube.legal_moves()
    assert isinstance(moves, list)
    assert len(moves) > 0


@pytest.mark.parametrize("n", CUBE_SIZES)
def test_apply_moves_empty_sequence(n):
    """apply_moves([]) returns state equal to self."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    result = cube.apply_moves([])
    assert result == cube
