"""Tests for the puzzle state validator."""
from __future__ import annotations

import numpy as np
import pytest

from rubiks_solve.core.validator import is_valid_state, is_reachable


def _get_cube_class(n: int):
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


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_solved_state_is_valid(n):
    """is_valid_state returns True for a freshly solved cube."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    assert is_valid_state(cube) is True


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_scrambled_state_is_valid(n):
    """A scrambled cube is still physically valid (just not solved)."""
    cls = _get_cube_class(n)
    rng = np.random.default_rng(99)
    scrambled = cls.solved_state().scramble(15, rng)
    assert is_valid_state(scrambled) is True


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_invalid_state_detected(n):
    """Manually corrupting a sticker makes is_valid_state return False."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    # Build a corrupted copy by duplicating one color and removing another
    bad_state = cube.state.copy()
    # Set (0,0,0) to the color of face 1, making face-0 color appear one too few
    # and face-1 color appear one too many
    bad_state[0, 0, 0] = 1
    # Construct a minimal puzzle-like object carrying the bad state
    bad_puzzle = _BadPuzzle(bad_state)
    assert is_valid_state(bad_puzzle) is False


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_color_count_check(n):
    """Each color must appear exactly n^2 times in a valid state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    state = cube.state
    expected = n * n
    for color in range(6):
        count = int(np.sum(state == color))
        assert count == expected, (
            f"Color {color} appears {count} times, expected {expected} for {n}x{n}"
        )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_is_reachable_solved(n):
    """is_reachable returns True for a solved cube."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    assert is_reachable(cube) is True


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_is_reachable_scrambled(n):
    """is_reachable returns True for a legally scrambled cube."""
    cls = _get_cube_class(n)
    rng = np.random.default_rng(13)
    scrambled = cls.solved_state().scramble(10, rng)
    assert is_reachable(scrambled) is True


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_is_reachable_invalid_is_false(n):
    """is_reachable returns False for a corrupted (invalid) state."""
    cls = _get_cube_class(n)
    cube = cls.solved_state()
    bad_state = cube.state.copy()
    bad_state[0, 0, 0] = 1
    bad_puzzle = _BadPuzzle(bad_state)
    assert is_reachable(bad_puzzle) is False


# ---------------------------------------------------------------------------
# Minimal stub puzzle that lets us inject an arbitrary state array
# ---------------------------------------------------------------------------

class _BadPuzzle:
    """Minimal stub implementing only the state property for validator tests."""

    def __init__(self, state: np.ndarray) -> None:
        self._state = state

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def is_solved(self) -> bool:
        return False
