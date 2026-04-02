"""Tests for the OneHotEncoder."""
from __future__ import annotations

import numpy as np
import pytest

from rubiks_solve.encoding.one_hot import OneHotEncoder


def _cube3x3_class():
    from rubiks_solve.core.cube_3x3 import Cube3x3
    return Cube3x3


def _cube2x2_class():
    from rubiks_solve.core.cube_2x2 import Cube2x2
    return Cube2x2


def _megaminx_class():
    from rubiks_solve.core.megaminx import Megaminx
    return Megaminx


@pytest.fixture
def encoder_3x3():
    return OneHotEncoder(_cube3x3_class())


@pytest.fixture
def encoder_megaminx():
    return OneHotEncoder(_megaminx_class())


@pytest.fixture
def solved_3x3():
    return _cube3x3_class().solved_state()


@pytest.fixture
def scrambled_3x3():
    rng = np.random.default_rng(42)
    return _cube3x3_class().solved_state().scramble(10, rng)


@pytest.fixture
def solved_megaminx():
    return _megaminx_class().solved_state()


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_output_shape_3x3(encoder_3x3):
    """3x3 one-hot output shape should be (324,) = 6*3*3*6."""
    assert encoder_3x3.output_shape == (324,)


def test_output_size_3x3(encoder_3x3):
    """output_size must equal the product of output_shape elements."""
    assert encoder_3x3.output_size == 324


def test_output_shape_megaminx(encoder_megaminx):
    """Megaminx one-hot output shape should be (1584,) = 12*11*12."""
    assert encoder_megaminx.output_shape == (1584,)


def test_output_size_megaminx(encoder_megaminx):
    """Megaminx output_size must be 1584."""
    assert encoder_megaminx.output_size == 1584


# ---------------------------------------------------------------------------
# Value correctness tests
# ---------------------------------------------------------------------------

def test_values_binary(encoder_3x3, solved_3x3):
    """All encoded values must be exactly 0.0 or 1.0."""
    encoded = encoder_3x3.encode(solved_3x3)
    unique = np.unique(encoded)
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_sum_per_cell_3x3(encoder_3x3, solved_3x3):
    """Each sticker's one-hot block must sum to exactly 1.0."""
    encoded = encoder_3x3.encode(solved_3x3)
    # 3x3: 6*3*3 = 54 cells, each one-hot over 6 colors
    n_cells = 6 * 3 * 3
    n_colors = 6
    blocks = encoded.reshape(n_cells, n_colors)
    sums = blocks.sum(axis=1)
    assert np.allclose(sums, 1.0), f"Not all cells sum to 1.0: {sums}"


def test_sum_per_cell_megaminx(encoder_megaminx, solved_megaminx):
    """Each Megaminx sticker's one-hot block sums to 1.0."""
    encoded = encoder_megaminx.encode(solved_megaminx)
    n_cells = 12 * 11
    n_colors = 12
    blocks = encoded.reshape(n_cells, n_colors)
    sums = blocks.sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_dtype(encoder_3x3, solved_3x3):
    """Encoded output must be float32."""
    encoded = encoder_3x3.encode(solved_3x3)
    assert encoded.dtype == np.float32


def test_encode_shape(encoder_3x3, solved_3x3):
    """encode() must return an array of shape output_shape."""
    encoded = encoder_3x3.encode(solved_3x3)
    assert encoded.shape == encoder_3x3.output_shape


# ---------------------------------------------------------------------------
# Batch encoding tests
# ---------------------------------------------------------------------------

def test_encode_batch_shape(encoder_3x3, solved_3x3, scrambled_3x3):
    """encode_batch of N puzzles returns (N, output_size)."""
    puzzles = [solved_3x3, scrambled_3x3, solved_3x3]
    batch = encoder_3x3.encode_batch(puzzles)
    assert batch.shape == (3, encoder_3x3.output_size)


def test_encode_batch_dtype(encoder_3x3, solved_3x3, scrambled_3x3):
    """encode_batch output must be float32."""
    batch = encoder_3x3.encode_batch([solved_3x3, scrambled_3x3])
    assert batch.dtype == np.float32


def test_encode_batch_matches_individual(encoder_3x3, solved_3x3, scrambled_3x3):
    """encode_batch rows must match individual encode() calls."""
    puzzles = [solved_3x3, scrambled_3x3]
    batch = encoder_3x3.encode_batch(puzzles)
    for i, puzzle in enumerate(puzzles):
        assert np.array_equal(batch[i], encoder_3x3.encode(puzzle))


# ---------------------------------------------------------------------------
# Differentiation tests
# ---------------------------------------------------------------------------

def test_solved_differs_from_scrambled(encoder_3x3, solved_3x3, scrambled_3x3):
    """Solved and scrambled states must produce different encodings."""
    enc_solved = encoder_3x3.encode(solved_3x3)
    enc_scrambled = encoder_3x3.encode(scrambled_3x3)
    assert not np.array_equal(enc_solved, enc_scrambled)


def test_same_state_same_encoding(encoder_3x3, solved_3x3):
    """Encoding the same state twice produces identical results."""
    enc1 = encoder_3x3.encode(solved_3x3)
    enc2 = encoder_3x3.encode(solved_3x3)
    assert np.array_equal(enc1, enc2)


# ---------------------------------------------------------------------------
# Different cube sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected_size", [
    (2, 6 * 2 * 2 * 6),   # 144
    (3, 6 * 3 * 3 * 6),   # 324
    (4, 6 * 4 * 4 * 6),   # 576
    (5, 6 * 5 * 5 * 6),   # 900
])
def test_output_size_by_cube_size(n, expected_size):
    """OneHotEncoder output_size scales correctly with cube size."""
    if n == 2:
        from rubiks_solve.core.cube_2x2 import Cube2x2 as cls
    elif n == 3:
        from rubiks_solve.core.cube_3x3 import Cube3x3 as cls
    elif n == 4:
        from rubiks_solve.core.cube_4x4 import Cube4x4 as cls
    elif n == 5:
        from rubiks_solve.core.cube_5x5 import Cube5x5 as cls
    encoder = OneHotEncoder(cls)
    assert encoder.output_size == expected_size
