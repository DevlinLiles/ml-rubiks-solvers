"""Tests for the CubieEncoder."""
from __future__ import annotations

import numpy as np
import pytest

from rubiks_solve.encoding.cubie import CubieEncoder


def _cube3x3_class():
    from rubiks_solve.core.cube_3x3 import Cube3x3
    return Cube3x3


def _cube2x2_class():
    from rubiks_solve.core.cube_2x2 import Cube2x2
    return Cube2x2


@pytest.fixture
def encoder_3x3():
    return CubieEncoder(_cube3x3_class())


@pytest.fixture
def encoder_2x2():
    return CubieEncoder(_cube2x2_class())


@pytest.fixture
def solved_3x3():
    return _cube3x3_class().solved_state()


@pytest.fixture
def solved_2x2():
    return _cube2x2_class().solved_state()


@pytest.fixture
def scrambled_3x3():
    rng = np.random.default_rng(42)
    return _cube3x3_class().solved_state().scramble(10, rng)


# ---------------------------------------------------------------------------
# Shape and size tests
# ---------------------------------------------------------------------------

def test_output_shape_3x3(encoder_3x3):
    """3x3 cubie encoder output shape must be (256,)."""
    # 8 corners: 8*8 + 8*3 = 88; 12 edges: 12*12 + 12*2 = 168; total 256
    assert encoder_3x3.output_shape == (256,)


def test_output_size_3x3(encoder_3x3):
    """3x3 cubie encoder output_size must be 256."""
    assert encoder_3x3.output_size == 256


def test_output_shape_2x2(encoder_2x2):
    """2x2 cubie encoder output shape must be (88,) — corners only."""
    # 8 corners: 8*8 + 8*3 = 88
    assert encoder_2x2.output_shape == (88,)


def test_output_size_2x2(encoder_2x2):
    """2x2 cubie encoder output_size must be 88."""
    assert encoder_2x2.output_size == 88


# ---------------------------------------------------------------------------
# Value correctness tests
# ---------------------------------------------------------------------------

def test_values_binary_3x3(encoder_3x3, solved_3x3):
    """All encoded values for 3x3 must be 0.0 or 1.0."""
    encoded = encoder_3x3.encode(solved_3x3)
    unique = set(np.unique(encoded).tolist())
    assert unique.issubset({0.0, 1.0})


def test_values_binary_2x2(encoder_2x2, solved_2x2):
    """All encoded values for 2x2 must be 0.0 or 1.0."""
    encoded = encoder_2x2.encode(solved_2x2)
    unique = set(np.unique(encoded).tolist())
    assert unique.issubset({0.0, 1.0})


def test_dtype_3x3(encoder_3x3, solved_3x3):
    """Encoded output must be float32."""
    encoded = encoder_3x3.encode(solved_3x3)
    assert encoded.dtype == np.float32


def test_dtype_2x2(encoder_2x2, solved_2x2):
    """Encoded output for 2x2 must be float32."""
    encoded = encoder_2x2.encode(solved_2x2)
    assert encoded.dtype == np.float32


def test_encode_shape_3x3(encoder_3x3, solved_3x3):
    """encode() must return array matching output_shape."""
    encoded = encoder_3x3.encode(solved_3x3)
    assert encoded.shape == encoder_3x3.output_shape


def test_encode_shape_2x2(encoder_2x2, solved_2x2):
    """encode() for 2x2 must return array matching output_shape."""
    encoded = encoder_2x2.encode(solved_2x2)
    assert encoded.shape == encoder_2x2.output_shape


# ---------------------------------------------------------------------------
# Solved state structure test
# ---------------------------------------------------------------------------

def test_solved_encoding_roundtrip_3x3(encoder_3x3, solved_3x3):
    """Solved state encoding must be stable (same result for two identical solves)."""
    enc1 = encoder_3x3.encode(solved_3x3)
    enc2 = encoder_3x3.encode(_cube3x3_class().solved_state())
    assert np.array_equal(enc1, enc2)


def test_solved_encoding_has_correct_ones_3x3(encoder_3x3, solved_3x3):
    """In the solved state each cubie block must have exactly one '1'."""
    encoded = encoder_3x3.encode(solved_3x3)
    # 8 corners: each has (pos_one_hot[8], orient_one_hot[3]) = 11 values, 2 ones
    # 12 edges:  each has (pos_one_hot[12], orient_one_hot[2]) = 14 values, 2 ones
    # Total ones = 8*2 + 12*2 = 40
    total_ones = int(np.sum(encoded == 1.0))
    assert total_ones == 40, f"Expected 40 ones in solved encoding, got {total_ones}"


def test_solved_encoding_has_correct_ones_2x2(encoder_2x2, solved_2x2):
    """In the solved 2x2 state each corner block has exactly 2 ones."""
    encoded = encoder_2x2.encode(solved_2x2)
    # 8 corners each with pos(8) + orient(3) = 2 ones per corner
    total_ones = int(np.sum(encoded == 1.0))
    assert total_ones == 16, f"Expected 16 ones in solved 2x2 encoding, got {total_ones}"


# ---------------------------------------------------------------------------
# Batch encoding tests
# ---------------------------------------------------------------------------

def test_encode_batch_shape_3x3(encoder_3x3, solved_3x3, scrambled_3x3):
    """encode_batch for 3 puzzles returns shape (3, 256)."""
    puzzles = [solved_3x3, scrambled_3x3, solved_3x3]
    batch = encoder_3x3.encode_batch(puzzles)
    assert batch.shape == (3, 256)


def test_encode_batch_dtype_3x3(encoder_3x3, solved_3x3, scrambled_3x3):
    """encode_batch output must be float32."""
    batch = encoder_3x3.encode_batch([solved_3x3, scrambled_3x3])
    assert batch.dtype == np.float32


def test_encode_batch_matches_individual_3x3(encoder_3x3, solved_3x3, scrambled_3x3):
    """encode_batch rows must match individual encode() calls."""
    puzzles = [solved_3x3, scrambled_3x3]
    batch = encoder_3x3.encode_batch(puzzles)
    for i, puzzle in enumerate(puzzles):
        assert np.array_equal(batch[i], encoder_3x3.encode(puzzle))


# ---------------------------------------------------------------------------
# Differentiation tests
# ---------------------------------------------------------------------------

def test_solved_differs_from_scrambled_3x3(encoder_3x3, solved_3x3, scrambled_3x3):
    """Solved and scrambled 3x3 must have different cubie encodings."""
    enc_solved = encoder_3x3.encode(solved_3x3)
    enc_scrambled = encoder_3x3.encode(scrambled_3x3)
    assert not np.array_equal(enc_solved, enc_scrambled)
