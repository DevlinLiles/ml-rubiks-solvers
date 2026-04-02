"""Tests for the ScrambleDataset training data generator."""
from __future__ import annotations

import numpy as np
import pytest

from rubiks_solve.training.data_gen import ScrambleDataset
from rubiks_solve.encoding.one_hot import OneHotEncoder


def _make_dataset(seed: int = 42) -> ScrambleDataset:
    from rubiks_solve.core.cube_3x3 import Cube3x3
    encoder = OneHotEncoder(Cube3x3)
    rng = np.random.default_rng(seed)
    return ScrambleDataset(
        puzzle_factory=Cube3x3.solved_state,
        encoder=encoder,
        rng=rng,
    )


@pytest.fixture
def dataset():
    return _make_dataset(42)


@pytest.fixture
def encoder():
    from rubiks_solve.core.cube_3x3 import Cube3x3
    return OneHotEncoder(Cube3x3)


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------

def test_generate_batch_shapes(dataset, encoder):
    """generate_batch returns arrays of the correct shapes."""
    batch_size = 32
    states, depths = dataset.generate_batch(batch_size=batch_size, min_depth=1, max_depth=10)
    assert states.shape == (batch_size, *encoder.output_shape)
    assert depths.shape == (batch_size,)


def test_depths_in_range(dataset):
    """All depth labels must lie within [min_depth, max_depth]."""
    min_depth, max_depth = 3, 8
    _, depths = dataset.generate_batch(batch_size=100, min_depth=min_depth, max_depth=max_depth)
    assert np.all(depths >= min_depth), f"Some depths below min: {depths.min()}"
    assert np.all(depths <= max_depth), f"Some depths above max: {depths.max()}"


def test_states_are_float32(dataset):
    """Encoded states must have dtype float32."""
    states, _ = dataset.generate_batch(batch_size=10)
    assert states.dtype == np.float32


def test_depths_are_int64(dataset):
    """Depth labels must have dtype int64."""
    _, depths = dataset.generate_batch(batch_size=10)
    assert depths.dtype == np.int64


def test_different_seeds_differ():
    """Batches from different RNG seeds produce different state arrays."""
    ds1 = _make_dataset(seed=1)
    ds2 = _make_dataset(seed=2)
    states1, _ = ds1.generate_batch(batch_size=10, min_depth=5, max_depth=10)
    states2, _ = ds2.generate_batch(batch_size=10, min_depth=5, max_depth=10)
    assert not np.array_equal(states1, states2)


def test_same_seed_same_batch():
    """Same seed produces identical batches (reproducibility)."""
    ds1 = _make_dataset(seed=99)
    ds2 = _make_dataset(seed=99)
    s1, d1 = ds1.generate_batch(batch_size=8)
    s2, d2 = ds2.generate_batch(batch_size=8)
    assert np.array_equal(s1, s2)
    assert np.array_equal(d1, d2)


# ---------------------------------------------------------------------------
# Policy batch tests
# ---------------------------------------------------------------------------

def test_policy_batch_move_indices_valid(dataset):
    """All action indices must be within [0, n_legal_moves) or -1."""
    from rubiks_solve.core.cube_3x3 import Cube3x3
    n_actions = len(Cube3x3.solved_state().legal_moves())
    _, labels = dataset.generate_policy_batch(batch_size=50, min_depth=1, max_depth=5,
                                               n_actions=n_actions)
    # Labels should be -1 or in [0, n_actions)
    valid_mask = (labels == -1) | ((labels >= 0) & (labels < n_actions))
    assert np.all(valid_mask), f"Invalid action indices found: {labels[~valid_mask]}"


def test_policy_batch_shapes(dataset, encoder):
    """generate_policy_batch returns correct shapes."""
    from rubiks_solve.core.cube_3x3 import Cube3x3
    n_actions = len(Cube3x3.solved_state().legal_moves())
    batch_size = 20
    states, labels = dataset.generate_policy_batch(
        batch_size=batch_size, min_depth=1, max_depth=5, n_actions=n_actions
    )
    assert states.shape == (batch_size, *encoder.output_shape)
    assert labels.shape == (batch_size,)


def test_policy_batch_labels_dtype(dataset):
    """Policy batch labels must be int64."""
    from rubiks_solve.core.cube_3x3 import Cube3x3
    n_actions = len(Cube3x3.solved_state().legal_moves())
    _, labels = dataset.generate_policy_batch(batch_size=5, n_actions=n_actions)
    assert labels.dtype == np.int64


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_min_depth_equals_max_depth(dataset):
    """When min_depth == max_depth all depths must equal that value."""
    depth = 5
    _, depths = dataset.generate_batch(batch_size=20, min_depth=depth, max_depth=depth)
    assert np.all(depths == depth)


def test_invalid_min_depth_raises(dataset):
    """min_depth < 1 raises ValueError."""
    with pytest.raises(ValueError, match="min_depth"):
        dataset.generate_batch(batch_size=5, min_depth=0, max_depth=5)


def test_max_depth_less_than_min_raises(dataset):
    """max_depth < min_depth raises ValueError."""
    with pytest.raises(ValueError, match="max_depth"):
        dataset.generate_batch(batch_size=5, min_depth=5, max_depth=3)
