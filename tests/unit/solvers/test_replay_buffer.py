"""Tests for the DQN replay buffer."""
from __future__ import annotations

import numpy as np
import pytest

from rubiks_solve.solvers.dqn.replay_buffer import ReplayBuffer, Transition


def _make_transition(state_val: float = 0.0, action: int = 0) -> Transition:
    """Create a dummy Transition for testing."""
    state = np.array([state_val], dtype=np.float32)
    next_state = np.array([state_val + 1.0], dtype=np.float32)
    return Transition(
        state=state,
        action=action,
        reward=1.0,
        next_state=next_state,
        done=False,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Push and length tests
# ---------------------------------------------------------------------------

def test_push_and_len():
    """Buffer length increases with each push up to capacity."""
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    for i in range(5):
        buf.push(_make_transition(float(i)))
    assert len(buf) == 5


def test_push_full_capacity():
    """Buffer length stops at capacity even after extra pushes."""
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(_make_transition(float(i)))
    assert len(buf) == 5


def test_capacity_limit():
    """Pushing beyond capacity wraps around (FIFO overwrite)."""
    capacity = 4
    buf = ReplayBuffer(capacity=capacity)
    for i in range(capacity + 2):
        buf.push(_make_transition(float(i)))
    assert len(buf) == capacity


def test_capacity_property():
    """capacity property returns the value passed to __init__."""
    buf = ReplayBuffer(capacity=100)
    assert buf.capacity == 100


# ---------------------------------------------------------------------------
# Sample tests
# ---------------------------------------------------------------------------

def test_sample_size(rng):
    """sample(n) returns exactly n transitions."""
    buf = ReplayBuffer(capacity=50)
    for i in range(20):
        buf.push(_make_transition(float(i)))
    samples = buf.sample(10, rng)
    assert len(samples) == 10


def test_sample_returns_transitions(rng):
    """All items in sample() are Transition instances."""
    buf = ReplayBuffer(capacity=20)
    for i in range(10):
        buf.push(_make_transition(float(i)))
    samples = buf.sample(5, rng)
    for item in samples:
        assert isinstance(item, Transition)


def test_sample_independence(rng):
    """Two samples from the same buffer are likely different (probabilistic)."""
    buf = ReplayBuffer(capacity=50)
    for i in range(30):
        buf.push(_make_transition(float(i)))
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    sample1_states = [t.state[0] for t in buf.sample(10, rng1)]
    sample2_states = [t.state[0] for t in buf.sample(10, rng2)]
    # With different seeds and 30 items sampling 10, expect different order/selection
    assert sample1_states != sample2_states


def test_empty_sample_raises(rng):
    """Sampling an empty buffer raises ValueError."""
    buf = ReplayBuffer(capacity=100)
    with pytest.raises(ValueError):
        buf.sample(1, rng)


def test_sample_exceeds_size_raises(rng):
    """Requesting more samples than buffer size raises ValueError."""
    buf = ReplayBuffer(capacity=100)
    buf.push(_make_transition())
    buf.push(_make_transition())
    with pytest.raises(ValueError):
        buf.sample(10, rng)


# ---------------------------------------------------------------------------
# is_ready tests
# ---------------------------------------------------------------------------

def test_is_ready_false_when_empty():
    """is_ready returns False for an empty buffer."""
    buf = ReplayBuffer(capacity=100)
    assert buf.is_ready(10) is False


def test_is_ready_true_when_full_enough():
    """is_ready returns True once buffer has min_size items."""
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(_make_transition(float(i)))
    assert buf.is_ready(10) is True
    assert buf.is_ready(11) is False


# ---------------------------------------------------------------------------
# Content correctness
# ---------------------------------------------------------------------------

def test_pushed_transitions_are_retrievable(rng):
    """A single pushed transition can be retrieved via sample."""
    buf = ReplayBuffer(capacity=10)
    t = _make_transition(state_val=99.0, action=7)
    buf.push(t)
    samples = buf.sample(1, rng)
    assert samples[0].action == 7
    assert samples[0].state[0] == pytest.approx(99.0)
