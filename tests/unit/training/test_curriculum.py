"""Tests for the ScrambleCurriculum depth scheduler."""
from __future__ import annotations

import numpy as np
import pytest

from rubiks_solve.training.curriculum import ScrambleCurriculum, CurriculumConfig


@pytest.fixture
def default_config():
    return CurriculumConfig(
        min_depth=1,
        max_depth=10,
        increase_threshold=0.8,
        increase_step=1,
        eval_window=10,
    )


@pytest.fixture
def curriculum(default_config):
    return ScrambleCurriculum(default_config)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Initial state tests
# ---------------------------------------------------------------------------

def test_initial_depth_is_min(default_config):
    """current_depth starts at min_depth."""
    curriculum = ScrambleCurriculum(default_config)
    assert curriculum.current_depth == default_config.min_depth


def test_initial_solve_rate_is_zero(curriculum):
    """current_solve_rate() returns 0.0 before any attempts are recorded."""
    assert curriculum.current_solve_rate() == 0.0


# ---------------------------------------------------------------------------
# Record attempt tests
# ---------------------------------------------------------------------------

def test_solve_rate_calculation(curriculum):
    """current_solve_rate() is correct over a partial window."""
    # Record 6 solved and 4 unsolved
    for _ in range(6):
        curriculum.record_attempt(True)
    for _ in range(4):
        curriculum.record_attempt(False)
    assert abs(curriculum.current_solve_rate() - 0.6) < 1e-9


def test_solve_rate_window_evicts_old(default_config):
    """Window caps at eval_window; older entries are evicted."""
    curriculum = ScrambleCurriculum(default_config)
    # Fill window with unsolved
    for _ in range(default_config.eval_window):
        curriculum.record_attempt(False)
    # Now add all solved on top — old unsolved should be evicted
    for _ in range(default_config.eval_window):
        curriculum.record_attempt(True)
    assert curriculum.current_solve_rate() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Depth increase tests
# ---------------------------------------------------------------------------

def test_depth_increases_after_threshold(default_config):
    """Depth increases when solve rate exceeds threshold and window is full."""
    curriculum = ScrambleCurriculum(default_config)
    initial_depth = curriculum.current_depth
    # Fill the window with successful solves (rate = 1.0 > 0.8)
    for _ in range(default_config.eval_window):
        curriculum.record_attempt(True)
    increased = curriculum.maybe_increase_depth()
    assert increased is True
    assert curriculum.current_depth == initial_depth + default_config.increase_step


def test_depth_does_not_increase_below_threshold(curriculum, default_config):
    """Depth stays the same if solve rate is below threshold."""
    initial_depth = curriculum.current_depth
    # Fill with 50% solved (below 0.8 threshold)
    for i in range(default_config.eval_window):
        curriculum.record_attempt(i % 2 == 0)
    increased = curriculum.maybe_increase_depth()
    assert increased is False
    assert curriculum.current_depth == initial_depth


def test_depth_does_not_increase_with_partial_window(curriculum, default_config):
    """Depth does not increase if fewer than eval_window attempts recorded."""
    for _ in range(default_config.eval_window - 1):
        curriculum.record_attempt(True)
    increased = curriculum.maybe_increase_depth()
    assert increased is False


def test_depth_does_not_exceed_max(default_config):
    """current_depth is clamped to max_depth regardless of how many thresholds are met."""
    curriculum = ScrambleCurriculum(default_config)
    # Drive the depth to max by repeatedly filling the window with successes
    for _ in range(100):
        for _ in range(default_config.eval_window):
            curriculum.record_attempt(True)
        curriculum.maybe_increase_depth()
    assert curriculum.current_depth <= default_config.max_depth


def test_window_cleared_after_depth_increase(default_config):
    """After a depth increase the window is cleared (solve rate resets to 0.0)."""
    curriculum = ScrambleCurriculum(default_config)
    for _ in range(default_config.eval_window):
        curriculum.record_attempt(True)
    curriculum.maybe_increase_depth()
    assert curriculum.current_solve_rate() == 0.0


# ---------------------------------------------------------------------------
# sample_depth tests
# ---------------------------------------------------------------------------

def test_sample_depth_in_range(curriculum, rng):
    """Sampled depth is always in [min_depth, current_depth]."""
    min_d = curriculum._config.min_depth
    for _ in range(50):
        d = curriculum.sample_depth(rng)
        assert min_d <= d <= curriculum.current_depth, (
            f"Sampled depth {d} out of range [{min_d}, {curriculum.current_depth}]"
        )


def test_sample_depth_at_min_when_min_equals_max():
    """When min_depth == current_depth, sample_depth always returns that depth."""
    config = CurriculumConfig(min_depth=5, max_depth=5)
    curriculum = ScrambleCurriculum(config)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert curriculum.sample_depth(rng) == 5


def test_sample_depth_after_increase(default_config):
    """After depth increase, sampled depths can reach the new depth."""
    curriculum = ScrambleCurriculum(default_config)
    for _ in range(default_config.eval_window):
        curriculum.record_attempt(True)
    curriculum.maybe_increase_depth()
    rng = np.random.default_rng(7)
    sampled = [curriculum.sample_depth(rng) for _ in range(100)]
    assert max(sampled) == curriculum.current_depth
