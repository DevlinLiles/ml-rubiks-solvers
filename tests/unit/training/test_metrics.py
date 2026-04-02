"""Tests for the MetricsTracker training metrics system."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from rubiks_solve.training.metrics import EpochMetrics, MetricsTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(epoch: int, solve_rate: float = 0.5, loss: float = 1.0) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        loss=loss,
        solve_rate=solve_rate,
        mean_solution_length=10.0,
        median_solution_length=9.0,
        mean_solve_time_seconds=0.5,
    )


@pytest.fixture
def tracker():
    return MetricsTracker(solver_name="TestSolver", puzzle_name="3x3")


# ---------------------------------------------------------------------------
# record and to_dataframe tests
# ---------------------------------------------------------------------------

def test_record_and_retrieve(tracker):
    """Recorded metrics appear in to_dataframe()."""
    m = _make_metrics(epoch=0, solve_rate=0.75, loss=0.3)
    tracker.record(m)
    df = tracker.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["epoch"] == 0
    assert abs(df.iloc[0]["solve_rate"] - 0.75) < 1e-9
    assert abs(df.iloc[0]["loss"] - 0.3) < 1e-9


def test_to_dataframe_multiple_records(tracker):
    """to_dataframe() contains one row per recorded epoch."""
    for i in range(5):
        tracker.record(_make_metrics(epoch=i))
    df = tracker.to_dataframe()
    assert len(df) == 5
    assert list(df["epoch"]) == [0, 1, 2, 3, 4]


def test_to_dataframe_has_expected_columns(tracker):
    """DataFrame contains key column names."""
    tracker.record(_make_metrics(0))
    df = tracker.to_dataframe()
    for col in ("epoch", "loss", "solve_rate", "mean_solution_length", "timestamp"):
        assert col in df.columns, f"Missing column: {col}"


def test_empty_tracker_dataframe(tracker):
    """to_dataframe() on an empty tracker returns an empty DataFrame."""
    df = tracker.to_dataframe()
    assert len(df) == 0


# ---------------------------------------------------------------------------
# summary tests
# ---------------------------------------------------------------------------

def test_summary_best_epoch(tracker):
    """summary identifies the epoch with the highest solve_rate."""
    tracker.record(_make_metrics(epoch=0, solve_rate=0.3, loss=2.0))
    tracker.record(_make_metrics(epoch=1, solve_rate=0.9, loss=1.0))
    tracker.record(_make_metrics(epoch=2, solve_rate=0.6, loss=0.5))
    s = tracker.summary()
    assert s["best_epoch"] == 1
    assert abs(s["best_solve_rate"] - 0.9) < 1e-9


def test_summary_total_epochs(tracker):
    """summary reports correct total_epochs count."""
    for i in range(7):
        tracker.record(_make_metrics(epoch=i))
    s = tracker.summary()
    assert s["total_epochs"] == 7


def test_summary_final_loss(tracker):
    """summary reports the loss of the last epoch as final_loss."""
    tracker.record(_make_metrics(epoch=0, loss=5.0))
    tracker.record(_make_metrics(epoch=1, loss=1.5))
    s = tracker.summary()
    assert abs(s["final_loss"] - 1.5) < 1e-9


def test_empty_tracker_summary(tracker):
    """summary on an empty tracker returns a dict with total_epochs=0 and no error."""
    s = tracker.summary()
    assert s["total_epochs"] == 0
    assert s["solver_name"] == "TestSolver"
    assert s["puzzle_name"] == "3x3"


def test_summary_solver_and_puzzle_names(tracker):
    """summary includes solver_name and puzzle_name."""
    tracker.record(_make_metrics(0))
    s = tracker.summary()
    assert s["solver_name"] == "TestSolver"
    assert s["puzzle_name"] == "3x3"


# ---------------------------------------------------------------------------
# JSONL save/load roundtrip tests
# ---------------------------------------------------------------------------

def test_jsonl_roundtrip(tracker):
    """save then load produces the same records."""
    tracker.record(_make_metrics(epoch=0, solve_rate=0.4, loss=2.1))
    tracker.record(_make_metrics(epoch=1, solve_rate=0.8, loss=0.9))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metrics.jsonl"
        tracker.save_jsonl(path)

        # Load into a fresh tracker
        new_tracker = MetricsTracker(solver_name="TestSolver", puzzle_name="3x3")
        new_tracker.load_jsonl(path)

        orig_df = tracker.to_dataframe()
        loaded_df = new_tracker.to_dataframe()

        assert len(loaded_df) == len(orig_df)
        for col in ("epoch", "loss", "solve_rate"):
            assert abs((loaded_df[col] - orig_df[col]).sum()) < 1e-6, (
                f"Column {col} differs after roundtrip"
            )


def test_jsonl_format_is_valid_json(tracker):
    """Each line in the JSONL file must be parseable JSON."""
    tracker.record(_make_metrics(0))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        tracker.save_jsonl(path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    assert "epoch" in obj


def test_jsonl_save_appends(tracker):
    """save_jsonl appends to existing file without overwriting."""
    tracker.record(_make_metrics(0))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "append.jsonl"
        tracker.save_jsonl(path)  # first save: 1 record
        tracker.record(_make_metrics(1))
        tracker.save_jsonl(path)  # second save: 2 more records (cumulative)

        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        # First save wrote 1 record; second save wrote 2 records (all current records)
        # Total lines = 1 + 2 = 3
        assert len(lines) == 3


def test_jsonl_file_created_if_missing(tracker):
    """save_jsonl creates the file (and parent dirs) if they do not exist."""
    tracker.record(_make_metrics(0))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "metrics.jsonl"
        tracker.save_jsonl(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Optional field handling
# ---------------------------------------------------------------------------

def test_optional_fields_preserved(tracker):
    """Optional fields like fitness_best survive a JSONL roundtrip."""
    m = _make_metrics(0)
    m.fitness_best = -42.5
    m.accuracy = 0.95
    tracker.record(m)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "opt.jsonl"
        tracker.save_jsonl(path)
        new_tracker = MetricsTracker("TestSolver", "3x3")
        new_tracker.load_jsonl(path)
        df = new_tracker.to_dataframe()
        assert abs(df.iloc[0]["fitness_best"] - (-42.5)) < 1e-9
        assert abs(df.iloc[0]["accuracy"] - 0.95) < 1e-9
