"""Training metrics accumulation and reporting."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class EpochMetrics:
    """Metrics snapshot for a single training epoch.

    Core fields (always populated):

    Attributes:
        epoch:                    Epoch index (0-based).
        loss:                     Mean training loss for this epoch.
        solve_rate:               Fraction of evaluation puzzles solved
                                  (in ``[0.0, 1.0]``).
        mean_solution_length:     Mean number of moves across solved puzzles.
        median_solution_length:   Median number of moves across solved puzzles.
        mean_solve_time_seconds:  Mean wall-clock solve time in seconds.

    Solver-specific optional fields:

    Attributes:
        fitness_best:   Best population fitness (genetic solvers).
        fitness_mean:   Mean population fitness (genetic solvers).
        mae:            Mean absolute error on distance predictions (CNN).
        accuracy:       Classification accuracy (policy networks).
        mean_q:         Mean Q-value estimate (DQN).
        timestamp:      When this record was created.
        metadata:       Arbitrary extra key-value data.
    """

    epoch: int
    loss: float
    solve_rate: float
    mean_solution_length: float
    median_solution_length: float
    mean_solve_time_seconds: float
    fitness_best: float | None = None
    fitness_mean: float | None = None
    mae: float | None = None
    accuracy: float | None = None
    mean_q: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    """Accumulate :class:`EpochMetrics` records and expose export/summary helpers.

    Args:
        solver_name:  Name of the solver being trained (for labelling exports).
        puzzle_name:  Short name of the puzzle type (e.g. ``"3x3"``).

    Example::

        tracker = MetricsTracker("CNNSolver", "3x3")
        tracker.record(epoch_metrics)
        tracker.save_jsonl(Path("./runs/metrics.jsonl"))
        print(tracker.summary())
    """

    def __init__(self, solver_name: str, puzzle_name: str) -> None:
        self.solver_name = solver_name
        self.puzzle_name = puzzle_name
        self._records: list[EpochMetrics] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, metrics: EpochMetrics) -> None:
        """Append a new :class:`EpochMetrics` snapshot.

        Args:
            metrics: The metrics object to store.
        """
        self._records.append(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all recorded metrics to a :class:`pandas.DataFrame`.

        Each row corresponds to one epoch.  The ``metadata`` column contains
        the raw dict; the ``timestamp`` column is a :class:`datetime` object.

        Returns:
            A :class:`pandas.DataFrame` with one row per :class:`EpochMetrics`.
        """
        if not self._records:
            return pd.DataFrame()
        rows = []
        for m in self._records:
            d = asdict(m)
            # asdict converts datetime to a plain dict representation; restore it.
            d["timestamp"] = m.timestamp
            rows.append(d)
        return pd.DataFrame(rows)

    def save_jsonl(self, path: Path) -> None:
        """Append all records to a JSONL file (one JSON object per line).

        The file is opened in **append** mode so successive calls accumulate
        records without overwriting earlier data.

        Args:
            path: Destination file path.  Parent directories are created if they
                  do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for m in self._records:
                d = asdict(m)
                d["timestamp"] = m.timestamp.isoformat()
                d["solver_name"] = self.solver_name
                d["puzzle_name"] = self.puzzle_name
                fh.write(json.dumps(d) + "\n")

    def load_jsonl(self, path: Path) -> None:
        """Load records from a JSONL file and append them to this tracker.

        Args:
            path: Source JSONL file written by :meth:`save_jsonl`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Remove tracker-level fields not in EpochMetrics.
                d.pop("solver_name", None)
                d.pop("puzzle_name", None)
                if "timestamp" in d:
                    d["timestamp"] = datetime.fromisoformat(d["timestamp"])
                self._records.append(EpochMetrics(**d))

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary dictionary.

        Computes:

        * ``best_solve_rate`` — highest solve_rate recorded.
        * ``best_epoch``      — epoch at which best_solve_rate was achieved.
        * ``final_loss``      — loss of the most recently recorded epoch.
        * ``total_epochs``    — number of epochs recorded.
        * ``solver_name``     — forwarded from constructor.
        * ``puzzle_name``     — forwarded from constructor.

        Returns:
            A :class:`dict` of summary statistics, or an empty dict with
            ``total_epochs: 0`` if no records have been added.
        """
        if not self._records:
            return {
                "solver_name": self.solver_name,
                "puzzle_name": self.puzzle_name,
                "total_epochs": 0,
            }

        best = max(self._records, key=lambda m: m.solve_rate)
        return {
            "solver_name": self.solver_name,
            "puzzle_name": self.puzzle_name,
            "total_epochs": len(self._records),
            "best_solve_rate": best.solve_rate,
            "best_epoch": best.epoch,
            "final_loss": self._records[-1].loss,
        }

    def log_epoch(self, metrics: EpochMetrics, logger: Any) -> None:
        """Emit a structured log line for *metrics* using the provided logger.

        Compatible with both :mod:`structlog` loggers (``logger.info(event, **kw)``)
        and :mod:`logging` standard library loggers (``logger.info(msg)``) via a
        duck-typing check.

        Args:
            metrics: The :class:`EpochMetrics` to log.
            logger:  A structlog or stdlib logger instance.
        """
        fields: dict[str, Any] = {
            "solver": self.solver_name,
            "puzzle": self.puzzle_name,
            "epoch": metrics.epoch,
            "loss": round(metrics.loss, 6),
            "solve_rate": round(metrics.solve_rate, 4),
            "mean_moves": round(metrics.mean_solution_length, 2),
            "mean_solve_s": round(metrics.mean_solve_time_seconds, 4),
        }
        # Include optional fields only when set.
        for opt_key in ("fitness_best", "fitness_mean", "mae", "accuracy", "mean_q"):
            val = getattr(metrics, opt_key)
            if val is not None:
                fields[opt_key] = round(float(val), 6)

        # structlog loggers accept keyword arguments after the event string.
        if hasattr(logger, "bind") or hasattr(logger, "new"):
            logger.info("epoch_metrics", **fields)
        else:
            msg = (
                f"[{self.solver_name}/{self.puzzle_name}] "
                f"epoch={metrics.epoch} "
                + " ".join(f"{k}={v}" for k, v in fields.items() if k not in ("solver", "puzzle", "epoch"))
            )
            logger.info(msg)
