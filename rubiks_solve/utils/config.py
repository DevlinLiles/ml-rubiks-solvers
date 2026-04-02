"""Application configuration models using Pydantic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging sub-configuration.

    Attributes:
        level:     Python log level name (``"DEBUG"``, ``"INFO"``, etc.).
        log_dir:   Directory where rotating log files are written.
        json_logs: Emit console output as JSON Lines when *True*.
    """

    level: str = "INFO"
    log_dir: Path = Path("logs")
    json_logs: bool = False


class TrainingConfig(BaseModel):
    """Training sub-configuration.

    Attributes:
        seed:             Global random seed for reproducibility.
        checkpoint_dir:   Directory for model checkpoints.
        metrics_dir:      Directory for training metrics CSV files.
        keep_checkpoints: Maximum number of checkpoint files to retain.
    """

    seed: int = 42
    checkpoint_dir: Path = Path("models")
    metrics_dir: Path = Path("logs/metrics")
    keep_checkpoints: int = 5


class BenchmarkConfig(BaseModel):
    """Benchmark sub-configuration.

    Attributes:
        time_budget_seconds: Maximum wall-clock seconds per puzzle per solver.
        n_puzzles:           Number of puzzles to solve per configuration.
        scramble_depths:     List of scramble depths to benchmark.
    """

    time_budget_seconds: float = 30.0
    n_puzzles: int = 100
    scramble_depths: list[int] = Field(default_factory=lambda: [5, 10, 15, 20])


class AppConfig(BaseModel):
    """Top-level application configuration.

    Loaded from a JSON or YAML file via :meth:`from_file`.  All sub-configs
    use sensible defaults so partial files are supported.

    Attributes:
        logging:   Logging settings.
        training:  Training settings.
        benchmark: Benchmark settings.
    """

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @classmethod
    def from_file(cls, path: Path) -> "AppConfig":
        """Load configuration from a JSON or YAML file.

        YAML support requires the ``pyyaml`` package.  If the file extension is
        ``.yaml`` or ``.yml`` and PyYAML is not installed, a helpful
        ``ImportError`` is raised.

        Args:
            path: Path to a ``.json``, ``.yaml``, or ``.yml`` config file.

        Returns:
            Fully populated :class:`AppConfig` instance.

        Raises:
            ValueError:    If the file extension is not recognised.
            FileNotFoundError: If *path* does not exist.
            ImportError:   If a YAML file is given but PyYAML is not installed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                ) from exc
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            raise ValueError(
                f"Unsupported config file extension {suffix!r}. "
                "Use .json, .yaml, or .yml."
            )

        return cls.model_validate(raw)

    def save(self, path: Path) -> None:
        """Persist this configuration to a JSON file.

        Parent directories are created automatically.

        Args:
            path: Destination file path.  Must end in ``.json``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
