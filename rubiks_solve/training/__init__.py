"""Training infrastructure — data generation, curriculum, checkpointing, metrics."""
try:
    from rubiks_solve.training.checkpoint import CheckpointManager, CheckpointMetadata
except ImportError:
    pass  # MLX not available (e.g. DGX/Linux)

from rubiks_solve.training.curriculum import CurriculumConfig, ScrambleCurriculum
from rubiks_solve.training.data_gen import ScrambleDataset
from rubiks_solve.training.metrics import EpochMetrics, MetricsTracker
from rubiks_solve.training.trainer_base import AbstractTrainer

__all__ = [
    "CheckpointManager",
    "CheckpointMetadata",
    "CurriculumConfig",
    "ScrambleCurriculum",
    "ScrambleDataset",
    "EpochMetrics",
    "MetricsTracker",
    "AbstractTrainer",
]
