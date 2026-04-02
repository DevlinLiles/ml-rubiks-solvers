"""Training infrastructure — data generation, curriculum, checkpointing, metrics."""
from rubiks_solve.training.checkpoint import CheckpointManager, CheckpointMetadata
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
