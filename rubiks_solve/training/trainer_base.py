"""Abstract base class for offline solvers trainers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from rubiks_solve.core.base import AbstractPuzzle
try:
    from rubiks_solve.training.checkpoint import CheckpointManager, CheckpointMetadata
except ImportError:
    CheckpointManager = None  # type: ignore[assignment,misc]
    CheckpointMetadata = None  # type: ignore[assignment,misc]
from rubiks_solve.training.curriculum import ScrambleCurriculum
from rubiks_solve.training.data_gen import ScrambleDataset
from rubiks_solve.training.metrics import EpochMetrics, MetricsTracker


class AbstractTrainer(ABC):
    """Shared epoch-loop scaffolding for all offline trainers.

    Subclasses implement :meth:`train_epoch` and :meth:`evaluate`.
    :class:`AbstractTrainer` coordinates:

    * **Curriculum** — calls :meth:`~ScrambleCurriculum.maybe_increase_depth`
      after each epoch so scramble difficulty grows as the solver improves.
    * **Checkpointing** — saves a checkpoint every 100 epochs and restores the
      latest one if one already exists (warm restart).
    * **Metrics tracking** — records an :class:`~metrics.EpochMetrics` object
      per epoch and logs it via the trainer's logger.

    Args:
        solver_name:         Human-readable name of the solver being trained.
        puzzle_type:         The :class:`~rubiks_solve.core.base.AbstractPuzzle`
                             subclass this trainer targets.
        dataset:             :class:`~data_gen.ScrambleDataset` used to generate
                             training batches.
        curriculum:          :class:`~curriculum.ScrambleCurriculum` that controls
                             scramble depth scheduling.
        checkpoint_manager:  :class:`~checkpoint.CheckpointManager` for saving
                             and restoring model weights.
        metrics_tracker:     :class:`~metrics.MetricsTracker` that accumulates
                             per-epoch metric records.

    Example::

        class MyCNNTrainer(AbstractTrainer):
            def train_epoch(self, epoch):
                states, labels = self.dataset.generate_batch(256)
                loss = self.model.update(states, labels)
                return {"loss": loss}

            def evaluate(self, n_puzzles=100):
                ...
                return {"solve_rate": rate, "mean_solution_length": mean_moves,
                        "median_solution_length": med_moves,
                        "mean_solve_time_seconds": mean_t}

        trainer = MyCNNTrainer(...)
        metrics = trainer.train(n_epochs=50)
    """

    def __init__(
        self,
        solver_name: str,
        puzzle_type: type[AbstractPuzzle],
        dataset: ScrambleDataset,
        curriculum: ScrambleCurriculum,
        checkpoint_manager: CheckpointManager,
        metrics_tracker: MetricsTracker,
    ) -> None:
        self.solver_name = solver_name
        self.puzzle_type = puzzle_type
        self.dataset = dataset
        self.curriculum = curriculum
        self.checkpoint_manager = checkpoint_manager
        self.metrics_tracker = metrics_tracker
        self._logger = logging.getLogger(self.__class__.__qualname__)
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one full training epoch.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            A :class:`dict` of metric names to scalar values.  Must include at
            minimum ``{"loss": <float>}``.
        """

    @abstractmethod
    def evaluate(self, n_puzzles: int = 100) -> dict[str, float]:
        """Evaluate the solver on *n_puzzles* random scrambles.

        Args:
            n_puzzles: Number of scrambles to attempt.

        Returns:
            A :class:`dict` that must include at minimum:

            * ``"solve_rate"``              — fraction solved.
            * ``"mean_solution_length"``    — mean moves across solved puzzles.
            * ``"median_solution_length"``  — median moves across solved puzzles.
            * ``"mean_solve_time_seconds"`` — mean wall-clock time in seconds.

            Optional solver-specific keys: ``"mae"``, ``"accuracy"``,
            ``"mean_q"``, ``"fitness_best"``, ``"fitness_mean"``.
        """

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_epochs: int) -> MetricsTracker:
        """Run the full training loop for *n_epochs* epochs.

        The loop:

        1. Calls :meth:`train_epoch` to run one epoch of gradient updates.
        2. Calls :meth:`evaluate` to measure current solver quality.
        3. Records an :class:`~metrics.EpochMetrics` snapshot and logs it.
        4. Calls :meth:`~curriculum.ScrambleCurriculum.maybe_increase_depth`
           to potentially advance the curriculum.
        5. Saves a checkpoint via :class:`~checkpoint.CheckpointManager`.

        Args:
            n_epochs: Number of training epochs to run.

        Returns:
            The :class:`~metrics.MetricsTracker` with all recorded metrics.
        """
        self._logger.info(
            "Starting training: solver=%s puzzle=%s epochs=%d",
            self.solver_name,
            self.puzzle_type.puzzle_name(),
            n_epochs,
        )

        for epoch in range(n_epochs):
            # --- Train ---
            train_metrics = self.train_epoch(epoch)
            self._global_step += 1

            # --- Evaluate ---
            eval_metrics = self.evaluate()

            # --- Build EpochMetrics ---
            epoch_metrics = self._build_epoch_metrics(epoch, train_metrics, eval_metrics)
            self.metrics_tracker.record(epoch_metrics)
            self.metrics_tracker.log_epoch(epoch_metrics, self._logger)

            # --- Curriculum update ---
            depth_increased = self.curriculum.maybe_increase_depth()
            if depth_increased:
                self._logger.info(
                    "Curriculum advanced to depth=%d", self.curriculum.current_depth
                )

            # --- Checkpoint (every 100 epochs) ---
            if (epoch + 1) % 100 == 0:
                self._save_checkpoint(epoch)

        self._logger.info("Training complete after %d epochs.", n_epochs)
        return self.metrics_tracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_epoch_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
    ) -> EpochMetrics:
        """Combine train and eval dicts into an :class:`~metrics.EpochMetrics`.

        Args:
            epoch:        Current epoch index.
            train_metrics: Output of :meth:`train_epoch`.
            eval_metrics:  Output of :meth:`evaluate`.

        Returns:
            A fully-populated :class:`~metrics.EpochMetrics` instance.
        """
        return EpochMetrics(
            epoch=epoch,
            loss=float(train_metrics.get("loss", float("nan"))),
            solve_rate=float(eval_metrics.get("solve_rate", 0.0)),
            mean_solution_length=float(eval_metrics.get("mean_solution_length", 0.0)),
            median_solution_length=float(eval_metrics.get("median_solution_length", 0.0)),
            mean_solve_time_seconds=float(eval_metrics.get("mean_solve_time_seconds", 0.0)),
            fitness_best=eval_metrics.get("fitness_best"),
            fitness_mean=eval_metrics.get("fitness_mean"),
            mae=eval_metrics.get("mae"),
            accuracy=eval_metrics.get("accuracy"),
            mean_q=eval_metrics.get("mean_q"),
            timestamp=datetime.now(timezone.utc),
            metadata={**train_metrics, **eval_metrics},
        )

    def _save_checkpoint(self, epoch: int) -> None:
        """Persist a checkpoint for *epoch* via the checkpoint manager.

        Requires the subclass to expose a ``model`` attribute and optionally an
        ``optimizer`` attribute.  If neither exists, checkpointing is skipped
        with a warning.

        Args:
            epoch: Current epoch index used to label the checkpoint file.
        """
        model = getattr(self, "model", None)
        if model is None:
            self._logger.warning(
                "Subclass has no 'model' attribute — skipping checkpoint save."
            )
            return

        optimizer = getattr(self, "optimizer", None)
        optimizer_state: dict[str, Any] = {}
        if optimizer is not None and hasattr(optimizer, "state"):
            optimizer_state = dict(optimizer.state)

        metadata = CheckpointMetadata(
            epoch=epoch,
            step=self._global_step,
            solver_name=self.solver_name,
            puzzle_name=self.puzzle_type.puzzle_name(),
            config=self._config_as_dict(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            path = self.checkpoint_manager.save(model, optimizer_state, metadata)
            self._logger.debug("Checkpoint saved: %s", path)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            self._logger.error("Checkpoint save failed: %s", exc)

    def _config_as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this trainer's config.

        Looks for a ``config`` attribute on self and tries to convert it to a
        dict.  Falls back to an empty dict if not available.

        Returns:
            A plain :class:`dict` suitable for JSON serialisation.
        """
        cfg = getattr(self, "config", None)
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return cfg
        # dataclass or object with __dict__
        try:
            return asdict(cfg)
        except TypeError:
            return vars(cfg) if hasattr(cfg, "__dict__") else {}
