"""Checkpoint management for MLX models and training state."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside every model checkpoint.

    Attributes:
        epoch:        Training epoch at the time of saving.
        step:         Global optimisation step at the time of saving.
        solver_name:  Name of the solver class being trained.
        puzzle_name:  Short name of the puzzle type (e.g. ``"3x3"``).
        config:       Arbitrary serialisable config dict (hyperparameters, etc.).
        timestamp:    ISO-8601 UTC timestamp string set at save time.
    """

    epoch: int
    step: int
    solver_name: str
    puzzle_name: str
    config: dict[str, Any]
    timestamp: str


class CheckpointManager:
    """Save and restore MLX model weights and training state.

    Model weights are stored with :func:`mlx.core.savez` (``.npz`` format).
    Metadata (epoch, step, config, …) is stored as a companion ``.json`` file
    with the same stem.

    A maximum of *keep_last* checkpoints are retained; older ones are deleted
    automatically after each :meth:`save` call.

    Args:
        checkpoint_dir: Directory in which checkpoints are stored.  Created on
                        first save if it does not exist.
        keep_last:      Number of most-recent checkpoints to keep.  Defaults to
                        ``5``.

    Example::

        manager = CheckpointManager(Path("./checkpoints"), keep_last=3)
        path = manager.save(model, optimizer_state, metadata)
        meta = manager.load_latest(model)
    """

    _WEIGHTS_SUFFIX = ".npz"
    _META_SUFFIX = ".json"

    def __init__(self, checkpoint_dir: Path, keep_last: int = 5) -> None:
        self._dir = Path(checkpoint_dir)
        self._keep_last = keep_last

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        optimizer_state: dict[str, Any],
        metadata: CheckpointMetadata,
    ) -> Path:
        """Save a checkpoint to disk.

        The weights file is named ``ckpt_epoch{epoch:04d}_step{step:08d}.npz``
        and the metadata file shares the same stem with a ``.json`` extension.

        After saving, old checkpoints beyond *keep_last* are pruned.

        Args:
            model:           The MLX model whose ``parameters()`` are saved.
            optimizer_state: Serialisable dictionary of optimizer state (e.g.
                             momentum buffers).  Values must be MLX arrays or
                             Python scalars.
            metadata:        A :class:`CheckpointMetadata` instance.

        Returns:
            The :class:`~pathlib.Path` to the saved ``.npz`` weights file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        stem = f"ckpt_epoch{metadata.epoch:04d}_step{metadata.step:08d}"
        weights_path = self._dir / (stem + self._WEIGHTS_SUFFIX)
        meta_path = self._dir / (stem + self._META_SUFFIX)

        # Flatten model parameters into a {name: array} dict.
        flat_weights = dict(mx.tree_flatten(model.parameters()))
        # Merge optimizer state (prefix keys to avoid collisions).
        flat_opt = {
            f"__opt_{k}": v for k, v in mx.tree_flatten(optimizer_state)
        }
        mx.savez(str(weights_path), **flat_weights, **flat_opt)

        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(metadata), fh, indent=2)

        self._prune_old_checkpoints()
        return weights_path

    def load_latest(self, model: Any) -> CheckpointMetadata | None:
        """Load the most recent checkpoint into *model*.

        Args:
            model: The MLX model whose parameters will be updated in-place.

        Returns:
            The :class:`CheckpointMetadata` of the loaded checkpoint, or
            ``None`` if no checkpoints exist in the checkpoint directory.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        weights_path, metadata = checkpoints[0]
        self.load(model, weights_path)
        return metadata

    def load(self, model: Any, checkpoint_path: Path) -> CheckpointMetadata:
        """Load a specific checkpoint into *model*.

        Args:
            model:           The MLX model to update in-place.
            checkpoint_path: Path to the ``.npz`` weights file.

        Returns:
            The :class:`CheckpointMetadata` read from the companion ``.json``
            file.

        Raises:
            FileNotFoundError: If *checkpoint_path* or its companion metadata
                               file does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        meta_path = checkpoint_path.with_suffix(self._META_SUFFIX)
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # Load weights and restore model parameters (ignore optimizer keys).
        loaded = dict(mx.load(str(checkpoint_path)))
        model_weights = {
            k: v for k, v in loaded.items() if not k.startswith("__opt_")
        }
        model.load_weights(list(model_weights.items()))
        mx.eval(model.parameters())

        with meta_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        return CheckpointMetadata(**raw)

    def list_checkpoints(self) -> list[tuple[Path, CheckpointMetadata]]:
        """Return all checkpoints in the directory, sorted newest-first.

        Returns:
            A list of ``(weights_path, metadata)`` tuples.  The list is sorted
            in descending order by ``(epoch, step)`` so the first entry is
            always the most recent checkpoint.
        """
        if not self._dir.exists():
            return []

        results: list[tuple[Path, CheckpointMetadata]] = []
        for meta_path in self._dir.glob(f"*{self._META_SUFFIX}"):
            weights_path = meta_path.with_suffix(self._WEIGHTS_SUFFIX)
            if not weights_path.exists():
                continue
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                metadata = CheckpointMetadata(**raw)
                results.append((weights_path, metadata))
            except Exception:  # noqa: BLE001
                continue

        results.sort(key=lambda t: (t[1].epoch, t[1].step), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old_checkpoints(self) -> None:
        """Delete checkpoints older than the *keep_last* most recent ones."""
        checkpoints = self.list_checkpoints()
        for weights_path, _ in checkpoints[self._keep_last :]:
            meta_path = weights_path.with_suffix(self._META_SUFFIX)
            weights_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    @staticmethod
    def _utc_now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()
