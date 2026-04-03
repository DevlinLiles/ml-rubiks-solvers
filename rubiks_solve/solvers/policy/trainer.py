"""Imitation-learning trainer for CubePolicyNet.

Training data: (encoded_state, optimal_move_index) pairs generated from known
solutions (e.g. BFS on 2x2, Kociemba on 3x3).

Loss = cross-entropy(log_probs, action) - entropy_coeff * H(pi)
where H(pi) is the policy entropy (encourages exploration during training).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rubiks_solve.solvers.policy.model import CubePolicyNet

logger = logging.getLogger(__name__)


@dataclass
class PolicyTrainerConfig:
    """Hyperparameters and paths for policy-network training.

    Attributes:
        learning_rate:  Adam learning rate.
        batch_size:     Samples per gradient step.
        epochs:         Total training epochs.
        checkpoint_dir: Directory for ``.npz`` checkpoints.
        entropy_coeff:  Weight of the entropy regularisation term
                        (higher → more uniform policy during training).
        log_interval:   Log metrics every this many epochs.
    """

    learning_rate: float = 1e-4
    batch_size: int = 512
    epochs: int = 100
    checkpoint_dir: Path = Path("models/policy")
    entropy_coeff: float = 0.01
    log_interval: int = 10


class PolicyTrainer:
    """Trains :class:`CubePolicyNet` via imitation learning.

    The loss combines standard cross-entropy with an entropy bonus to
    prevent the policy from collapsing to a deterministic mapping too early
    in training.

    Args:
        model:  Policy network (modified in-place).
        config: Training hyperparameters.
    """

    def __init__(self, model: CubePolicyNet, config: PolicyTrainerConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)
        self._logger = logging.getLogger(self.__class__.__qualname__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _loss_and_acc(
        self,
        model: CubePolicyNet,
        x: mx.array,
        actions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute combined loss and accuracy for a mini-batch.

        Args:
            model:   The policy network.
            x:       Encoded states, shape ``(batch, input_size)``.
            actions: Ground-truth action indices, shape ``(batch,)``, dtype int32.

        Returns:
            Tuple ``(loss, accuracy)`` as scalar ``mx.array`` values.

        The loss is:
            L = cross_entropy(log_probs, actions) - entropy_coeff * H(pi)
        """
        log_probs = model(x)  # (batch, n_actions)

        # Cross-entropy: -log_prob[correct_action]
        ce_loss = nn.losses.cross_entropy(log_probs, actions, reduction="mean")

        # Entropy of the policy distribution: H = -sum(p * log_p)
        probs = mx.exp(log_probs)  # (batch, n_actions)
        entropy = -mx.sum(probs * log_probs, axis=-1).mean()

        loss = ce_loss - self.config.entropy_coeff * entropy

        # Accuracy: argmax prediction matches label.
        predicted = mx.argmax(log_probs, axis=-1)
        accuracy = mx.mean((predicted == actions).astype(mx.float32))

        return loss, accuracy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, float]:
        """Train for one complete pass over the dataset.

        The dataset is randomly shuffled before batching.

        Args:
            states:  Float32 array of shape ``(N, input_size)``.
            actions: Int array of shape ``(N,)`` with ground-truth move indices.

        Returns:
            Dict with keys ``'loss'`` (mean combined loss) and
            ``'accuracy'`` (mean fraction of correctly predicted moves).
        """
        n = len(states)
        idx = np.random.permutation(n)
        states = states[idx]
        actions = actions[idx].astype(np.int32)

        n_batches = math.ceil(n / self.config.batch_size)
        total_loss = 0.0
        total_acc = 0.0

        def loss_fn(model: CubePolicyNet, x: mx.array, a: mx.array) -> mx.array:
            loss, _ = self._loss_and_acc(model, x, a)
            return loss

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        for i in range(n_batches):
            start = i * self.config.batch_size
            end = min(start + self.config.batch_size, n)

            x_batch = mx.array(states[start:end])
            a_batch = mx.array(actions[start:end])

            loss_val, grads = loss_and_grad_fn(self.model, x_batch, a_batch)
            self.optimizer.update(self.model, grads)

            # Recompute accuracy (cheap, shares cached activations in eager mode).
            _, acc_val = self._loss_and_acc(self.model, x_batch, a_batch)

            mx.eval(loss_val, acc_val, self.model.parameters())

            total_loss += float(loss_val)
            total_acc += float(acc_val)

        return {
            "loss": total_loss / n_batches,
            "accuracy": total_acc / n_batches,
        }

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> list[dict[str, float]]:
        """Run the full training loop for ``config.epochs`` epochs.

        Checkpoints are written every ``config.log_interval`` epochs and
        at the final epoch.

        Args:
            states:  Float32 array of shape ``(N, input_size)``.
            actions: Int array of shape ``(N,)`` with move indices.

        Returns:
            List of per-epoch metric dicts.
        """
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []

        try:
            from tqdm import tqdm
            epoch_bar = tqdm(range(1, self.config.epochs + 1), desc="Policy", unit="epoch", dynamic_ncols=True)
        except ImportError:
            epoch_bar = range(1, self.config.epochs + 1)

        for epoch in epoch_bar:
            metrics = self.train_epoch(states, actions)
            history.append(metrics)

            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['accuracy']:.4f}")

            if epoch % self.config.log_interval == 0 or epoch == self.config.epochs:
                self._logger.info(
                    "Epoch %d/%d | loss=%.4f | accuracy=%.4f",
                    epoch,
                    self.config.epochs,
                    metrics["loss"],
                    metrics["accuracy"],
                )
                ckpt_path = self.config.checkpoint_dir / f"ckpt_epoch_{epoch:04d}.npz"
                self.save_checkpoint(ckpt_path, epoch)

        return history

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Persist model weights to a ``.npz`` file.

        Args:
            path:  Destination file path.
            epoch: Current epoch, stored as metadata inside the checkpoint.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        flat: dict[str, mx.array] = {}
        _flatten_params(dict(self.model.parameters()), "", flat)
        flat["__epoch__"] = mx.array(epoch)
        mx.savez(str(path), **flat)
        self._logger.debug("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: Path) -> int:
        """Restore model weights from a ``.npz`` checkpoint.

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            The epoch number stored in the checkpoint.
        """
        weights = mx.load(str(path))
        epoch = int(weights.pop("__epoch__"))
        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())
        self._logger.debug("Checkpoint loaded ← %s (epoch %d)", path, epoch)
        return epoch


# ---------------------------------------------------------------------------
# Shared utility (also used by other trainers in this package)
# ---------------------------------------------------------------------------

def _flatten_params(
    params: dict,
    prefix: str,
    out: dict[str, mx.array],
) -> None:
    """Recursively flatten a nested parameter dict into *out*.

    Keys are joined with ``'.'`` which ``mx.savez`` accepts and
    ``nn.Module.load_weights`` can reconstruct via its flat-key format.
    """
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_params(v, full_key, out)
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                list_key = f"{full_key}.{idx}"
                if isinstance(item, dict):
                    _flatten_params(item, list_key, out)
                else:
                    out[list_key] = item
        else:
            out[full_key] = v
