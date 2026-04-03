"""PyTorch imitation-learning trainer for CubePolicyNet — DGX Spark backend."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from rubiks_solve.solvers.policy.model_torch import CubePolicyNet

logger = logging.getLogger(__name__)


@dataclass
class PolicyTrainerConfig:
    """Hyperparameters for policy-network training (PyTorch). Matches MLX interface."""

    learning_rate: float = 1e-4
    batch_size: int = 512
    epochs: int = 100
    checkpoint_dir: Path = Path("models/policy")
    entropy_coeff: float = 0.01
    log_interval: int = 10


class PolicyTrainer:
    """Trains CubePolicyNet via imitation learning (PyTorch).

    Loss = cross_entropy(log_probs, actions) - entropy_coeff * H(pi)

    Args:
        model:  Policy network (modified in-place).
        config: Training hyperparameters.
    """

    def __init__(self, model: CubePolicyNet, config: PolicyTrainerConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self._logger = logging.getLogger(self.__class__.__qualname__)

    def _loss_and_acc(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = self.model(x)                              # (batch, n_actions)
        ce_loss = nn.functional.nll_loss(log_probs, actions)  # cross-entropy on log-probs
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = ce_loss - self.config.entropy_coeff * entropy
        predicted = log_probs.argmax(dim=-1)
        accuracy = (predicted == actions).float().mean()
        return loss, accuracy

    def train_epoch(self, states: torch.Tensor, actions: torch.Tensor) -> dict[str, float]:
        """Train for one full epoch.

        Args:
            states:  Float32 CUDA tensor of shape (N, input_size).
            actions: Long CUDA tensor of shape (N,) with ground-truth move indices.

        Returns:
            Dict with keys 'loss' and 'accuracy'.
        """
        self.model.train()
        n = states.shape[0]
        idx = torch.randperm(n, device=self.device)
        states = states[idx]
        actions = actions[idx]

        n_batches = math.ceil(n / self.config.batch_size)
        total_loss = 0.0
        total_acc = 0.0

        for i in range(n_batches):
            start = i * self.config.batch_size
            end = min(start + self.config.batch_size, n)

            x = states[start:end]
            a = actions[start:end]

            self.optimizer.zero_grad()
            loss, acc = self._loss_and_acc(x, a)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        return {
            "loss": total_loss / n_batches,
            "accuracy": total_acc / n_batches,
        }

    def train(self, states: np.ndarray, actions: np.ndarray) -> list[dict[str, float]]:
        """Run the full training loop for config.epochs epochs.

        Args:
            states:  Float32 array of shape (N, input_size).
            actions: Int array of shape (N,) with move indices.

        Returns:
            List of per-epoch metric dicts.
        """
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []

        # Upload the full dataset to GPU once — avoids per-batch CPU→GPU copies.
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions.astype(np.int64)).to(self.device)

        if _tqdm is not None:
            epoch_bar = _tqdm(
                range(1, self.config.epochs + 1), desc="Policy", unit="epoch", dynamic_ncols=True
            )
        else:
            epoch_bar = range(1, self.config.epochs + 1)

        for epoch in epoch_bar:
            metrics = self.train_epoch(states_t, actions_t)
            history.append(metrics)

            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['accuracy']:.4f}")

            if epoch % self.config.log_interval == 0 or epoch == self.config.epochs:
                self._logger.info(
                    "Epoch %d/%d | loss=%.4f | accuracy=%.4f",
                    epoch, self.config.epochs, metrics["loss"], metrics["accuracy"],
                )
                ckpt_path = self.config.checkpoint_dir / f"ckpt_epoch_{epoch:04d}.npz"
                self.save_checkpoint(ckpt_path, epoch)

        return history

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Persist model weights to a .npz file (numpy-portable format)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        arrays["__epoch__"] = np.array(epoch)
        np.savez(str(path), **arrays)
        self._logger.debug("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: Path) -> int:
        """Load model weights from a .npz checkpoint."""
        data = np.load(str(path))
        epoch = int(data["__epoch__"])
        state_dict = {k: torch.tensor(data[k]) for k in data.files if k != "__epoch__"}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self._logger.debug("Checkpoint loaded ← %s (epoch %d)", path, epoch)
        return epoch
