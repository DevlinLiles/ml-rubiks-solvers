"""PyTorch supervised trainer for CubeValueNet — DGX Spark backend."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rubiks_solve.solvers.cnn.model_torch import CubeValueNet

logger = logging.getLogger(__name__)


@dataclass
class CNNTrainerConfig:
    """Hyperparameters and paths for CNN value-network training (PyTorch).

    Matches the interface of the MLX CNNTrainerConfig.
    """

    learning_rate: float = 1e-4
    batch_size: int = 512
    epochs: int = 100
    checkpoint_dir: Path = Path("models/cnn")
    log_interval: int = 10


class CNNTrainer:
    """Trains CubeValueNet via supervised regression on scramble depth (PyTorch).

    Args:
        model:  The value network to train.
        config: Training hyperparameters.
    """

    def __init__(self, model: CubeValueNet, config: CNNTrainerConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self._logger = logging.getLogger(self.__class__.__qualname__)

    def train_epoch(self, states: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Train for one full epoch.

        Args:
            states: Float32 CUDA tensor of shape (N, input_size).
            labels: Float32 CUDA tensor of shape (N,) with scramble depths.

        Returns:
            Dict with keys 'loss' (MSE) and 'mae'.
        """
        self.model.train()
        n = states.shape[0]
        idx = torch.randperm(n, device=self.device)
        states = states[idx]
        labels = labels[idx]

        n_batches = math.ceil(n / self.config.batch_size)
        total_loss = 0.0
        total_mae = 0.0

        for i in range(n_batches):
            start = i * self.config.batch_size
            end = min(start + self.config.batch_size, n)

            x = states[start:end]
            y = labels[start:end]

            self.optimizer.zero_grad()
            preds = self.model(x).squeeze(-1)
            loss = nn.functional.mse_loss(preds, y)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                mae = torch.mean(torch.abs(preds.detach() - y))

            total_loss += loss.item()
            total_mae += mae.item()

        return {
            "loss": total_loss / n_batches,
            "mae": total_mae / n_batches,
        }

    def train(self, states: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
        """Run the full training loop for config.epochs epochs.

        Args:
            states: Float32 array of shape (N, input_size).
            labels: Float32 array of shape (N,) with scramble depths.

        Returns:
            List of per-epoch metric dicts.
        """
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []

        # Upload the full dataset to GPU once — avoids per-batch CPU→GPU copies.
        states_t = torch.from_numpy(states).to(self.device)
        labels_t = torch.from_numpy(labels).to(self.device)

        try:
            from tqdm import tqdm
            epoch_bar = tqdm(range(1, self.config.epochs + 1), desc="CNN", unit="epoch", dynamic_ncols=True)
        except ImportError:
            epoch_bar = range(1, self.config.epochs + 1)

        for epoch in epoch_bar:
            metrics = self.train_epoch(states_t, labels_t)
            history.append(metrics)

            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(loss=f"{metrics['loss']:.4f}", mae=f"{metrics['mae']:.4f}")

            if epoch % self.config.log_interval == 0 or epoch == self.config.epochs:
                self._logger.info(
                    "Epoch %d/%d | loss=%.4f | mae=%.4f",
                    epoch, self.config.epochs, metrics["loss"], metrics["mae"],
                )
                ckpt_path = self.config.checkpoint_dir / f"ckpt_epoch_{epoch:04d}.npz"
                self.save_checkpoint(ckpt_path, epoch)

        return history

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Persist model weights to a .npz file (numpy-portable format).

        Args:
            path:  Destination file path.
            epoch: Current epoch number, stored as metadata.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        arrays["__epoch__"] = np.array(epoch)
        np.savez(str(path), **arrays)
        self._logger.debug("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: Path) -> int:
        """Load model weights from a .npz checkpoint.

        Args:
            path: Path to the .npz file produced by save_checkpoint.

        Returns:
            The epoch number stored in the checkpoint.
        """
        data = np.load(str(path))
        epoch = int(data["__epoch__"])
        state_dict = {k: torch.tensor(data[k]) for k in data.files if k != "__epoch__"}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self._logger.debug("Checkpoint loaded ← %s (epoch %d)", path, epoch)
        return epoch
