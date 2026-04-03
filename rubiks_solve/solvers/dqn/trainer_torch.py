"""PyTorch DQN trainer — DGX Spark backend equivalent of trainer.py."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.solvers.dqn.model_torch import DuelingDQN
from rubiks_solve.solvers.dqn.replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)


@dataclass
class DQNTrainerConfig:
    """Hyperparameters for DQN training (PyTorch). Matches the MLX interface."""

    learning_rate: float = 5e-5
    batch_size: int = 256
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_update_freq: int = 1_000
    replay_buffer_size: int = 100_000
    min_replay_size: int = 1_000
    epochs: int = 100
    steps_per_epoch: int = 1_000
    checkpoint_dir: Path = Path("models/dqn")
    log_interval: int = 10
    max_grad_norm: float = 1.0


class DQNTrainer:
    """Trains DuelingDQN using experience replay and a target network (PyTorch).

    Args:
        model:        Online Q-network.
        target_model: Target Q-network (periodically synced from model).
        env:          Puzzle environment exposing reset/step/legal_moves.
        config:       Training hyperparameters.
        encoder:      State encoder for puzzle → array conversion.
    """

    def __init__(
        self,
        model: DuelingDQN,
        target_model: DuelingDQN,
        env: Any,
        config: DQNTrainerConfig,
        encoder: AbstractStateEncoder,
    ) -> None:
        self.model = model
        self.target_model = target_model
        self.env = env
        self.config = config
        self.encoder = encoder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
        self._rng = np.random.default_rng()
        self._global_step: int = 0
        self._logger = logging.getLogger(self.__class__.__qualname__)

        self._sync_target()

    @property
    def epsilon(self) -> float:
        """Current epsilon value for ε-greedy exploration."""
        progress = min(self._global_step / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def train_step(self, batch: list[Transition]) -> dict[str, float]:
        """One gradient update step on a sampled mini-batch."""
        states = np.stack([t.state for t in batch]).astype(np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        x = torch.from_numpy(states).to(self.device, non_blocking=True)
        x_next = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        a = torch.from_numpy(actions).to(self.device, non_blocking=True)
        r = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        d = torch.from_numpy(dones).to(self.device, non_blocking=True)

        with torch.no_grad():
            q_next = self.target_model(x_next)
            q_next_max = q_next.max(dim=-1).values
            targets = r + self.config.gamma * q_next_max * (1.0 - d)

        self.optimizer.zero_grad()
        self.model.train()
        q_all = self.model(x)
        q_selected = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.huber_loss(q_selected, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        mean_q = float(q_selected.detach().mean().item())
        return {"loss": loss.item(), "mean_q": mean_q}

    def train_epoch(self, _epoch: int) -> dict[str, float]:
        """Run steps_per_epoch environment steps and gradient updates."""
        legal_moves = self.env.legal_moves()
        n_actions = len(legal_moves)
        current_puzzle = self.env.reset()

        epoch_losses: list[float] = []
        epoch_qs: list[float] = []

        for _ in range(self.config.steps_per_epoch):
            encoded = self.encoder.encode(current_puzzle)
            eps = self.epsilon

            if self._rng.random() < eps:
                action_idx = int(self._rng.integers(n_actions))
            else:
                self.model.eval()
                with torch.no_grad():
                    x_enc = torch.from_numpy(encoded[np.newaxis, :].astype(np.float32))
                    x = x_enc.to(self.device, non_blocking=True)
                    q_vals = self.model(x)
                action_idx = int(q_vals.argmax(dim=-1).item())

            move = legal_moves[action_idx]
            next_puzzle, reward, done = self.env.step(current_puzzle, move)
            next_encoded = self.encoder.encode(next_puzzle)

            self.replay_buffer.push(
                Transition(
                    state=encoded,
                    action=action_idx,
                    reward=float(reward),
                    next_state=next_encoded,
                    done=bool(done),
                )
            )

            current_puzzle = self.env.reset() if done else next_puzzle
            self._global_step += 1

            if self.replay_buffer.is_ready(self.config.min_replay_size):
                batch = self.replay_buffer.sample(self.config.batch_size, self._rng)
                step_metrics = self.train_step(batch)
                epoch_losses.append(step_metrics["loss"])
                epoch_qs.append(step_metrics["mean_q"])

            if self._global_step % self.config.target_update_freq == 0:
                self._sync_target()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        avg_q = float(np.mean(epoch_qs)) if epoch_qs else float("nan")
        return {"loss": avg_loss, "mean_q": avg_q, "epsilon": self.epsilon}

    def train(self) -> list[dict[str, float]]:
        """Run the complete DQN training loop."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []

        if _tqdm is not None:
            epoch_bar = _tqdm(
                range(1, self.config.epochs + 1), desc="DQN", unit="epoch", dynamic_ncols=True
            )
        else:
            epoch_bar = range(1, self.config.epochs + 1)

        for epoch in epoch_bar:
            metrics = self.train_epoch(epoch)
            history.append(metrics)

            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    q=f"{metrics['mean_q']:.4f}",
                    eps=f"{metrics['epsilon']:.3f}",
                )

            if epoch % self.config.log_interval == 0 or epoch == self.config.epochs:
                self._logger.info(
                    "Epoch %d/%d | loss=%.4f | mean_q=%.4f | epsilon=%.3f",
                    epoch, self.config.epochs,
                    metrics["loss"], metrics["mean_q"], metrics["epsilon"],
                )
                ckpt_path = self.config.checkpoint_dir / f"ckpt_step_{self._global_step:07d}.npz"
                self.save_checkpoint(ckpt_path, self._global_step)

        return history

    def save_checkpoint(self, path: Path, step: int) -> None:
        """Serialise model weights and global step counter to a .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        arrays["__step__"] = np.array(step)
        np.savez(str(path), **arrays)
        self._logger.debug("Checkpoint saved → %s (step %d)", path, step)

    def load_checkpoint(self, path: Path) -> int:
        """Restore model weights from a .npz checkpoint; return the saved step."""
        data = np.load(str(path))
        step = int(data["__step__"])
        state_dict = {k: torch.tensor(data[k]) for k in data.files if k != "__step__"}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self._sync_target()
        self._global_step = step
        self._logger.debug("Checkpoint loaded ← %s (step %d)", path, step)
        return step

    def _sync_target(self) -> None:
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
