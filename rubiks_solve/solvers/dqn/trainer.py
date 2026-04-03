"""DQN trainer with epsilon-greedy exploration, experience replay, and target network."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import mlx.core as mx
import mlx.nn as nn  # pylint: disable=consider-using-from-import
import mlx.optimizers as optim

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.solvers.dqn.model import DuelingDQN
from rubiks_solve.solvers.dqn.replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)


@dataclass
class DQNTrainerConfig:
    """Hyperparameters for DQN training.

    Attributes:
        learning_rate:       Adam learning rate.
        batch_size:          Transitions per gradient step.
        gamma:               Discount factor for future rewards.
        epsilon_start:       Initial exploration probability.
        epsilon_end:         Final (minimum) exploration probability.
        epsilon_decay_steps: Number of environment steps over which epsilon
                             is linearly decayed from start to end.
        target_update_freq:  Copy online weights → target every this many steps.
        replay_buffer_size:  Maximum transitions in the replay buffer.
        min_replay_size:     Start training only after the buffer holds this many
                             transitions.
        epochs:              Total number of training epochs.
        steps_per_epoch:     Environment steps (and potential gradient updates)
                             per epoch.
        checkpoint_dir:      Directory for ``.npz`` checkpoints.
        log_interval:        Log metrics every this many epochs.
    """

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
    """Trains :class:`DuelingDQN` using experience replay and a target network.

    The training loop follows the standard DQN recipe:
    1. Collect experience with epsilon-greedy policy.
    2. Store transitions in a :class:`ReplayBuffer`.
    3. Sample mini-batches and compute TD targets using the frozen target net.
    4. Update the online network with a gradient step.
    5. Periodically hard-copy online weights to the target network.

    Args:
        model:        Online Q-network (modified in-place).
        target_model: Target Q-network (periodically synced from *model*).
        env:          Puzzle environment.  Must expose:
                      - ``reset() -> AbstractPuzzle`` — return a scrambled puzzle.
                      - ``step(puzzle, move) -> (next_puzzle, reward, done)`` — apply
                        one move and return the outcome.
                      - ``legal_moves() -> list[Move]`` — list of all moves.
        config:       Training hyperparameters.
        encoder:      State encoder used to convert puzzle states to arrays.
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

        self.optimizer = optim.Adam(learning_rate=config.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
        self._rng = np.random.default_rng()

        self._global_step: int = 0  # Total environment steps taken so far.
        self._logger = logging.getLogger(self.__class__.__qualname__)

        # Initialise target network to match the online network.
        self._sync_target()

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current exploration probability (linearly annealed)."""
        progress = min(self._global_step / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    # ------------------------------------------------------------------
    # Training primitives
    # ------------------------------------------------------------------

    def train_step(self, batch: list[Transition]) -> dict[str, float]:
        """Perform one gradient update on the online network.

        Computes the Bellman TD target using the frozen target network:

            y_i = r_i + gamma * max_a' Q_target(s'_i, a')   (if not done)
            y_i = r_i                                         (if done)

        Then minimises MSE between Q_online(s_i, a_i) and y_i.

        Args:
            batch: Mini-batch of :class:`Transition` objects.

        Returns:
            Dict with keys ``'loss'`` (mean TD loss) and
            ``'mean_q'`` (mean predicted Q-value for selected actions).
        """
        states = np.stack([t.state for t in batch]).astype(np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        x = mx.array(states)
        x_next = mx.array(next_states)
        r = mx.array(rewards)
        d = mx.array(dones)
        a = mx.array(actions)

        # --- Compute TD targets with the target network (no grad needed). ---
        q_next = self.target_model(x_next)  # (batch, n_actions)
        mx.eval(q_next)
        q_next_max = mx.max(q_next, axis=-1)  # (batch,)
        targets = r + self.config.gamma * q_next_max * (1.0 - d)  # (batch,)
        # Detach targets from the computation graph.
        targets = mx.stop_gradient(targets)

        def loss_fn(model: DuelingDQN, x_: mx.array, a_: mx.array, t_: mx.array) -> mx.array:
            q_all = model(x_)  # (batch, n_actions)
            # Gather Q-values for the selected actions.
            batch_size = q_all.shape[0]
            row_idx = mx.arange(batch_size)
            # MLX uses fancy indexing: q_all[row_idx, a_]
            q_selected = q_all[row_idx, a_]  # (batch,)
            loss = nn.losses.huber_loss(q_selected, t_, reduction="mean")
            return loss

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss_val, grads = loss_and_grad_fn(self.model, x, a, targets)
        grads, _ = optim.clip_grad_norm(grads, max_norm=self.config.max_grad_norm)
        self.optimizer.update(self.model, grads)

        # Mean Q for diagnostics (reuse forward pass).
        q_all = self.model(x)
        row_idx = mx.arange(q_all.shape[0])
        mean_q = mx.mean(q_all[row_idx, a])

        mx.eval(loss_val, mean_q, self.model.parameters())

        return {
            "loss": float(loss_val),
            "mean_q": float(mean_q),
        }

    def train_epoch(self, _epoch: int) -> dict[str, float]:
        """Run ``config.steps_per_epoch`` environment steps and gradient updates.

        Each step:
        1. Takes an epsilon-greedy action in the environment.
        2. Stores the transition in the replay buffer.
        3. If the buffer is ready, samples a mini-batch and calls
           :meth:`train_step`.
        4. Updates the target network every ``target_update_freq`` steps.

        Args:
            epoch: Current epoch number (used for logging only).

        Returns:
            Dict with epoch-averaged ``'loss'``, ``'mean_q'``, and
            ``'epsilon'``.
        """
        legal_moves = self.env.legal_moves()
        n_actions = len(legal_moves)

        current_puzzle = self.env.reset()

        epoch_losses: list[float] = []
        epoch_qs: list[float] = []

        for _ in range(self.config.steps_per_epoch):
            # --- Epsilon-greedy action selection. ---
            encoded = self.encoder.encode(current_puzzle)
            eps = self.epsilon

            if self._rng.random() < eps:
                action_idx = int(self._rng.integers(n_actions))
            else:
                x = mx.array(encoded[np.newaxis, :])
                q_vals = self.model(x)
                mx.eval(q_vals)
                action_idx = int(np.argmax(np.array(q_vals).reshape(-1)))

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

            # --- Gradient update (if buffer is warm). ---
            if self.replay_buffer.is_ready(self.config.min_replay_size):
                batch = self.replay_buffer.sample(self.config.batch_size, self._rng)
                step_metrics = self.train_step(batch)
                epoch_losses.append(step_metrics["loss"])
                epoch_qs.append(step_metrics["mean_q"])

            # --- Target network sync. ---
            if self._global_step % self.config.target_update_freq == 0:
                self._sync_target()
                self._logger.debug(
                    "Step %d: target network synced.", self._global_step
                )

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        avg_q = float(np.mean(epoch_qs)) if epoch_qs else float("nan")

        return {
            "loss": avg_loss,
            "mean_q": avg_q,
            "epsilon": self.epsilon,
        }

    def train(self) -> list[dict[str, float]]:
        """Run the complete DQN training loop.

        Returns:
            List of per-epoch metric dicts.
        """
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []

        if _tqdm is not None:
            epoch_bar = _tqdm(range(1, self.config.epochs + 1), desc="DQN", unit="epoch", dynamic_ncols=True)
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
                    epoch,
                    self.config.epochs,
                    metrics["loss"],
                    metrics["mean_q"],
                    metrics["epsilon"],
                )
                ckpt_path = (
                    self.config.checkpoint_dir
                    / f"ckpt_step_{self._global_step:07d}.npz"
                )
                self.save_checkpoint(ckpt_path, self._global_step)

        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, step: int) -> None:
        """Persist online-network weights to a ``.npz`` file.

        Args:
            path: Destination file path.
            step: Global environment step count stored as metadata.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        flat: dict[str, mx.array] = {}
        _flatten_params(dict(self.model.parameters()), "", flat)
        flat["__step__"] = mx.array(step)
        mx.savez(str(path), **flat)
        self._logger.debug("Checkpoint saved → %s (step %d)", path, step)

    def load_checkpoint(self, path: Path) -> int:
        """Load online-network weights from a ``.npz`` checkpoint.

        After loading, the target network is synced to match.

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            The global step count stored in the checkpoint.
        """
        weights = mx.load(str(path))
        step = int(weights.pop("__step__"))
        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())
        self._sync_target()
        self._global_step = step
        self._logger.debug("Checkpoint loaded ← %s (step %d)", path, step)
        return step

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """Hard-copy online network weights to the target network."""
        # Use load_weights to copy parameters without graph linkage.
        params = dict(self.model.parameters())
        flat: dict[str, mx.array] = {}
        _flatten_params(params, "", flat)
        self.target_model.load_weights(list(flat.items()))
        mx.eval(self.target_model.parameters())


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _flatten_params(
    params: dict,
    prefix: str,
    out: dict[str, mx.array],
) -> None:
    """Recursively flatten a nested MLX parameter dict into *out*."""
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
