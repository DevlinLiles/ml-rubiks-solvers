"""Autodidactic Iteration (ADI) trainer for CubeValueNet.

ADI bootstraps better distance labels by using the current CNN beam-search
solver to solve scrambled puzzles.  Every state along a *solved* solution path
has an exact known distance-to-solved (``solution_length - step_index``), which
is a much more accurate training signal than the raw scramble depth.

Training procedure (one iteration):
1. Solve *n_solve* scrambled puzzles with beam search.
2. For each solved puzzle, extract ``(encoded_state, true_distance)`` pairs
   from every step in the solution path.
3. Mix with ``n_fallback`` scramble-depth pairs to prevent catastrophic
   forgetting of states the solver hasn't seen yet.
4. Retrain the shared ``CubeValueNet`` for ``epochs_per_iter`` epochs.

Successive iterations increase ``max_scramble`` so the solver bootstraps its
way to harder puzzles as accuracy improves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.solvers.cnn.model import CubeValueNet
from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig
from rubiks_solve.solvers.cnn.trainer import CNNTrainer, CNNTrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class ADIConfig:
    """Configuration for :class:`ADITrainer`.

    Attributes:
        n_iterations:          Number of ADI bootstrap rounds.
        n_solve_per_iter:      Puzzles to attempt with beam search each round.
        n_fallback_per_iter:   Scramble-depth fallback samples each round
                               (prevents catastrophic forgetting).
        start_max_scramble:    Scramble depth ceiling in the first iteration.
        scramble_depth_increment: Increase ceiling by this many moves each iter.
        epochs_per_iter:       Training epochs per ADI round.
        batch_size:            Mini-batch size for training.
        learning_rate:         Adam learning rate.
        beam_width:            Beam width for the internal CNN solver.
        beam_max_depth:        Max beam-search depth for the internal solver.
        checkpoint_dir:        Where to save per-iteration ``.npz`` checkpoints.
        log_interval:          Log training metrics every N epochs.
    """

    n_iterations: int = 5
    n_solve_per_iter: int = 500
    n_fallback_per_iter: int = 10_000
    start_max_scramble: int = 11
    scramble_depth_increment: int = 2
    epochs_per_iter: int = 150
    batch_size: int = 2048
    learning_rate: float = 1e-4
    beam_width: int = 1024
    beam_max_depth: int = 200
    checkpoint_dir: Path = Path("models/cnn_adi/3x3")
    log_interval: int = 50


class ADITrainer:
    """Trains :class:`CubeValueNet` with Autodidactic Iteration.

    Args:
        puzzle_cls: Concrete puzzle class (e.g. ``Cube3x3``).
        encoder:    State encoder shared by solver and trainer.
        model:      Value network to be trained in-place.
        config:     ADI hyper-parameters.
        rng:        NumPy random generator (seeded for reproducibility).
        seed:       Seed forwarded to :class:`CNNTrainer` for data shuffling.
    """

    def __init__(
        self,
        puzzle_cls: type[AbstractPuzzle],
        encoder: AbstractStateEncoder,
        model: CubeValueNet,
        config: ADIConfig,
        rng: np.random.Generator,
    ) -> None:
        self.puzzle_cls = puzzle_cls
        self.encoder = encoder
        self.model = model
        self.config = config
        self.rng = rng
        self._logger = logging.getLogger(self.__class__.__qualname__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """Execute all ADI iterations.

        Returns:
            List of per-iteration summary dicts with keys
            ``iteration``, ``n_solved``, ``n_adi_samples``,
            ``final_loss``, ``final_mae``.
        """
        cfg = self.config
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        iteration_summaries: list[dict] = []
        max_scramble = cfg.start_max_scramble

        for iteration in range(1, cfg.n_iterations + 1):
            self._logger.info(
                "ADI iteration %d/%d  max_scramble=%d",
                iteration, cfg.n_iterations, max_scramble,
            )

            # ---- Phase 1: solve puzzles, collect exact-distance data ----
            adi_states, adi_distances, n_solved = self._collect_adi_data(
                max_scramble=max_scramble,
            )
            self._logger.info(
                "  Solved %d/%d puzzles → %d ADI samples",
                n_solved, cfg.n_solve_per_iter, len(adi_states),
            )

            # ---- Phase 2: scramble-depth fallback data ------------------
            fb_states, fb_distances = self._scramble_fallback(
                n_samples=cfg.n_fallback_per_iter,
                max_depth=max_scramble,
            )

            # ---- Combine and shuffle ------------------------------------
            states = np.concatenate([adi_states, fb_states], axis=0) if len(adi_states) else fb_states
            labels = np.concatenate([adi_distances, fb_distances], axis=0) if len(adi_distances) else fb_distances

            idx = self.rng.permutation(len(states))
            states, labels = states[idx], labels[idx]

            # ---- Phase 3: retrain ---------------------------------------
            trainer_cfg = CNNTrainerConfig(
                epochs=cfg.epochs_per_iter,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                checkpoint_dir=cfg.checkpoint_dir,
                log_interval=cfg.log_interval,
            )
            trainer = CNNTrainer(self.model, trainer_cfg)
            history = trainer.train(states, labels)

            final = history[-1]
            self._logger.info(
                "  Epoch %d done | loss=%.4f | mae=%.4f",
                cfg.epochs_per_iter, final["loss"], final["mae"],
            )

            iteration_summaries.append({
                "iteration": iteration,
                "max_scramble": max_scramble,
                "n_solved": n_solved,
                "n_adi_samples": len(adi_states),
                "final_loss": final["loss"],
                "final_mae": final["mae"],
            })

            max_scramble += cfg.scramble_depth_increment

        return iteration_summaries

    # ------------------------------------------------------------------
    # Data generation helpers
    # ------------------------------------------------------------------

    def _collect_adi_data(
        self,
        max_scramble: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Solve puzzles with beam search and extract (state, distance) pairs.

        Only uses puzzles that are *successfully solved* — partial paths would
        introduce noisy distance labels.

        Returns:
            ``(states, distances, n_solved)`` where ``states`` is float32
            ``(N, input_size)`` and ``distances`` is float32 ``(N,)``.
        """
        cfg = self.config
        cnn_cfg = CNNConfig(
            beam_width=cfg.beam_width,
            max_depth=cfg.beam_max_depth,
        )
        solver = CNNSolver(self.puzzle_cls, self.encoder, cnn_cfg)
        # Share weights with the model being trained.
        solver.model = self.model

        states_list: list[np.ndarray] = []
        distances_list: list[float] = []
        n_solved = 0

        for _ in range(cfg.n_solve_per_iter):
            depth = int(self.rng.integers(1, max_scramble + 1))
            puzzle = self.puzzle_cls.solved_state().scramble(depth, self.rng)

            result = solver.solve(puzzle)
            if not result.solved:
                continue

            n_solved += 1
            current = puzzle
            n_moves = len(result.moves)

            for step_idx, move in enumerate(result.moves):
                true_dist = float(n_moves - step_idx)
                states_list.append(self.encoder.encode(current))
                distances_list.append(true_dist)
                current = current.apply_move(move)

            # Final solved state has distance 0.
            states_list.append(self.encoder.encode(current))
            distances_list.append(0.0)

        if not states_list:
            empty = np.empty((0, self.encoder.output_size), dtype=np.float32)
            return empty, np.empty((0,), dtype=np.float32), 0

        return (
            np.stack(states_list, axis=0).astype(np.float32),
            np.array(distances_list, dtype=np.float32),
            n_solved,
        )

    def _scramble_fallback(
        self,
        n_samples: int,
        max_depth: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate classic scramble-depth ``(state, depth)`` pairs.

        These prevent the model from forgetting states it doesn't currently
        solve, maintaining a reasonable prior across the full depth range.
        """
        solved = self.puzzle_cls.solved_state()
        states_list: list[np.ndarray] = []
        depths_list: list[float] = []

        for _ in range(n_samples):
            depth = int(self.rng.integers(1, max_depth + 1))
            puzzle = solved.scramble(depth, self.rng)
            states_list.append(self.encoder.encode(puzzle))
            depths_list.append(float(depth))

        return (
            np.stack(states_list, axis=0).astype(np.float32),
            np.array(depths_list, dtype=np.float32),
        )
