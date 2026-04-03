"""DQN solver: greedy max-Q policy rollout using a trained DuelingDQN."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mlx.core as mx

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.solvers.dqn.model import DuelingDQN


@dataclass
class DQNConfig:
    """Configuration for :class:`DQNSolver`.

    Attributes:
        model_path: Optional path to a pre-trained ``.npz`` checkpoint.
        max_steps:  Maximum number of moves before declaring failure.
                    Defaults to 200; callers may scale with
                    ``puzzle_type.move_limit()``.
    """

    model_path: Path | None = None
    max_steps: int = 200


class DQNSolver(AbstractSolver):
    """Follows the greedy max-Q policy of a trained :class:`DuelingDQN`.

    At each step the encoded state is passed through the Q-network and the
    action with the highest Q-value is selected.  The process continues until
    the puzzle is solved or ``max_steps`` is exhausted.

    Args:
        puzzle_type: Concrete puzzle class.
        encoder:     State encoder.
        config:      Solver configuration.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        encoder: AbstractStateEncoder,
        config: DQNConfig | None = None,
    ) -> None:
        if config is None:
            config = DQNConfig()
        super().__init__(puzzle_type, config)
        self.encoder = encoder
        self.legal_moves: list[Move] = puzzle_type.solved_state().legal_moves()
        n_actions = len(self.legal_moves)

        self.model = DuelingDQN(
            input_size=encoder.output_size,
            n_actions=n_actions,
        )
        mx.eval(self.model.parameters())

        if config.model_path is not None:
            self.load_model(config.model_path)

    # ------------------------------------------------------------------
    # AbstractSolver interface
    # ------------------------------------------------------------------

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Apply the greedy max-Q policy until solved or the step limit.

        Args:
            puzzle: The scrambled puzzle to solve.

        Returns:
            :class:`~rubiks_solve.solvers.base.SolveResult` with:
            - ``solved``: True if the puzzle was solved.
            - ``moves``: Sequence of moves applied.
            - ``metadata``: Contains ``'q_values'`` — a list of lists,
              one entry per step taken, each being the full Q-value vector
              over all actions at that state.
        """
        config: DQNConfig = self.config
        start_time = time.perf_counter()

        if puzzle.is_solved:
            return SolveResult(
                solved=True,
                moves=[],
                solve_time_seconds=0.0,
                iterations=0,
                metadata={"q_values": []},
            )

        current = puzzle
        moves_taken: list[Move] = []
        q_values_history: list[list[float]] = []

        self.model.eval()

        for step in range(config.max_steps):
            encoded = self.encoder.encode(current)  # (input_size,) float32
            x = mx.array(encoded[np.newaxis, :])    # (1, input_size)

            q_vals = self.model(x)                   # (1, n_actions)
            mx.eval(q_vals)

            q_np = np.array(q_vals).reshape(-1)      # (n_actions,)
            q_values_history.append(q_np.tolist())

            action_idx = int(np.argmax(q_np))
            move = self.legal_moves[action_idx]
            current = current.apply_move(move)
            moves_taken.append(move)

            if current.is_solved:
                elapsed = time.perf_counter() - start_time
                return SolveResult(
                    solved=True,
                    moves=moves_taken,
                    solve_time_seconds=elapsed,
                    iterations=step + 1,
                    metadata={"q_values": q_values_history},
                )

        elapsed = time.perf_counter() - start_time
        return SolveResult(
            solved=False,
            moves=moves_taken,
            solve_time_seconds=elapsed,
            iterations=config.max_steps,
            metadata={"q_values": q_values_history},
        )

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_model(self, path: Path) -> None:
        """Load DQN weights from a ``.npz`` checkpoint.

        Args:
            path: Path to the ``.npz`` file produced by
                  :meth:`DQNTrainer.save_checkpoint`.
        """
        weights = mx.load(str(path))
        weights.pop("__step__", None)
        weights.pop("__epoch__", None)
        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())
        self._logger.info("Loaded DQN model weights from %s", path)
