"""Policy-network solver: greedy or temperature-sampled rollout."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mlx.core as mx

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.solvers.policy.model import CubePolicyNet


@dataclass
class PolicyConfig:
    """Configuration for :class:`PolicyNetworkSolver`.

    Attributes:
        model_path:   Optional path to a pre-trained ``.npz`` checkpoint.
        hidden_dims:  Hidden layer widths for the policy network.
        temperature:  Softmax temperature applied before sampling.
                      ``temperature=1.0`` is unmodified; lower values make
                      the policy more greedy, higher values more uniform.
                      Only used when ``deterministic=False``.
        deterministic: If ``True``, always choose ``argmax`` over action
                       log-probabilities (fastest, no randomness).
    """

    model_path: Path | None = None
    hidden_dims: list[int] = field(default_factory=lambda: [4096, 2048])
    temperature: float = 1.0
    deterministic: bool = True


class PolicyNetworkSolver(AbstractSolver):
    """Greedily follows a trained :class:`CubePolicyNet` until solved or depth limit.

    At each step the current state is encoded, passed through the network, and
    the highest-probability (or temperature-sampled) move is applied.  The
    process repeats until ``puzzle.is_solved`` or ``puzzle_type.move_limit()``
    steps are exhausted.

    Args:
        puzzle_type: Concrete puzzle class.
        encoder:     State-to-array encoder.
        config:      Solver hyper-parameters.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        encoder: AbstractStateEncoder,
        config: PolicyConfig | None = None,
    ) -> None:
        if config is None:
            config = PolicyConfig()
        super().__init__(puzzle_type, config)
        self.encoder = encoder
        legal_moves = puzzle_type.solved_state().legal_moves()
        self.legal_moves: list[Move] = legal_moves
        self.n_actions = len(legal_moves)

        self.model = CubePolicyNet(
            input_size=encoder.output_size,
            n_actions=self.n_actions,
            hidden_dims=config.hidden_dims,
        )
        mx.eval(self.model.parameters())

        if config.model_path is not None:
            self.load_model(config.model_path)

    # ------------------------------------------------------------------
    # AbstractSolver interface
    # ------------------------------------------------------------------

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Apply the policy greedily until the puzzle is solved or the move budget runs out.

        Args:
            puzzle: The scrambled puzzle instance.

        Returns:
            :class:`~rubiks_solve.solvers.base.SolveResult` with:
            - ``solved``: True if solved within the move limit.
            - ``moves``: Sequence of moves applied.
            - ``metadata``: Contains ``'move_probabilities'`` — the
              probability distribution over moves for the **last** step
              taken, as a list of floats indexed by move index.
        """
        config: PolicyConfig = self.config
        max_steps = self.puzzle_type.move_limit()
        rng = np.random.default_rng()

        start_time = time.perf_counter()

        if puzzle.is_solved:
            return SolveResult(
                solved=True,
                moves=[],
                solve_time_seconds=0.0,
                iterations=0,
                metadata={"move_probabilities": []},
            )

        current = puzzle
        moves_taken: list[Move] = []
        last_probs: list[float] = []

        self.model.eval()

        for step in range(max_steps):
            encoded = self.encoder.encode(current)  # (input_size,) float32
            x = mx.array(encoded[np.newaxis, :])    # (1, input_size)

            log_probs = self.model(x)                # (1, n_actions)
            mx.eval(log_probs)

            log_probs_np = np.array(log_probs).reshape(-1)  # (n_actions,)

            if config.deterministic:
                action_idx = int(np.argmax(log_probs_np))
            else:
                # Apply temperature scaling in log-space, then softmax to sample.
                scaled_logits = log_probs_np / max(config.temperature, 1e-8)
                # Numerically stable softmax.
                shifted = scaled_logits - scaled_logits.max()
                probs = np.exp(shifted)
                probs /= probs.sum()
                action_idx = int(rng.choice(self.n_actions, p=probs))

            # Record probabilities (in probability space) for metadata.
            log_probs_shifted = log_probs_np - log_probs_np.max()
            probs_full = np.exp(log_probs_shifted)
            probs_full /= probs_full.sum()
            last_probs = probs_full.tolist()

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
                    metadata={"move_probabilities": last_probs},
                )

        elapsed = time.perf_counter() - start_time
        return SolveResult(
            solved=False,
            moves=moves_taken,
            solve_time_seconds=elapsed,
            iterations=max_steps,
            metadata={"move_probabilities": last_probs},
        )

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_model(self, path: Path) -> None:
        """Load policy-network weights from a ``.npz`` checkpoint.

        Args:
            path: Path to the ``.npz`` file produced by
                  :meth:`PolicyTrainer.save_checkpoint`.
        """
        weights = mx.load(str(path))
        weights.pop("__epoch__", None)
        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())
        self._logger.info("Loaded policy model weights from %s", path)
