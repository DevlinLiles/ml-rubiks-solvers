"""CNN-based solver: beam search guided by a trained value network."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import mlx.core as mx

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.solvers.cnn.model import CubeValueNet


@dataclass
class CNNConfig:
    """Configuration for :class:`CNNSolver`.

    Attributes:
        model_path:  Optional path to a pre-trained ``.npz`` checkpoint.
                     When provided the weights are loaded in ``__init__``.
        hidden_dims: Hidden layer widths used when constructing the value net.
        beam_width:  Number of states kept alive at each beam-search step.
        max_depth:   Maximum number of moves before declaring failure.
                     Defaults to 30; callers may scale this with
                     ``puzzle_type.move_limit()``.
    """

    model_path: Path | None = None
    hidden_dims: list[int] = field(default_factory=lambda: [4096, 2048, 512])
    beam_width: int = 128
    max_depth: int = 30


@dataclass(order=True)
class _BeamNode:
    """A single node in the beam, ordered by estimated value (ascending = better)."""

    value: float
    puzzle: Any = field(compare=False)
    moves: list[Move] = field(compare=False)


class CNNSolver(AbstractSolver):
    """Solve a puzzle via beam search guided by a :class:`CubeValueNet`.

    At each depth the solver:
    1. Expands every state in the current beam by applying all legal moves.
    2. Batches the resulting states through the value network.
    3. Keeps the ``beam_width`` states with the lowest estimated value.
    4. Terminates early if any state is solved.

    Args:
        puzzle_type: Concrete puzzle class (used to retrieve legal moves).
        encoder:     Encodes puzzle states to float32 arrays.
        config:      Beam-search and model hyper-parameters.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        encoder: AbstractStateEncoder,
        config: CNNConfig | None = None,
    ) -> None:
        if config is None:
            config = CNNConfig()
        super().__init__(puzzle_type, config)
        self.encoder = encoder
        self.model = CubeValueNet(
            input_size=encoder.output_size,
            hidden_dims=config.hidden_dims,
        )
        # Evaluate model parameters so MLX allocates buffers up-front.
        mx.eval(self.model.parameters())

        if config.model_path is not None:
            self.load_model(config.model_path)

    # ------------------------------------------------------------------
    # AbstractSolver interface
    # ------------------------------------------------------------------

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Attempt to solve *puzzle* using beam search + value network.

        Args:
            puzzle: The scrambled puzzle instance to solve.

        Returns:
            :class:`~rubiks_solve.solvers.base.SolveResult` with:
            - ``solved``: True if a solution was found within ``max_depth``.
            - ``moves``: The solution move sequence.
            - ``metadata``: Contains ``beam_width`` and ``nodes_expanded``.
        """
        config: CNNConfig = self.config
        legal_moves = puzzle.legal_moves()
        start_time = time.perf_counter()

        # Short-circuit: already solved.
        if puzzle.is_solved:
            return SolveResult(
                solved=True,
                moves=[],
                solve_time_seconds=0.0,
                iterations=0,
                metadata={"beam_width": config.beam_width, "nodes_expanded": 0},
            )

        beam: list[_BeamNode] = [_BeamNode(value=0.0, puzzle=puzzle, moves=[])]
        nodes_expanded = 0

        for depth in range(1, config.max_depth + 1):
            # Expand all states in the current beam.
            candidates: list[tuple[AbstractPuzzle, list[Move]]] = []
            for node in beam:
                for move in legal_moves:
                    next_puzzle = node.puzzle.apply_move(move)
                    next_moves = node.moves + [move]
                    candidates.append((next_puzzle, next_moves))
                nodes_expanded += len(legal_moves)

            # Batch encode candidates.
            puzzles_batch = [c[0] for c in candidates]
            encoded = self.encoder.encode_batch(puzzles_batch)  # (N, input_size)
            x = mx.array(encoded)

            # Forward pass through value network.
            self.model.eval()
            values = self.model(x)  # (N, 1)
            mx.eval(values)
            values_np = np.array(values).reshape(-1)  # (N,)

            # Check for solved states — pick lowest-move-count solution.
            solved_solution: list[Move] | None = None
            for idx, (cand_puzzle, cand_moves) in enumerate(candidates):
                if cand_puzzle.is_solved:
                    if solved_solution is None or len(cand_moves) < len(solved_solution):
                        solved_solution = cand_moves

            if solved_solution is not None:
                elapsed = time.perf_counter() - start_time
                return SolveResult(
                    solved=True,
                    moves=solved_solution,
                    solve_time_seconds=elapsed,
                    iterations=depth,
                    metadata={
                        "beam_width": config.beam_width,
                        "nodes_expanded": nodes_expanded,
                    },
                )

            # Rank candidates by value (lower = better) and keep top beam_width.
            ranked_indices = np.argsort(values_np)[: config.beam_width]
            beam = [
                _BeamNode(
                    value=float(values_np[i]),
                    puzzle=candidates[i][0],
                    moves=candidates[i][1],
                )
                for i in ranked_indices
            ]

        elapsed = time.perf_counter() - start_time
        return SolveResult(
            solved=False,
            moves=[],
            solve_time_seconds=elapsed,
            iterations=config.max_depth,
            metadata={
                "beam_width": config.beam_width,
                "nodes_expanded": nodes_expanded,
            },
        )

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_model(self, path: Path) -> None:
        """Load value-network weights from a ``.npz`` checkpoint.

        The checkpoint format is produced by :meth:`CNNTrainer.save_checkpoint`.
        The ``__epoch__`` metadata key is silently ignored.

        Args:
            path: Path to the ``.npz`` file.
        """
        weights = mx.load(str(path))
        weights.pop("__epoch__", None)
        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())
        self._logger.info("Loaded CNN model weights from %s", path)
