"""IDA* solver using a CNN value network as the search heuristic.

The heuristic is the CNN's estimated distance-to-solved, optionally scaled by
``heuristic_weight`` to bias toward admissibility (h never overestimates the
true cost).  When ``heuristic_weight < 1`` the solver is suboptimal but
complete; when ``heuristic_weight == 1`` it is optimal only if the network
never overestimates.

Efficiency trick: all children of a node are encoded and evaluated in a single
MLX batch, avoiding per-state Python/Metal overhead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.solvers.cnn.model import CubeValueNet


@dataclass
class IDAStarConfig:
    """Configuration for :class:`IDAStarSolver`.

    Attributes:
        model_path:        Path to a ``.npz`` checkpoint produced by
                           :class:`~rubiks_solve.solvers.cnn.trainer.CNNTrainer`
                           or :class:`~rubiks_solve.solvers.cnn.adi_trainer.ADITrainer`.
        hidden_dims:       Hidden-layer widths of the value network.
        max_depth:         Maximum search depth (IDA* bound ceiling).
        heuristic_weight:  Multiply CNN output by this before using as h(n).
                           Values < 1.0 make the heuristic more conservative
                           (fewer nodes pruned, but more likely admissible).
        time_limit_seconds: Hard wall-clock limit; returns best-effort result.
    """

    model_path: Path | None = None
    hidden_dims: list[int] = field(default_factory=lambda: [4096, 2048, 512])
    max_depth: int = 30
    heuristic_weight: float = 0.85
    time_limit_seconds: float = 60.0


class IDAStarSolver(AbstractSolver):
    """Iterative-Deepening A* guided by a trained :class:`CubeValueNet`.

    At each IDA* threshold, depth-first search expands nodes whose estimated
    total cost ``f = g + h`` does not exceed the current bound.  All children
    of a node are encoded and passed through the value network in a single
    batch before branching, which amortises the MLX dispatch overhead.

    Args:
        puzzle_type: Concrete puzzle class.
        encoder:     State encoder.
        config:      Solver hyper-parameters.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        encoder: AbstractStateEncoder,
        config: IDAStarConfig | None = None,
    ) -> None:
        if config is None:
            config = IDAStarConfig()
        super().__init__(puzzle_type, config)
        self.encoder = encoder
        self.legal_moves: list[Move] = puzzle_type.solved_state().legal_moves()

        self.model = CubeValueNet(
            input_size=encoder.output_size,
            hidden_dims=config.hidden_dims,
        )
        mx.eval(self.model.parameters())

        if config.model_path is not None:
            self._load_model(config.model_path)

    # ------------------------------------------------------------------
    # AbstractSolver interface
    # ------------------------------------------------------------------

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Run IDA* from *puzzle* until solved, depth exceeded, or time out.

        Args:
            puzzle: Scrambled puzzle to solve.

        Returns:
            :class:`~rubiks_solve.solvers.base.SolveResult` where
            ``metadata`` contains ``nodes_expanded`` and ``iterations``
            (number of threshold deepening steps).
        """
        config: IDAStarConfig = self.config
        start = time.perf_counter()

        if puzzle.is_solved:
            return SolveResult(
                solved=True, moves=[], solve_time_seconds=0.0,
                iterations=0, metadata={"nodes_expanded": 0},
            )

        self.model.eval()
        bound = self._h(puzzle)
        total_nodes = 0
        iteration = 0
        best_partial: list[Move] = []

        while bound <= config.max_depth:
            if time.perf_counter() - start > config.time_limit_seconds:
                break

            path: list[Move] = []
            t, solution, n_exp = self._dfs(
                puzzle, 0, bound, path, None, start, []
            )
            total_nodes += n_exp
            iteration += 1

            if solution is not None:
                elapsed = time.perf_counter() - start
                return SolveResult(
                    solved=True,
                    moves=solution,
                    solve_time_seconds=elapsed,
                    iterations=iteration,
                    metadata={"nodes_expanded": total_nodes},
                )

            if t == float("inf"):
                break  # exhausted

            bound = t
            # Keep track of the deepest partial path explored so far.
            if path:
                best_partial = list(path)

        elapsed = time.perf_counter() - start
        return SolveResult(
            solved=False,
            moves=best_partial,
            solve_time_seconds=elapsed,
            iterations=iteration,
            metadata={"nodes_expanded": total_nodes},
        )

    # ------------------------------------------------------------------
    # IDA* internals
    # ------------------------------------------------------------------

    def _dfs(
        self,
        puzzle: AbstractPuzzle,
        g: int,
        bound: float,
        path: list[Move],
        last_move: Optional[Move],
        start_time: float,
        _: list,  # unused, kept for signature symmetry
    ) -> tuple[float, Optional[list[Move]], int]:
        """Recursive DFS step.

        Returns ``(next_bound, solution_or_None, nodes_expanded)``.
        ``next_bound`` is the minimum f-value that exceeded the current bound
        (used to set the next IDA* threshold).  ``-1`` signals success.
        """
        config: IDAStarConfig = self.config
        h = self._h(puzzle)
        f = g + h

        if f > bound:
            return f, None, 0
        if puzzle.is_solved:
            return -1.0, list(path), 0
        if time.perf_counter() - start_time > config.time_limit_seconds:
            return float("inf"), None, 0

        # --- Prune legal moves ----------------------------------------
        candidates: list[tuple[Move, AbstractPuzzle]] = []
        inv_name = last_move.inverse().name if last_move is not None else None

        for move in self.legal_moves:
            # Skip immediate reversal.
            if move.name == inv_name:
                continue
            # Skip same-face as last move (avoids R R, R R', R R2 redundancy).
            if last_move is not None and move.face == last_move.face:
                continue
            candidates.append((move, puzzle.apply_move(move)))

        if not candidates:
            return float("inf"), None, 0

        # --- Batch heuristic for all children -------------------------
        child_states = np.stack(
            [self.encoder.encode(p) for _, p in candidates], axis=0
        ).astype(np.float32)
        x = mx.array(child_states)
        vals = self.model(x)          # (N, 1)
        mx.eval(vals)
        h_children = np.array(vals).reshape(-1) * config.heuristic_weight
        h_children = np.clip(h_children, 0.0, None)

        # Sort by f = (g+1) + h to explore most-promising children first.
        order = np.argsort(h_children)

        next_bound = float("inf")
        nodes_expanded = len(candidates)

        for idx in order:
            move, child = candidates[idx]
            child_f = (g + 1) + float(h_children[idx])

            if child_f > bound:
                # All remaining candidates have f >= this (sorted), so prune.
                next_bound = min(next_bound, child_f)
                break

            path.append(move)
            t, solution, n_exp = self._dfs(
                child, g + 1, bound, path, move, start_time, []
            )
            nodes_expanded += n_exp

            if solution is not None:
                return -1.0, solution, nodes_expanded

            next_bound = min(next_bound, t)

            path.pop()

        return next_bound, None, nodes_expanded

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    def _h(self, puzzle: AbstractPuzzle) -> float:
        """Evaluate h(puzzle) = CNN(puzzle) * heuristic_weight, clamped ≥ 0."""
        encoded = self.encoder.encode(puzzle)
        x = mx.array(encoded[np.newaxis, :])
        val = self.model(x)
        mx.eval(val)
        raw = float(np.array(val).reshape(-1)[0])
        return max(0.0, raw * self.config.heuristic_weight)

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def _load_model(self, path: Path) -> None:
        weights = mx.load(str(path))
        weights.pop("__epoch__", None)
        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())
        self._logger.info("IDA* heuristic loaded from %s", path)
