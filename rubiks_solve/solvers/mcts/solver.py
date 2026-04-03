"""Monte Carlo Tree Search solver."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.solvers.base import AbstractSolver, SolveResult
from rubiks_solve.solvers.mcts.node import MCTSNode


@dataclass
class MCTSConfig:
    """Hyper-parameters for :class:`MCTSSolver`.

    Attributes:
        n_simulations: Maximum number of MCTS simulations (selection +
            expansion + rollout + backpropagation cycles) to run.
        max_rollout_depth: Maximum number of random moves per rollout.
        c_puct: Exploration constant used in the UCB1 formula.  Higher values
            increase exploration of less-visited nodes.  Defaults to
            ``sqrt(2) ≈ 1.414``.
        time_limit_seconds: Wall-clock time budget.  The solver stops when
            either *n_simulations* or *time_limit_seconds* is exceeded,
            whichever comes first.
        seed: Random seed passed to ``numpy.random.default_rng`` for
            reproducible rollouts.
    """

    n_simulations: int = 10_000
    max_rollout_depth: int = 50
    c_puct: float = 1.414
    time_limit_seconds: float = 10.0
    seed: int = 42


class MCTSSolver(AbstractSolver):
    """Solves puzzles using Monte Carlo Tree Search (MCTS).

    The algorithm alternates between four phases in each simulation:

    1. **Selection** — descend the tree using UCB1 until a node is found that
       is not fully expanded or is terminal.
    2. **Expansion** — add one new child to the selected node (if not
       terminal).
    3. **Rollout** — simulate a random play-out from the new child up to
       ``config.max_rollout_depth`` moves.
    4. **Backpropagation** — propagate the rollout reward up to the root.

    The solver terminates early when any node in the tree is in the solved
    state.  If the budget is exhausted without finding a solution, the path to
    the most-visited leaf is returned with ``solved=False``.

    Metadata keys in the returned :class:`~rubiks_solve.solvers.base.SolveResult`:
        ``simulations_run``: number of complete MCTS simulations performed.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        config: MCTSConfig = MCTSConfig(),
    ) -> None:
        """Initialise the MCTS solver.

        Args:
            puzzle_type: The concrete puzzle class to solve.
            config: Algorithm hyper-parameters.  Defaults to
                :class:`MCTSConfig` with sensible values.
        """
        super().__init__(puzzle_type, config)

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Run MCTS to solve *puzzle*.

        Args:
            puzzle: The scrambled puzzle instance to solve.

        Returns:
            A :class:`~rubiks_solve.solvers.base.SolveResult` with the best
            move sequence found, whether it constitutes a full solution, the
            number of simulations run, and a ``simulations_run`` metadata key.
        """
        cfg: MCTSConfig = self.config
        rng = np.random.default_rng(cfg.seed)
        start_time = time.perf_counter()

        # Trivial case: already solved
        if puzzle.is_solved:
            return SolveResult(
                solved=True,
                moves=[],
                solve_time_seconds=time.perf_counter() - start_time,
                iterations=0,
                metadata={"simulations_run": 0},
            )

        root = MCTSNode(puzzle=puzzle)
        best_solved_node: MCTSNode | None = None
        simulations_run = 0

        for _ in range(cfg.n_simulations):
            elapsed = time.perf_counter() - start_time
            if elapsed >= cfg.time_limit_seconds:
                break

            # --- Selection ---
            node = self._select(root, cfg.c_puct)

            # --- Expansion ---
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # --- Check for solution after expansion ---
            if node.is_terminal():
                best_solved_node = node
                simulations_run += 1
                # Still backpropagate the win
                node.backpropagate(1.0)
                break

            # --- Rollout ---
            reward = node.rollout(cfg.max_rollout_depth, rng)

            # --- Backpropagation ---
            node.backpropagate(reward)
            simulations_run += 1

            # Check if any child of the newly expanded node is solved
            if reward == 1.0:
                # The rollout reached a solved state — but the node itself may
                # not be solved; check the node directly.
                if node.is_terminal():
                    best_solved_node = node
                    break

        elapsed = time.perf_counter() - start_time

        if best_solved_node is not None:
            solution_moves = best_solved_node.solution_path()
            return SolveResult(
                solved=True,
                moves=solution_moves,
                solve_time_seconds=elapsed,
                iterations=simulations_run,
                metadata={"simulations_run": simulations_run},
            )

        # Return the path to the most promising node found
        best_moves = self._best_path(root, cfg.c_puct)
        return SolveResult(
            solved=False,
            moves=best_moves,
            solve_time_seconds=elapsed,
            iterations=simulations_run,
            metadata={"simulations_run": simulations_run},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode, c: float) -> MCTSNode:
        """Descend the tree using UCB1 until a non-fully-expanded or terminal node.

        Args:
            node: Starting node (typically the root).
            c: Exploration constant for UCB1.

        Returns:
            The selected leaf node.
        """
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(c)
        return node

    def _best_path(self, root: MCTSNode, _c: float) -> list[Move]:
        """Return the greedy move sequence from *root* following highest-visit children.

        Used when no solution was found to return the best partial path.

        Args:
            root: The root node of the search tree.
            c: Exploration constant (passed to :meth:`~MCTSNode.best_child` with
                ``c=0`` to select by exploitation only).

        Returns:
            List of moves along the most-visited path from the root.
        """
        moves: list[Move] = []
        node = root
        visited: set[int] = set()

        while node.children:
            node_id = id(node)
            if node_id in visited:
                break
            visited.add(node_id)
            # Use c=0 to select purely by exploitation (highest average reward)
            best = max(node.children, key=lambda n: n.visits)
            if best.move is not None:
                moves.append(best.move)
            node = best

        return moves
