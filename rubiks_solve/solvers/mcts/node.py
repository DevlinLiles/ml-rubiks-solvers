"""Monte Carlo Tree Search node implementation."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle, Move

if TYPE_CHECKING:
    pass


def _count_misplaced(puzzle: AbstractPuzzle) -> int:
    """Return the number of facelets not in their solved position.

    Args:
        puzzle: Puzzle state to evaluate.

    Returns:
        Integer count of out-of-place facelets.
    """
    solved = type(puzzle).solved_state()
    return int(np.sum(puzzle.state != solved.state))


class MCTSNode:
    """A node in the Monte Carlo Tree Search.

    Each node represents a puzzle state reached by a sequence of moves from the
    root.  The node stores visit statistics used by the UCB1 formula to balance
    exploration against exploitation.

    Attributes:
        puzzle: The puzzle state at this node (immutable).
        parent: Parent node, or ``None`` for the root.
        move: The move applied to the parent's state to reach this node.
        children: Expanded child nodes.
        visits: Number of times this node has been visited.
        value: Cumulative reward accumulated through this node.
    """

    def __init__(
        self,
        puzzle: AbstractPuzzle,
        parent: MCTSNode | None = None,
        move: Move | None = None,
    ) -> None:
        """Initialise a new MCTS node.

        Args:
            puzzle: The puzzle state represented by this node.
            parent: The parent node in the search tree (``None`` for root).
            move: The move that led from the parent state to this state.
        """
        self.puzzle: AbstractPuzzle = puzzle
        self.parent: MCTSNode | None = parent
        self.move: Move | None = move
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self._untried_moves: list[Move] | None = None

    # ------------------------------------------------------------------
    # UCB1 / selection
    # ------------------------------------------------------------------

    def ucb1(self, c: float = 1.414) -> float:
        """Upper Confidence Bound 1 score for this node.

        Balances exploitation (high average reward) against exploration (low
        visit count relative to the parent).  Nodes with zero visits return
        positive infinity so that every child is visited at least once before
        any is revisited.

        Args:
            c: Exploration constant.  Higher values favour less-visited nodes.
                Defaults to ``sqrt(2) ≈ 1.414``.

        Returns:
            UCB1 score as a float.
        """
        if self.visits == 0:
            return math.inf
        parent_visits = self.parent.visits if self.parent is not None else self.visits
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = 1.414) -> MCTSNode:
        """Return the child with the highest UCB1 score.

        Args:
            c: Exploration constant forwarded to :meth:`ucb1`.

        Returns:
            The child node with the highest UCB1 value.

        Raises:
            ValueError: If this node has no children.
        """
        if not self.children:
            raise ValueError("best_child called on a node with no children.")
        return max(self.children, key=lambda child: child.ucb1(c))

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(self) -> MCTSNode:
        """Add one untried child node and return it.

        The first call initialises the untried-move list from
        ``puzzle.legal_moves()``.  Subsequent calls pop the first untried move
        and create the corresponding child.

        Returns:
            The newly created child node.

        Raises:
            RuntimeError: If the node is already fully expanded.
        """
        if self._untried_moves is None:
            self._untried_moves = list(self.puzzle.legal_moves())

        if not self._untried_moves:
            raise RuntimeError("expand() called on a fully expanded node.")

        move = self._untried_moves.pop(0)
        child_puzzle = self.puzzle.apply_move(move)
        child = MCTSNode(puzzle=child_puzzle, parent=self, move=move)
        self.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(self, max_depth: int, rng: np.random.Generator) -> float:
        """Perform a random rollout from this node's state.

        Applies random moves up to *max_depth* times.  Returns 1.0 if the
        puzzle is solved at any point during the rollout, or a heuristic
        reward in ``[0, 1)`` proportional to the fraction of facelets that are
        in their correct position.

        Args:
            max_depth: Maximum number of random moves to apply.
            rng: NumPy random generator used for move selection.

        Returns:
            Reward in ``[0.0, 1.0]`` (1.0 = solved, higher = closer to solved).
        """
        state = self.puzzle
        legal_moves = state.legal_moves()
        n_moves = len(legal_moves)

        for _ in range(max_depth):
            if state.is_solved:
                return 1.0
            idx = int(rng.integers(n_moves))
            state = state.apply_move(legal_moves[idx])

        if state.is_solved:
            return 1.0

        # Heuristic: fraction of correctly placed facelets
        total = int(state.state.size)
        if total == 0:
            return 0.0
        misplaced = _count_misplaced(state)
        return float(total - misplaced) / float(total)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backpropagate(self, reward: float) -> None:
        """Propagate *reward* up to the root, updating visit counts and values.

        Increments the visit count of every node from this node to the root
        (inclusive) and accumulates the reward into each node's value.

        Args:
            reward: The reward obtained from a rollout (typically in
                ``[0.0, 1.0]``).
        """
        node: MCTSNode | None = self
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        """Return ``True`` if the puzzle at this node is in the solved state.

        Returns:
            ``True`` when ``puzzle.is_solved`` is ``True``.
        """
        return self.puzzle.is_solved

    def is_fully_expanded(self) -> bool:
        """Return ``True`` when all legal moves have been expanded.

        A node is considered fully expanded once :meth:`expand` has been called
        for every legal move (i.e. the untried-move list has been initialised
        and is now empty).

        Returns:
            ``True`` if no untried moves remain.
        """
        if self._untried_moves is None:
            return False
        return len(self._untried_moves) == 0

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    def solution_path(self) -> list[Move]:
        """Reconstruct the move sequence from the root to this node.

        Walks up the parent chain collecting moves, then reverses the result.

        Returns:
            Ordered list of :class:`~rubiks_solve.core.base.Move` objects from
            the root state to this node's state.
        """
        moves: list[Move] = []
        node: MCTSNode | None = self
        while node is not None and node.move is not None:
            moves.append(node.move)
            node = node.parent
        moves.reverse()
        return moves

    def __repr__(self) -> str:
        return (
            f"MCTSNode(move={self.move!r}, visits={self.visits}, "
            f"value={self.value:.3f}, children={len(self.children)})"
        )
