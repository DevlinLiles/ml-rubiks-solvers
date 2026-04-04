"""Tests for the MCTSNode implementation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rubiks_solve.solvers.mcts.node import MCTSNode
from rubiks_solve.core.base import Move


# ---------------------------------------------------------------------------
# Helpers: minimal mock puzzle classes
# ---------------------------------------------------------------------------

class _FakePuzzle:
    """Minimal puzzle stub for MCTS node tests."""

    def __init__(self, solved: bool = False, n_moves: int = 3) -> None:
        self._solved = solved
        self._n_moves = n_moves
        self._state = np.zeros((6, 3, 3), dtype=np.uint8)

    @property
    def is_solved(self) -> bool:
        return self._solved

    @property
    def state(self) -> np.ndarray:
        return self._state

    def legal_moves(self) -> list[Move]:
        return [
            Move(name=f"M{i}", face="U", layer=0, direction=1, double=False)
            for i in range(self._n_moves)
        ]

    def apply_move(self, move: Move) -> "_FakePuzzle":
        return _FakePuzzle(solved=False, n_moves=self._n_moves)

    @classmethod
    def solved_state(cls) -> "_FakePuzzle":
        return cls(solved=True)


class _SolvedPuzzle(_FakePuzzle):
    """Puzzle that reports is_solved = True."""

    def __init__(self) -> None:
        super().__init__(solved=True)


# ---------------------------------------------------------------------------
# UCB1 tests
# ---------------------------------------------------------------------------

def test_ucb1_unvisited():
    """An unvisited node's UCB1 score must be positive infinity."""
    puzzle = _FakePuzzle()
    node = MCTSNode(puzzle)
    assert node.ucb1() == math.inf


def test_ucb1_formula():
    """UCB1 should match the analytical formula for a visited node."""
    puzzle = _FakePuzzle()
    parent = MCTSNode(puzzle)
    child = MCTSNode(puzzle, parent=parent)

    # Manually set statistics
    parent.visits = 10
    child.visits = 4
    child.value = 3.0

    c = 1.414
    expected = (3.0 / 4.0) + c * math.sqrt(math.log(10) / 4)
    assert abs(child.ucb1(c=c) - expected) < 1e-9


def test_ucb1_unvisited_with_visited_parent():
    """Unvisited child still returns infinity regardless of parent visits."""
    puzzle = _FakePuzzle()
    parent = MCTSNode(puzzle)
    parent.visits = 5
    child = MCTSNode(puzzle, parent=parent)
    assert child.ucb1() == math.inf


# ---------------------------------------------------------------------------
# Expansion tests
# ---------------------------------------------------------------------------

def test_expand_creates_child():
    """expand() adds exactly one child."""
    puzzle = _FakePuzzle(n_moves=3)
    node = MCTSNode(puzzle)
    child = node.expand()
    assert len(node.children) == 1
    assert child in node.children


def test_expand_reduces_untried():
    """After expand(), untried moves list has one fewer entry."""
    puzzle = _FakePuzzle(n_moves=3)
    node = MCTSNode(puzzle)
    node.expand()
    # First call initialises _untried_moves (3 moves) and pops one → 2 left
    assert len(node._untried_moves) == 2


def test_expand_all_moves():
    """Expanding n times fully expands a node with n legal moves."""
    n = 4
    puzzle = _FakePuzzle(n_moves=n)
    node = MCTSNode(puzzle)
    for _ in range(n):
        node.expand()
    assert node.is_fully_expanded()
    assert len(node.children) == n


def test_expand_raises_when_fully_expanded():
    """expand() on a fully expanded node raises RuntimeError."""
    puzzle = _FakePuzzle(n_moves=2)
    node = MCTSNode(puzzle)
    node.expand()
    node.expand()
    with pytest.raises(RuntimeError, match="fully expanded"):
        node.expand()


def test_expand_child_has_correct_parent():
    """The child returned by expand() has the correct parent reference."""
    puzzle = _FakePuzzle(n_moves=3)
    node = MCTSNode(puzzle)
    child = node.expand()
    assert child.parent is node


def test_expand_child_has_move():
    """The child returned by expand() has a non-None move attribute."""
    puzzle = _FakePuzzle(n_moves=3)
    node = MCTSNode(puzzle)
    child = node.expand()
    assert child.move is not None


# ---------------------------------------------------------------------------
# Backpropagation tests
# ---------------------------------------------------------------------------

def test_backpropagate_updates_visits():
    """backpropagate increments visit counts up to the root."""
    puzzle = _FakePuzzle()
    root = MCTSNode(puzzle)
    child = MCTSNode(puzzle, parent=root)
    grandchild = MCTSNode(puzzle, parent=child)

    grandchild.backpropagate(1.0)

    assert grandchild.visits == 1
    assert child.visits == 1
    assert root.visits == 1


def test_backpropagate_updates_value():
    """backpropagate accumulates reward in all ancestor nodes."""
    puzzle = _FakePuzzle()
    root = MCTSNode(puzzle)
    child = MCTSNode(puzzle, parent=root)

    child.backpropagate(0.75)

    assert abs(child.value - 0.75) < 1e-9
    assert abs(root.value - 0.75) < 1e-9


def test_backpropagate_multiple_updates():
    """Multiple backpropagations accumulate correctly."""
    puzzle = _FakePuzzle()
    root = MCTSNode(puzzle)
    child = MCTSNode(puzzle, parent=root)

    child.backpropagate(0.5)
    child.backpropagate(0.3)

    assert child.visits == 2
    assert abs(child.value - 0.8) < 1e-9
    assert abs(root.value - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# is_terminal tests
# ---------------------------------------------------------------------------

def test_is_terminal_solved_puzzle():
    """is_terminal() returns True when puzzle.is_solved is True."""
    solved = _SolvedPuzzle()
    node = MCTSNode(solved)
    assert node.is_terminal() is True


def test_is_terminal_unsolved_puzzle():
    """is_terminal() returns False when puzzle is not solved."""
    puzzle = _FakePuzzle(solved=False)
    node = MCTSNode(puzzle)
    assert node.is_terminal() is False


# ---------------------------------------------------------------------------
# Solution path tests
# ---------------------------------------------------------------------------

def test_solution_path_root():
    """Root node solution_path() returns an empty list."""
    puzzle = _FakePuzzle()
    root = MCTSNode(puzzle)
    assert not root.solution_path()


def test_solution_path_correct():
    """solution_path() returns moves from root to the node in order."""
    puzzle = _FakePuzzle(n_moves=3)
    move_a = Move(name="A", face="U", layer=0, direction=1, double=False)
    move_b = Move(name="B", face="D", layer=0, direction=-1, double=False)

    root = MCTSNode(puzzle)
    child = MCTSNode(puzzle, parent=root, move=move_a)
    grandchild = MCTSNode(puzzle, parent=child, move=move_b)

    path = grandchild.solution_path()
    assert path == [move_a, move_b]


def test_solution_path_single_move():
    """A child node's solution_path() contains exactly one move."""
    puzzle = _FakePuzzle(n_moves=2)
    move = Move(name="X", face="F", layer=0, direction=1, double=False)
    root = MCTSNode(puzzle)
    child = MCTSNode(puzzle, parent=root, move=move)
    assert child.solution_path() == [move]


# ---------------------------------------------------------------------------
# best_child tests
# ---------------------------------------------------------------------------

def test_best_child_raises_no_children():
    """best_child() raises ValueError when there are no children."""
    puzzle = _FakePuzzle()
    node = MCTSNode(puzzle)
    with pytest.raises(ValueError, match="no children"):
        node.best_child()


def test_best_child_returns_highest_ucb1():
    """best_child() returns the child with the highest UCB1 score."""
    puzzle = _FakePuzzle(n_moves=2)
    parent = MCTSNode(puzzle)
    parent.visits = 10

    # Child A: visited 1 time, value 0.9
    child_a = MCTSNode(puzzle, parent=parent)
    child_a.visits = 1
    child_a.value = 0.9
    parent.children.append(child_a)

    # Child B: unvisited → UCB1 = inf
    child_b = MCTSNode(puzzle, parent=parent)
    parent.children.append(child_b)

    assert parent.best_child() is child_b
