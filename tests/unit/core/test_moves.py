"""Tests for the moves module: move counts, inverses, and naming."""
from __future__ import annotations

import pytest

from rubiks_solve.core.moves import get_moves, get_inverse_move
from rubiks_solve.core.base import Move


@pytest.mark.parametrize("n,expected_count", [
    (2, 9),
    (3, 18),
    (4, 45),
    (5, 72),
])
def test_get_moves_count(n, expected_count):
    """Verify the expected number of moves for each cube size."""
    moves = get_moves(n)
    assert len(moves) == expected_count, (
        f"Expected {expected_count} moves for {n}x{n}, got {len(moves)}"
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_move_has_inverse(n):
    """Every move's inverse must also be in the move set (by name or by double identity)."""
    moves = get_moves(n)
    move_names = {m.name for m in moves}
    for move in moves:
        inv = get_inverse_move(move)
        # For double moves the inverse IS the same move
        if move.double:
            assert inv == move
        else:
            assert inv.name in move_names, (
                f"Inverse of '{move.name}' ('{inv.name}') not found in move set for {n}x{n}"
            )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_inverse_of_inverse(n):
    """inverse(inverse(m)) == m for all moves."""
    for move in get_moves(n):
        double_inv = get_inverse_move(get_inverse_move(move))
        assert double_inv == move, (
            f"inverse(inverse({move!r})) != {move!r}: got {double_inv!r}"
        )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_move_names_unique(n):
    """All move names must be distinct within a puzzle size."""
    moves = get_moves(n)
    names = [m.name for m in moves]
    assert len(names) == len(set(names)), (
        f"Duplicate move names found for {n}x{n}: "
        + str([name for name in names if names.count(name) > 1])
    )


def test_double_move_is_self_inverse():
    """A double move's inverse must be the same move (double=True)."""
    moves_3x3 = get_moves(3)
    double_moves = [m for m in moves_3x3 if m.double]
    assert len(double_moves) > 0, "No double moves found for 3x3"
    for move in double_moves:
        inv = move.inverse()
        assert inv == move, f"Double move {move!r} inverse should be itself, got {inv!r}"


def test_get_moves_invalid_size():
    """get_moves raises ValueError for unsupported cube size."""
    with pytest.raises(ValueError, match="Unsupported cube size"):
        get_moves(6)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_moves_are_move_instances(n):
    """All returned objects must be Move instances."""
    for move in get_moves(n):
        assert isinstance(move, Move)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_moves_have_valid_direction(n):
    """Every move's direction is +1 or -1."""
    for move in get_moves(n):
        assert move.direction in (+1, -1), (
            f"Move {move!r} has unexpected direction {move.direction}"
        )


def test_inverse_name_convention():
    """CW move inverse appends apostrophe; CCW move inverse strips it."""
    cw_move = Move(name="R", face="R", layer=0, direction=1, double=False)
    ccw_move = Move(name="R'", face="R", layer=0, direction=-1, double=False)
    assert cw_move.inverse().name == "R'"
    assert ccw_move.inverse().name == "R"


def test_move_dataclass_frozen():
    """Move is a frozen dataclass — attempts to mutate raise AttributeError."""
    move = Move(name="U", face="U", layer=0, direction=1, double=False)
    with pytest.raises(AttributeError):
        move.direction = -1  # type: ignore[misc]
