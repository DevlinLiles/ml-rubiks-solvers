"""
rubiks_solve.core — Puzzle engine for NxN Rubik's cubes.

Public API:
  Base types:
    AbstractPuzzle  — abstract base class for all puzzle types
    Move            — immutable move descriptor

  Move utilities:
    get_moves(n)         — legal moves for an NxN cube
    get_inverse_move(m)  — inverse of a move
    MOVES_2x2            — module-level constant (9 moves)
    MOVES_3x3            — module-level constant (18 moves)
    MOVES_4x4            — module-level constant (45 moves)
    MOVES_5x5            — module-level constant (72 moves)

  Cube classes:
    CubeNNN   — generic NxN cube (parameterised by n)
    Cube2x2   — 2x2 Pocket Cube
    Cube3x3   — 3x3 Standard Cube
    Cube4x4   — 4x4 Revenge Cube
    Cube5x5   — 5x5 Professor's Cube

  Validation:
    is_valid_state(puzzle)   — checks sticker colour counts
    has_parity_error(puzzle) — detects OLL/PLL parity (even-n cubes)
    is_reachable(puzzle)     — combined validity + parity check
"""
from rubiks_solve.core.base import AbstractPuzzle, Move

from rubiks_solve.core.moves import (
    MOVES_2x2,
    MOVES_3x3,
    MOVES_4x4,
    MOVES_5x5,
    get_moves,
    get_inverse_move,
)

from rubiks_solve.core.cube_nnn import CubeNNN
from rubiks_solve.core.cube_2x2 import Cube2x2
from rubiks_solve.core.cube_3x3 import Cube3x3
from rubiks_solve.core.cube_4x4 import Cube4x4
from rubiks_solve.core.cube_5x5 import Cube5x5

from rubiks_solve.core.validator import (
    is_valid_state,
    has_parity_error,
    is_reachable,
)

# Megaminx is a separate puzzle type; import guarded so NxN code works without it.
try:
    from rubiks_solve.core.megaminx import Megaminx, MEGAMINX_MOVES  # type: ignore[import]
    _MEGAMINX_AVAILABLE = True
except ImportError:
    _MEGAMINX_AVAILABLE = False

from rubiks_solve.core.skewb_ultimate import SkewbUltimate, SKEWB_ULTIMATE_MOVES

__all__ = [
    # Base
    "AbstractPuzzle",
    "Move",
    # Moves
    "get_moves",
    "get_inverse_move",
    "MOVES_2x2",
    "MOVES_3x3",
    "MOVES_4x4",
    "MOVES_5x5",
    # Cubes
    "CubeNNN",
    "Cube2x2",
    "Cube3x3",
    "Cube4x4",
    "Cube5x5",
    # Validation
    "is_valid_state",
    "has_parity_error",
    "is_reachable",
]

if _MEGAMINX_AVAILABLE:
    __all__ += ["Megaminx", "MEGAMINX_MOVES"]

__all__ += ["SkewbUltimate", "SKEWB_ULTIMATE_MOVES"]
