"""
Skewb Ultimate puzzle implementation.

The Skewb Ultimate (also sold as "Puzzleball") is a dodecahedral twisty puzzle
built on the Skewb mechanism.  It has 12 pentagonal faces and turns on 4 axes,
each passing through a pair of opposite vertices of the dodecahedron.  Every
turn is a deep cut: each move rotates exactly half the puzzle by 120 degrees.

Pieces
------
* 8 corner pieces  — sit at the 8 vertices of a cube inscribed in the
  dodecahedron.  Each has 3 possible orientations.
* 6 face pieces    — rhombus-shaped pieces, one for each face of the inscribed
  cube.  Each has 2 possible orientations (flipped / not-flipped).

State representation
--------------------
numpy array of shape (14, 2), dtype uint8.

  Rows 0-7:   corner slots.  state[i] = [piece_id, orientation]
                 piece_id  ∈ 0-7,  orientation ∈ 0-2.
  Rows 8-13:  face piece slots.  state[8+j] = [piece_id, orientation]
                 piece_id  ∈ 0-5,  orientation ∈ 0-1.

Solved state: state[i] = [i, 0] for i ∈ 0..7 (corners),
              state[8+j] = [j, 0] for j ∈ 0..5 (face pieces).

Face piece indices (faces of the inscribed cube, corners at ±x/±y/±z):
  0 = L  (x = 0 face)    1 = R  (x = 1 face)
  2 = B  (y = 0 face)    3 = F  (y = 1 face)
  4 = D  (z = 0 face)    5 = U  (z = 1 face)

Corner indices (vertices of the inscribed cube, x + 2y + 4z encoding):
  0 = LBD  1 = RBD  2 = LFD  3 = RFD
  4 = LBU  5 = RBU  6 = LFU  7 = RFU

Axes and moves
--------------
The 4 turning axes each pass through one pair of opposite cube corners:

  Axis L: corners 0(LBD) ↔ 7(RFU)   move name "L" / "L'"
  Axis R: corners 1(RBD) ↔ 6(LFU)   move name "R" / "R'"
  Axis F: corners 2(LFD) ↔ 5(RBU)   move name "F" / "F'"
  Axis B: corners 3(RFD) ↔ 4(LBU)   move name "B" / "B'"

For each axis the CW move (direction=+1) is defined by rotating the half
containing the *lower-indexed* corner clockwise when viewed from that corner.
Inverse / CCW moves use direction=-1.

God's number: 14 moves (every state solvable in at most 14 HTM moves).
State space:  100,776,960 reachable positions.
"""
from __future__ import annotations

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle, Move

# ---------------------------------------------------------------------------
# Corner and face-piece index constants
# ---------------------------------------------------------------------------

# Face-piece indices (faces of the inscribed cube)
_L, _R, _B, _F, _D, _U = 0, 1, 2, 3, 4, 5  # noqa: E741

# Corner indices (vertices of the inscribed cube)
_LBD, _RBD, _LFD, _RFD, _LBU, _RBU, _LFU, _RFU = range(8)

# ---------------------------------------------------------------------------
# Move definitions
# ---------------------------------------------------------------------------
# Each axis has a CW move (direction +1) with:
#   corner_cycle: (a, b, c) — corner piece at slot a moves to slot b,
#                              b to c, c to a.  The axis-corner (near pole)
#                              is not in the cycle; it stays in place.
#   face_cycle:   (a, b, c) — face piece at slot a moves to slot b, etc.
#   corner_delta: orientation added to each cycling corner (mod 3).
#   face_delta:   orientation added to each cycling face piece (mod 2).
#
# Derivation: each move is a +120° rotation of the "near half" of the puzzle
# (the half containing the lower-indexed corner of the axis) around the body
# diagonal.  The rotation cycles the three non-axis corners on that half and
# the three face pieces on that half.
#
# Convention details:
#   Face piece U (index 5) lies on the "far half" for all four axes and is
#   therefore never moved by any of the 8 generators.  This is equivalent to
#   choosing face U as the orientation-reference fixed piece.
#
#   Two corner orbits under the generated group: {0,3,5,6} and {1,2,4,7}.
#   Each CW move cycles exactly the three non-axis pieces in one orbit with
#   corner_delta=+1, giving a total orbit-orientation change of +3 ≡ 0 (mod 3).
#
# axis_name → (corner_cycle, face_cycle, corner_delta, face_delta)
_AXIS_CW: dict[str, tuple[tuple[int, int, int],
                           tuple[int, int, int],
                           int, int]] = {
    # Axis L: near corner LBD(0), far corner RFU(7).
    # Near-half corners (non-axis): RBD(1), LFD(2), LBU(4)  [orbit {1,2,4,7}]
    # Near-half face pieces: L(0), B(2), D(4)
    # Rotation +x→+y→+z→+x maps displacements: +x→+y, +y→+z, +z→+x
    # ⟹ corner cycle 1→2→4→1, face cycle L→B→D→L
    "L": ((_RBD, _LFD, _LBU), (_L, _B, _D), 1, 1),

    # Axis R: near corner RBD(1), far corner LFU(6).
    # Near-half corners (non-axis): LBD(0), RFD(3), RBU(5)  [orbit {0,3,5,6}]
    # Near-half face pieces: R(1), B(2), D(4)
    # Rotation: -x→+y→+z→-x  ⟹  corner cycle 0→3→5→0, face cycle R→B→D→R
    "R": ((_LBD, _RFD, _RBU), (_R, _B, _D), 1, 1),

    # Axis F: near corner LFD(2), far corner RBU(5).
    # Near-half corners (non-axis): LBD(0), LFU(6), RFD(3)  [orbit {0,3,5,6}]
    # Near-half face pieces: L(0), F(3), D(4)
    # Rotation: -y→+z→+x→-y  ⟹  corner cycle 0→6→3→0, face cycle L→F→D→L
    "F": ((_LBD, _LFU, _RFD), (_L, _F, _D), 1, 1),

    # Axis B: near corner RFD(3), far corner LBU(4).
    # Near-half corners (non-axis): RBD(1), LFD(2), RFU(7)  [orbit {1,2,4,7}]
    # Near-half face pieces: R(1), D(4), F(3)
    # Rotation: -y→-x→+z→-y  ⟹  corner cycle 1→2→7→1, face cycle R→D→F→R
    "B": ((_RBD, _LFD, _RFU), (_R, _D, _F), 1, 1),
}

_MOVE_NAMES = ["L", "L'", "R", "R'", "F", "F'", "B", "B'"]


def _build_moves() -> list[Move]:
    """Build the list of all 8 Skewb Ultimate moves."""
    moves: list[Move] = []
    for axis in ("L", "R", "F", "B"):
        moves.append(Move(name=axis,    face=axis, layer=0, direction=+1))
        moves.append(Move(name=axis+"'", face=axis, layer=0, direction=-1))
    return moves


SKEWB_ULTIMATE_MOVES: list[Move] = _build_moves()
_MOVE_BY_NAME: dict[str, Move] = {m.name: m for m in SKEWB_ULTIMATE_MOVES}


# ---------------------------------------------------------------------------
# Move application helper
# ---------------------------------------------------------------------------

def _apply_cycle(
    state: np.ndarray,
    cycle: tuple[int, int, int],
    deltas: tuple[int, int, int],
    max_ori: int,
    row_offset: int,
) -> None:
    """Apply a 3-cycle in-place to *state* rows in the half [row_offset:].

    The cycle (a, b, c) moves piece from slot a→b, b→c, c→a.  Each transition
    adds the corresponding delta from *deltas* to the piece's orientation
    (mod *max_ori*).

    Args:
        state:      State array being mutated (already a copy).
        cycle:      (a, b, c) slot indices within the piece group.
        deltas:     (da, db, dc) orientation increments for the transitions
                    a→b, b→c, and c→a respectively.
        max_ori:    Modulus for orientation (3 for corners, 2 for faces).
        row_offset: Row in state where this piece group starts (0 for
                    corners, 8 for face pieces).
    """
    a, b, c = cycle
    da, db, dc = deltas
    ra, rb, rc = row_offset + a, row_offset + b, row_offset + c
    # Save originals before any overwrite.
    pid_a, ori_a = int(state[ra, 0]), int(state[ra, 1])
    pid_b, ori_b = int(state[rb, 0]), int(state[rb, 1])
    pid_c, ori_c = int(state[rc, 0]), int(state[rc, 1])
    # Write new positions: a→b, b→c, c→a.
    state[rb, 0] = pid_a
    state[rb, 1] = (ori_a + da) % max_ori
    state[rc, 0] = pid_b
    state[rc, 1] = (ori_b + db) % max_ori
    state[ra, 0] = pid_c
    state[ra, 1] = (ori_c + dc) % max_ori


# ---------------------------------------------------------------------------
# SkewbUltimate class
# ---------------------------------------------------------------------------

class SkewbUltimate(AbstractPuzzle):
    """Immutable Skewb Ultimate puzzle state.

    The Skewb Ultimate is a dodecahedral twisty puzzle built on the Skewb
    mechanism, with 4 axes of rotation and 8 possible moves (four axes ×
    CW / CCW).

    State is stored as a (14, 2) numpy array:
      rows 0-7  — corner slots: [piece_id (0-7), orientation (0-2)]
      rows 8-13 — face-piece slots: [piece_id (0-5), orientation (0-1)]

    Immutability contract: :meth:`apply_move` always returns a new instance
    and never modifies ``self``.
    """

    def __init__(self, state: np.ndarray) -> None:
        """Initialise a SkewbUltimate from an explicit state array.

        Args:
            state: numpy array of shape (14, 2) and dtype uint8.

        Raises:
            ValueError: If ``state`` has the wrong shape or dtype.
        """
        if state.shape != (14, 2):
            raise ValueError(
                f"SkewbUltimate state must have shape (14, 2), got {state.shape}"
            )
        if state.dtype != np.uint8:
            raise ValueError(
                f"SkewbUltimate state must have dtype uint8, got {state.dtype}"
            )
        self._state: np.ndarray = state

    # ------------------------------------------------------------------
    # AbstractPuzzle interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Canonical state array of shape (14, 2) dtype uint8.

        Returns a read-only view; mutate via :meth:`apply_move` instead.
        """
        view = self._state.view()
        view.flags.writeable = False
        return view

    @property
    def is_solved(self) -> bool:
        """Return True iff every piece is in its home slot with orientation 0."""
        for i in range(8):
            if self._state[i, 0] != i or self._state[i, 1] != 0:
                return False
        for j in range(6):
            if self._state[8 + j, 0] != j or self._state[8 + j, 1] != 0:
                return False
        return True

    def apply_move(self, move: Move) -> "SkewbUltimate":
        """Return a new SkewbUltimate with the given move applied.

        Args:
            move: A Move whose ``face`` is one of 'L', 'R', 'F', 'B' and
                  whose ``direction`` is +1 (CW) or -1 (CCW).

        Returns:
            New SkewbUltimate instance after the move.

        Raises:
            ValueError: If ``move.face`` is not a valid axis name.
        """
        if move.face not in _AXIS_CW:
            raise ValueError(
                f"Unknown SkewbUltimate axis: {move.face!r}. "
                f"Valid axes: {list(_AXIS_CW)}"
            )

        corner_cycle, face_cycle, c_delta, _f_delta = _AXIS_CW[move.face]

        if move.direction == +1:
            # CW: uniform corner delta; asymmetric face deltas (1,1,0) → sum=2≡0 mod 2 ✓
            c_deltas = (c_delta, c_delta, c_delta)
            f_deltas = (1, 1, 0)
        else:
            # CCW: reverse cycle order, corner delta = 3-1=2; face deltas (0,1,1) → sum=2≡0 mod 2 ✓
            corner_cycle = (corner_cycle[0], corner_cycle[2], corner_cycle[1])
            face_cycle   = (face_cycle[0],   face_cycle[2],   face_cycle[1])
            c_deltas = (2, 2, 2)
            f_deltas = (0, 1, 1)

        new_state = self._state.copy()
        _apply_cycle(new_state, corner_cycle, c_deltas, 3, row_offset=0)
        _apply_cycle(new_state, face_cycle,   f_deltas, 2, row_offset=8)
        return SkewbUltimate(new_state)

    def legal_moves(self) -> list[Move]:
        """Return all 8 legal Skewb Ultimate moves.

        Returns:
            List of 8 Move objects (4 axes × CW / CCW).
        """
        return list(SKEWB_ULTIMATE_MOVES)

    def scramble(self, n_moves: int, rng: np.random.Generator) -> "SkewbUltimate":
        """Return a new scrambled SkewbUltimate reached by applying n_moves random moves.

        Avoids immediately inverting the previous move to prevent trivial
        cancellations.

        Args:
            n_moves: Number of random moves to apply.
            rng:     NumPy random generator for reproducible scrambles.

        Returns:
            New SkewbUltimate instance after the scramble sequence.
        """
        puzzle: SkewbUltimate = self
        last_move: Move | None = None

        for _ in range(n_moves):
            candidates = SKEWB_ULTIMATE_MOVES
            if last_move is not None:
                inv = last_move.inverse()
                candidates = [m for m in SKEWB_ULTIMATE_MOVES if m.name != inv.name]
            chosen = candidates[int(rng.integers(len(candidates)))]
            puzzle = puzzle.apply_move(chosen)
            last_move = chosen

        return puzzle

    def copy(self) -> "SkewbUltimate":
        """Return a deep copy of this SkewbUltimate instance."""
        return SkewbUltimate(self._state.copy())

    @classmethod
    def solved_state(cls) -> "SkewbUltimate":
        """Return a new SkewbUltimate in the canonical solved state.

        In the solved state piece i occupies slot i with orientation 0.

        Returns:
            New solved SkewbUltimate instance.
        """
        state = np.zeros((14, 2), dtype=np.uint8)
        for i in range(8):
            state[i, 0] = i   # corner piece i in slot i
        for j in range(6):
            state[8 + j, 0] = j  # face piece j in slot j
        return cls(state)

    @classmethod
    def move_limit(cls) -> int:
        """Return the maximum move budget for solving (God's number = 14 HTM).

        Returns:
            14
        """
        return 14

    @classmethod
    def puzzle_name(cls) -> str:
        """Return the short human-readable puzzle name.

        Returns:
            ``'skewb_ultimate'``
        """
        return "skewb_ultimate"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def corner_state(self) -> np.ndarray:
        """Return the 8×2 corner sub-array (read-only view).

        Returns:
            shape (8, 2) uint8: ``[piece_id, orientation]`` per corner slot.
        """
        view = self._state[:8].view()
        view.flags.writeable = False
        return view

    def face_piece_state(self) -> np.ndarray:
        """Return the 6×2 face-piece sub-array (read-only view).

        Returns:
            shape (6, 2) uint8: ``[piece_id, orientation]`` per face-piece slot.
        """
        view = self._state[8:].view()
        view.flags.writeable = False
        return view

    def __repr__(self) -> str:
        status = "solved" if self.is_solved else "scrambled"
        return f"SkewbUltimate({status})"
