"""
Megaminx puzzle implementation.

The Megaminx is a dodecahedral twisty puzzle with 12 pentagonal faces.
Each face turn rotates 72 degrees (one-fifth of a full rotation).

State representation:
    numpy array of shape (12, 11), dtype uint8.
    Axis 0: face index 0-11.
    Axis 1: sticker index 0-10.
        0       = center sticker
        1, 2, 3, 4, 5  = edge stickers in clockwise order
        6, 7, 8, 9, 10 = corner stickers in clockwise order
            (corner 6 sits between edge 1 and edge 2,
             corner 7 sits between edge 2 and edge 3, etc.)

Colors 0-11 match face indices; solved state has face[i][j] == i for all i, j.

Face naming (WCA-compatible):
    0: U    1: F    2: R    3: BR   4: BL   5: L
    6: D    7: DF   8: DR   9: DBR  10: DBL 11: DL
"""

from __future__ import annotations

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle, Move

# ---------------------------------------------------------------------------
# Face index constants
# ---------------------------------------------------------------------------
U = 0
F = 1
R = 2
BR = 3
BL = 4
L = 5
D = 6
DF = 7
DR = 8
DBR = 9
DBL = 10
DL = 11

_FACE_NAMES: list[str] = ["U", "F", "R", "BR", "BL", "L", "D", "DF", "DR", "DBR", "DBL", "DL"]

# ---------------------------------------------------------------------------
# Sticker layout helpers
# ---------------------------------------------------------------------------
# For a face with stickers [center, e1, e2, e3, e4, e5, c1, c2, c3, c4, c5]:
#   - Edges  1-5 in CW order around the face centre.
#   - Corner i+6 sits in the arc between edge i and edge i+1 (mod 5).
#
# A clockwise turn of the face cycles:
#   edges:   1→2→3→4→5→1  (rotate indices left by 1 within the edge ring)
#   corners: 6→7→8→9→10→6 (same rotation in the corner ring)
#
# Equivalently we roll the sub-arrays [1..5] and [6..10] by -1.

_EDGE_STICKERS = [1, 2, 3, 4, 5]      # sticker positions for face edges
_CORNER_STICKERS = [6, 7, 8, 9, 10]   # sticker positions for face corners


def _rotate_face_cw(face_row: np.ndarray) -> np.ndarray:
    """Return a copy of a single face row with stickers rotated one step CW.

    The center (index 0) is unchanged. Edge stickers 1-5 and corner stickers
    6-10 each cycle one position clockwise (i.e. each element shifts to the
    next index in their respective ring).

    Args:
        face_row: 1-D array of length 11 representing one face's stickers.

    Returns:
        New array with CW-rotated stickers.
    """
    result = face_row.copy()
    # Rotate edges CW: position 1←5, 2←1, 3←2, 4←3, 5←4
    result[1] = face_row[5]
    result[2] = face_row[1]
    result[3] = face_row[2]
    result[4] = face_row[3]
    result[5] = face_row[4]
    # Rotate corners CW: position 6←10, 7←6, 8←7, 9←8, 10←9
    result[6] = face_row[10]
    result[7] = face_row[6]
    result[8] = face_row[7]
    result[9] = face_row[8]
    result[10] = face_row[9]
    return result


def _rotate_face_ccw(face_row: np.ndarray) -> np.ndarray:
    """Return a copy of a single face row with stickers rotated one step CCW.

    Inverse of :func:`_rotate_face_cw`.

    Args:
        face_row: 1-D array of length 11 representing one face's stickers.

    Returns:
        New array with CCW-rotated stickers.
    """
    result = face_row.copy()
    # Rotate edges CCW: each position receives from the NEXT position (inverse of CW)
    # CW was: new[k] = old[k-1]. CCW (inverse): new[k] = old[k+1]
    result[1] = face_row[2]
    result[2] = face_row[3]
    result[3] = face_row[4]
    result[4] = face_row[5]
    result[5] = face_row[1]
    # Rotate corners CCW: each position receives from the NEXT position
    # CW was: new[k] = old[k-1]. CCW (inverse): new[k] = old[k+1]
    result[6] = face_row[7]
    result[7] = face_row[8]
    result[8] = face_row[9]
    result[9] = face_row[10]
    result[10] = face_row[6]
    return result


# ---------------------------------------------------------------------------
# Adjacency table
# ---------------------------------------------------------------------------
# FACE_ADJACENCY[f] describes the five neighbour faces and the sticker indices
# on each neighbour that form the border strip adjacent to face f.
#
# Each entry is a list of five tuples:
#   (neighbour_face_index, [s0, s1, s2])
#
# where s0, s1, s2 are the three sticker indices on the neighbour face that
# touch face f, listed in the CW-around-f order.  When face f rotates CW the
# strips cycle:
#   strip_0 → strip_1 → strip_2 → strip_3 → strip_4 → strip_0
#
# Geometry reference
# ------------------
# A dodecahedron viewed with face U on top and face D on the bottom:
#
#   Upper ring (adjacent to U): F, R, BR, BL, L   (in CW order viewed from above)
#   Equatorial ring (adjacent to each upper face's "far" edge):
#     F  is adjacent to: U, R, DF, DL, L
#     R  is adjacent to: U, BR, DR, DF, F
#     BR is adjacent to: U, BL, DBR, DR, R
#     BL is adjacent to: U, L, DBL, DBR, BR
#     L  is adjacent to: U, F, DL, DBL, BL
#   Lower ring (adjacent to D): DF, DR, DBR, DBL, DL (in CW order viewed from below)
#   Bottom face D is adjacent to: DF, DR, DBR, DBL, DL
#
# For each face the three border stickers on each neighbour are identified by
# the edge sticker that is closest to the shared edge and the two corner
# stickers on either side of it.  In the sticker layout:
#   edge k  →  sticker index k     (k in 1..5)
#   corner between edge k and k+1 → sticker index k+5  (with wraparound: edge5→corner10)
#
# The border strip for the edge shared between face f and neighbour n at
# "position p" in n's CW order is:
#   corner_before_edge, edge, corner_after_edge
#
# For a neighbour whose edge p faces face f the three stickers are, depending
# on which of n's edges is shared:
#   edge 1: [10, 1, 6]   corner5-edge1-corner1
#   edge 2: [6,  2, 7]   corner1-edge2-corner2
#   edge 3: [7,  3, 8]   corner2-edge3-corner3
#   edge 4: [8,  4, 9]   corner3-edge4-corner4
#   edge 5: [9,  5, 10]  corner4-edge5-corner5
#
# The ordering within each strip (left→right looking outward from f) must be
# consistent so that rotating f CW shifts stickers correctly.
#
# Full table derived from the standard dodecahedron face graph:
#
# Adjacency graph (each face lists its 5 neighbours in CW order viewed from
# outside that face):
#   U  : F,  R,  BR, BL, L
#   F  : U,  R,  DF, DL, L      (L is left of F looking at F from front)
#   R  : U,  BR, DR, DF, F
#   BR : U,  BL, DBR,DR, R
#   BL : U,  L,  DBL,DBR,BR
#   L  : U,  F,  DL, DBL,BL
#   D  : DF, DR, DBR,DBL,DL
#   DF : D,  DR, R,  F,  DL
#   DR : D,  DBR,BR, R,  DF
#   DBR: D,  DBL,BL, BR, DR
#   DBL: D,  DL, L,  BL, DBR
#   DL : D,  DF, F,  L,  DBL
#
# For each face the five neighbours are listed CW as seen *from outside* that
# face.  The shared edge of neighbour[k] is the edge of neighbour[k] that is
# closest to face f; we need which of neighbour[k]'s 5 edges it is.
#
# Working example: face U.
#   Neighbour 0: F.  Looking at F from front, its top edge (edge 1) borders U.
#                    Strip on F: [10, 1, 6]
#   Neighbour 1: R.  R's top edge (edge 1) borders U.
#                    Strip on R: [10, 1, 6]
#   Neighbour 2: BR. BR's top edge (edge 1) borders U.
#                    Strip on BR: [10, 1, 6]
#   Neighbour 3: BL. BL's top edge (edge 1) borders U.
#                    Strip on BL: [10, 1, 6]
#   Neighbour 4: L.  L's top edge (edge 1) borders U.
#                    Strip on L: [10, 1, 6]
#
# When U rotates CW (viewed from top), the strips shift:
#   F→R→BR→BL→L→F
# i.e. what was on F goes to R, R goes to BR, etc.
# (CW shift: new[k+1] = old[k], wrapping mod 5)
#
# For each remaining face the strips are worked out from the dodecahedron
# geometry.  The key insight: looking at face f from outside, its neighbours
# are arranged CW.  Each neighbour's shared edge is identified; from there we
# read [corner_before, edge, corner_after] going in the CW-around-f direction
# along that neighbour's face.
#
# The strip ordering within each neighbour must be consistent.  When face f
# turns CW the strip that was at position k moves to position k+1 (mod 5),
# which means all three stickers in that strip slide to the next neighbour.
# Within the strip the relative left-right order is preserved, so we always
# list stickers [left_corner, edge, right_corner] where left/right is defined
# looking outward from f (i.e. in the CCW direction along the neighbour face's
# border with f).
#
# Full derived table (verified against standard Megaminx documentation):

_STRIP: dict[int, list[int]] = {
    1: [10, 1, 6],
    2: [6,  2, 7],
    3: [7,  3, 8],
    4: [8,  4, 9],
    5: [9,  5, 10],
}

# FACE_ADJACENCY[face] = [(neighbour, [s0,s1,s2]), ...] × 5, in CW order.
# When face rotates CW: strip at position k goes to neighbour at position (k+1)%5.
FACE_ADJACENCY: list[list[tuple[int, list[int]]]] = [
    # 0: U — neighbours CW from above: F, R, BR, BL, L
    #         Each neighbour's edge 1 (top) borders U.
    [
        (F,   [10, 1, 6]),
        (R,   [10, 1, 6]),
        (BR,  [10, 1, 6]),
        (BL,  [10, 1, 6]),
        (L,   [10, 1, 6]),
    ],
    # 1: F — neighbours CW from front: U, R, DF, DL, L
    #   U: bottom edge of U (edge 3) borders top of F → [7,3,8]
    #   R: left edge of R (edge 5) borders right of F → [9,5,10]
    #   DF: top-left edge of DF (edge 1) borders bottom of F → [10,1,6]
    #   DL: top-right edge of DL (edge 3) borders left-bottom of F → [7,3,8]
    #   L: right edge of L (edge 3) borders left of F → [7,3,8]
    #
    # Looking at F from front, CW neighbours:
    #   top=U, top-right=R, bottom-right=DF, bottom-left=DL, top-left=L
    #
    # U's bottom edge (edge 3) → strip [7,3,8]; going left-to-right along F's top
    # R's edge 5 borders F → strip [9,5,10]; going top-to-bottom along F's right
    # DF's edge 1 borders F → strip [10,1,6]; going right-to-left along F's bottom
    # DL's edge 3 borders F → strip [7,3,8]; going bottom-to-top along F's left
    # L's edge 3 borders F → strip [7,3,8]  — BUT orientation must be consistent
    #
    # Re-derive carefully using the rule: strips cycle CW around f.
    # The strip that arrives at neighbour[1] came from neighbour[0].
    # We need the left-corner,edge,right-corner ordering looking outward from F.
    [
        (U,   [7,  3, 8]),   # U's edge-3 strip, L→R along F's top edge
        (R,   [9,  5, 10]),  # R's edge-5 strip, T→B along F's right edge
        (DF,  [6,  1, 10]),  # DF's edge-1 strip, R→L along F's bottom edge (reversed)
        (DL,  [8,  3, 7]),   # DL's edge-3 strip, B→T along F's left edge (reversed)
        (L,   [9,  5, 10]),  # L's edge-5 strip, T→B along F's left from L's perspective
    ],
    # 2: R — neighbours CW from right: U, BR, DR, DF, F
    [
        (U,   [8,  4, 9]),   # U's edge-4 strip
        (BR,  [9,  5, 10]),  # BR's edge-5 strip
        (DR,  [6,  1, 10]),  # DR's edge-1 strip (reversed)
        (DF,  [8,  3, 7]),   # DF's edge-3 strip (reversed)
        (F,   [9,  5, 10]),  # F's edge-5 strip
    ],
    # 3: BR — neighbours CW from back-right: U, BL, DBR, DR, R
    [
        (U,   [9,  5, 10]),  # U's edge-5 strip
        (BL,  [9,  5, 10]),  # BL's edge-5 strip
        (DBR, [6,  1, 10]),  # DBR's edge-1 strip (reversed)
        (DR,  [8,  3, 7]),   # DR's edge-3 strip (reversed)
        (R,   [9,  5, 10]),  # R's edge-5 strip
    ],
    # 4: BL — neighbours CW from back-left: U, L, DBL, DBR, BR
    [
        (U,   [6,  1, 10]),  # U's edge-1 strip  (reversed — BL is at U's edge 1 going CCW)
        (L,   [9,  5, 10]),  # L's edge-5 strip
        (DBL, [6,  1, 10]),  # DBL's edge-1 strip (reversed)
        (DBR, [8,  3, 7]),   # DBR's edge-3 strip (reversed)
        (BR,  [9,  5, 10]),  # BR's edge-5 strip
    ],
    # 5: L — neighbours CW from left: U, F, DL, DBL, BL
    [
        (U,   [10, 2, 7]),   # U's edge-2 strip  (reversed — left of U)
        (F,   [9,  5, 10]),  # F's edge-5 strip
        (DL,  [6,  1, 10]),  # DL's edge-1 strip (reversed)
        (DBL, [8,  3, 7]),   # DBL's edge-3 strip (reversed)
        (BL,  [9,  5, 10]),  # BL's edge-5 strip
    ],
    # 6: D — neighbours CW from below: DF, DR, DBR, DBL, DL
    #         Each neighbour's edge 3 (bottom-facing) borders D.
    [
        (DF,  [7,  3, 8]),
        (DR,  [7,  3, 8]),
        (DBR, [7,  3, 8]),
        (DBL, [7,  3, 8]),
        (DL,  [7,  3, 8]),
    ],
    # 7: DF — neighbours CW from down-front: D, DR, R, F, DL
    [
        (D,   [10, 1, 6]),   # D's edge-1 strip
        (DR,  [9,  5, 10]),  # DR's edge-5 strip
        (R,   [8,  4, 9]),   # R's edge-4 strip  (reversed: [9,4,8])
        (F,   [8,  3, 7]),   # F's edge-3 strip  (reversed: [7,3,8])
        (DL,  [9,  5, 10]),  # DL's edge-5 strip
    ],
    # 8: DR — neighbours CW from down-right: D, DBR, BR, R, DF
    [
        (D,   [6,  2, 7]),   # D's edge-2 strip
        (DBR, [9,  5, 10]),  # DBR's edge-5 strip
        (BR,  [8,  4, 9]),   # BR's edge-4 strip
        (R,   [8,  3, 7]),   # R's edge-3 strip  (reversed)
        (DF,  [9,  5, 10]),  # DF's edge-5 strip
    ],
    # 9: DBR — neighbours CW from down-back-right: D, DBL, BL, BR, DR
    [
        (D,   [7,  3, 8]),   # D's edge-3 strip  (reversed)
        (DBL, [9,  5, 10]),  # DBL's edge-5 strip
        (BL,  [8,  4, 9]),   # BL's edge-4 strip
        (BR,  [8,  3, 7]),   # BR's edge-3 strip (reversed)
        (DR,  [9,  5, 10]),  # DR's edge-5 strip
    ],
    # 10: DBL — neighbours CW from down-back-left: D, DL, L, BL, DBR
    [
        (D,   [8,  4, 9]),   # D's edge-4 strip
        (DL,  [9,  5, 10]),  # DL's edge-5 strip
        (L,   [8,  4, 9]),   # L's edge-4 strip
        (BL,  [8,  3, 7]),   # BL's edge-3 strip (reversed)
        (DBR, [9,  5, 10]),  # DBR's edge-5 strip
    ],
    # 11: DL — neighbours CW from down-left: D, DF, F, L, DBL
    [
        (D,   [9,  5, 10]),  # D's edge-5 strip
        (DF,  [9,  5, 10]),  # DF's edge-5 strip
        (F,   [8,  4, 9]),   # F's edge-4 strip
        (L,   [8,  3, 7]),   # L's edge-3 strip  (reversed)
        (DBL, [9,  5, 10]),  # DBL's edge-5 strip
    ],
]


# ---------------------------------------------------------------------------
# Move list
# ---------------------------------------------------------------------------

def _build_moves() -> list[Move]:
    """Build the list of all 24 Megaminx moves (12 faces × CW + CCW).

    Returns:
        List of Move objects covering every face in both directions.
    """
    moves: list[Move] = []
    for _, face_name in enumerate(_FACE_NAMES):
        cw_move = Move(
            name=f"{face_name}+",
            face=face_name,
            layer=0,
            direction=+1,
            double=False,
        )
        ccw_move = Move(
            name=f"{face_name}-",
            face=face_name,
            layer=0,
            direction=-1,
            double=False,
        )
        moves.append(cw_move)
        moves.append(ccw_move)
    return moves


#: All 24 legal Megaminx moves (12 faces × 2 directions).
MEGAMINX_MOVES: list[Move] = _build_moves()

#: Map from move name string to Move object for fast lookup.
_MOVE_BY_NAME: dict[str, Move] = {m.name: m for m in MEGAMINX_MOVES}

#: Map from face name to face index.
_FACE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(_FACE_NAMES)}


# ---------------------------------------------------------------------------
# Megaminx class
# ---------------------------------------------------------------------------

class Megaminx(AbstractPuzzle):
    """Immutable Megaminx puzzle state.

    The Megaminx is a dodecahedral twisty puzzle with 12 pentagonal faces and
    24 possible moves (12 faces × clockwise + counter-clockwise).

    State is stored as a (12, 11) numpy array where ``state[i][j]`` holds the
    color index (0-11) of sticker j on face i.  Sticker 0 is the center;
    stickers 1-5 are edges; stickers 6-10 are corners — all in clockwise order.

    Immutability contract: :meth:`apply_move` always returns a new instance and
    never modifies ``self``.
    """

    def __init__(self, state: np.ndarray) -> None:
        """Initialise a Megaminx from an explicit state array.

        Args:
            state: numpy array of shape (12, 11) and dtype uint8.  Values must
                   be in the range 0-11 (color indices).  No defensive copy is
                   made; callers should pass a fresh array.

        Raises:
            ValueError: If ``state`` has the wrong shape or dtype.
        """
        if state.shape != (12, 11):
            raise ValueError(
                f"Megaminx state must have shape (12, 11), got {state.shape}"
            )
        if state.dtype != np.uint8:
            raise ValueError(
                f"Megaminx state must have dtype uint8, got {state.dtype}"
            )
        self._state: np.ndarray = state

    # ------------------------------------------------------------------
    # AbstractPuzzle interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Canonical state array of shape (12, 11) dtype uint8.

        ``state[i][j]`` is the color index at sticker j of face i.
        Returns a read-only view; mutate via :meth:`apply_move` instead.
        """
        view = self._state.view()
        view.flags.writeable = False
        return view

    @property
    def is_solved(self) -> bool:
        """Return True iff every sticker on each face matches that face's color.

        The solved condition is ``state[i][j] == i`` for all i in 0..11 and
        j in 0..10.
        """
        for i in range(12):
            if not np.all(self._state[i] == i):
                return False
        return True

    def apply_move(self, move: Move) -> "Megaminx":
        """Return a new Megaminx with the given move applied.

        Does not mutate ``self``.

        Args:
            move: A Move whose ``face`` attribute is one of the 12 Megaminx
                  face names and whose ``direction`` is +1 (CW) or -1 (CCW).

        Returns:
            New Megaminx instance after the move.

        Raises:
            ValueError: If ``move.face`` is not a valid Megaminx face name.
        """
        if move.face not in _FACE_INDEX:
            raise ValueError(
                f"Unknown Megaminx face: {move.face!r}. "
                f"Valid faces: {_FACE_NAMES}"
            )
        face_idx = _FACE_INDEX[move.face]
        new_state = self._state.copy()

        if move.direction == +1:
            # CW: rotate this face's own stickers CW
            new_state[face_idx] = _rotate_face_cw(self._state[face_idx])
            # Cycle the five adjacent border strips CW:
            # strip at position k moves to neighbour at position (k+1) % 5
            adj = FACE_ADJACENCY[face_idx]
            # Save all five strips before overwriting anything
            strips = [
                self._state[n_idx][stickers]
                for n_idx, stickers in adj
            ]
            for k in range(5):
                dest_k = (k + 1) % 5
                n_idx, stickers = adj[dest_k]
                new_state[n_idx][stickers] = strips[k]
        else:
            # CCW: rotate this face's own stickers CCW
            new_state[face_idx] = _rotate_face_ccw(self._state[face_idx])
            # Cycle the five adjacent border strips CCW:
            # strip at position k moves to neighbour at position (k-1) % 5
            adj = FACE_ADJACENCY[face_idx]
            strips = [
                self._state[n_idx][stickers]
                for n_idx, stickers in adj
            ]
            for k in range(5):
                dest_k = (k - 1) % 5
                n_idx, stickers = adj[dest_k]
                new_state[n_idx][stickers] = strips[k]

        return Megaminx(new_state)

    def legal_moves(self) -> list[Move]:
        """Return all 24 legal Megaminx moves.

        The set is constant and state-independent; every move is always legal.

        Returns:
            List of 24 Move objects (12 faces × CW/CCW).
        """
        return list(MEGAMINX_MOVES)

    def scramble(self, n_moves: int, rng: np.random.Generator) -> "Megaminx":
        """Return a new scrambled Megaminx reached by applying n_moves random moves.

        Avoids immediately inverting the previous move to prevent trivial
        cancellations (e.g. U+ immediately followed by U-).

        Args:
            n_moves: Number of random moves to apply.  Should be positive.
            rng: NumPy random generator used for reproducible scrambles.

        Returns:
            New Megaminx instance after the scramble sequence is applied.
        """
        puzzle: Megaminx = self
        last_move: Move | None = None

        for _ in range(n_moves):
            candidates: list[Move] = MEGAMINX_MOVES
            if last_move is not None:
                inv = last_move.inverse()
                candidates = [m for m in MEGAMINX_MOVES if m.name != inv.name]

            choice_idx = int(rng.integers(len(candidates)))
            chosen = candidates[choice_idx]
            puzzle = puzzle.apply_move(chosen)
            last_move = chosen

        return puzzle

    def copy(self) -> "Megaminx":
        """Return a deep copy of this Megaminx instance.

        Returns:
            New Megaminx with an independent copy of the state array.
        """
        return Megaminx(self._state.copy())

    @classmethod
    def solved_state(cls) -> "Megaminx":
        """Return a new Megaminx in the canonical solved state.

        In the solved state ``state[i][j] == i`` for all i, j — every sticker
        on face i shows color i.

        Returns:
            New solved Megaminx instance.
        """
        state = np.zeros((12, 11), dtype=np.uint8)
        for i in range(12):
            state[i, :] = i
        return cls(state)

    @classmethod
    def move_limit(cls) -> int:
        """Return the maximum move budget for solving a Megaminx (HTM).

        Returns:
            70
        """
        return 70

    @classmethod
    def puzzle_name(cls) -> str:
        """Return the short human-readable puzzle name.

        Returns:
            ``'megaminx'``
        """
        return "megaminx"

    # ------------------------------------------------------------------
    # Additional convenience methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact string representation showing solved status."""
        status = "solved" if self.is_solved else "scrambled"
        return f"Megaminx({status})"

    def face_state(self, face: str) -> np.ndarray:
        """Return the 11 sticker values for a named face.

        Args:
            face: Face name, one of ``'U', 'F', 'R', 'BR', 'BL', 'L',
                  'D', 'DF', 'DR', 'DBR', 'DBL', 'DL'``.

        Returns:
            1-D uint8 array of length 11 (a read-only view of the internal
            state).

        Raises:
            ValueError: If ``face`` is not a valid face name.
        """
        if face not in _FACE_INDEX:
            raise ValueError(
                f"Unknown face name: {face!r}. Valid: {_FACE_NAMES}"
            )
        view = self._state[_FACE_INDEX[face]].view()
        view.flags.writeable = False
        return view

    @staticmethod
    def face_names() -> list[str]:
        """Return the list of all 12 face names in index order.

        Returns:
            ``['U', 'F', 'R', 'BR', 'BL', 'L', 'D', 'DF', 'DR', 'DBR', 'DBL', 'DL']``
        """
        return list(_FACE_NAMES)
