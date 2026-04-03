"""Cubie-based state encoder for Rubik's cubes.

Encodes the state as cubie positions and orientations using one-hot vectors,
which is more compact than raw one-hot encoding and preserves the underlying
group-theoretic structure of the puzzle.

Fallback: puzzles other than 2x2 and 3x3 (4x4, 5x5, Megaminx) delegate to
OneHotEncoder because their piece structures are too complex for simple cubie
extraction from a raw facelet array.
"""
from __future__ import annotations

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.encoding.one_hot import OneHotEncoder

# ---------------------------------------------------------------------------
# Sticker coordinates for 3x3 corners and edges.
#
# Each corner is defined by three (face, row, col) triplets ordered by the
# standard Kociemba convention: U-face sticker comes first, then the
# clockwise-adjacent sticker when looking at the corner from outside.
#
# Face indices: U=0, D=1, F=2, B=3, L=4, R=5
# ---------------------------------------------------------------------------

# Corner sticker triples  (face, row, col) – 8 corners, 3 stickers each.
# Corners in standard order: URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB
_CORNER_STICKERS: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = [
    # 0 URF: U(2,2), R(0,0), F(0,2)
    ((0, 2, 2), (5, 0, 0), (2, 0, 2)),
    # 1 UFL: U(2,0), F(0,0), L(0,2)
    ((0, 2, 0), (2, 0, 0), (4, 0, 2)),
    # 2 ULB: U(0,0), L(0,0), B(0,2)
    ((0, 0, 0), (4, 0, 0), (3, 0, 2)),
    # 3 UBR: U(0,2), B(0,0), R(0,2)
    ((0, 0, 2), (3, 0, 0), (5, 0, 2)),
    # 4 DFR: D(0,2), F(2,2), R(2,0)
    ((1, 0, 2), (2, 2, 2), (5, 2, 0)),
    # 5 DLF: D(0,0), L(2,2), F(2,0)
    ((1, 0, 0), (4, 2, 2), (2, 2, 0)),
    # 6 DBL: D(2,0), B(2,2), L(2,0)
    ((1, 2, 0), (3, 2, 2), (4, 2, 0)),
    # 7 DRB: D(2,2), R(2,2), B(2,0)
    ((1, 2, 2), (5, 2, 2), (3, 2, 0)),
]

# Edge sticker pairs  (face, row, col) – 12 edges, 2 stickers each.
# Edges in standard order: UR, UF, UL, UB, DR, DF, DL, DB, FR, FL, BL, BR
_EDGE_STICKERS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    # 0 UR: U(1,2), R(0,1)
    ((0, 1, 2), (5, 0, 1)),
    # 1 UF: U(2,1), F(0,1)
    ((0, 2, 1), (2, 0, 1)),
    # 2 UL: U(1,0), L(0,1)
    ((0, 1, 0), (4, 0, 1)),
    # 3 UB: U(0,1), B(0,1)
    ((0, 0, 1), (3, 0, 1)),
    # 4 DR: D(1,2), R(2,1)
    ((1, 1, 2), (5, 2, 1)),
    # 5 DF: D(0,1), F(2,1)
    ((1, 0, 1), (2, 2, 1)),
    # 6 DL: D(1,0), L(2,1)
    ((1, 1, 0), (4, 2, 1)),
    # 7 DB: D(2,1), B(2,1)
    ((1, 2, 1), (3, 2, 1)),
    # 8 FR: F(1,2), R(1,0)
    ((2, 1, 2), (5, 1, 0)),
    # 9 FL: F(1,0), L(1,2)
    ((2, 1, 0), (4, 1, 2)),
    # 10 BL: B(1,2), L(1,0)
    ((3, 1, 2), (4, 1, 0)),
    # 11 BR: B(1,0), R(1,2)
    ((3, 1, 0), (5, 1, 2)),
]

# Solved color for each face: face index → color index (0-5).
# In the solved state, face f has all stickers with color f.
_FACE_COLOR = [0, 1, 2, 3, 4, 5]  # U=0,D=1,F=2,B=3,L=4,R=5

# For each corner in the solved position, record which color appears on
# each of its three sticker slots so we can identify permutation & orientation.
# corner_solved_colors[i] = (c0, c1, c2) where c0 is the U/D sticker color.
# Color of each sticker = face index of that sticker (since face f has color f).
_CORNER_SOLVED_COLORS: list[tuple[int, int, int]] = [
    (_FACE_COLOR[a[0]], _FACE_COLOR[b[0]], _FACE_COLOR[c[0]])
    for a, b, c in _CORNER_STICKERS
]

_EDGE_SOLVED_COLORS: list[tuple[int, int]] = [
    (_FACE_COLOR[a[0]], _FACE_COLOR[b[0]])
    for (a, b) in _EDGE_STICKERS
]

# Build lookup: frozenset of two colors → solved edge index
_EDGE_COLOR_SET_TO_IDX: dict[frozenset, int] = {
    frozenset(colors): idx for idx, colors in enumerate(_EDGE_SOLVED_COLORS)
}

# Build lookup: frozenset of three colors → solved corner index
_CORNER_COLOR_SET_TO_IDX: dict[frozenset, int] = {
    frozenset(colors): idx for idx, colors in enumerate(_CORNER_SOLVED_COLORS)
}


def _extract_3x3_cubies(
    state: np.ndarray,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Extract corner and edge cubie positions and orientations from a 3x3 state.

    Parameters
    ----------
    state:
        Raw state array of shape (6, 3, 3) with uint8 color indices 0-5.

    Returns
    -------
    corner_pos : list[int]
        Length-8 list. ``corner_pos[slot]`` = which solved corner is here.
    corner_orient : list[int]
        Length-8 list. Orientation 0-2 of the corner in that slot.
    edge_pos : list[int]
        Length-12 list. ``edge_pos[slot]`` = which solved edge is here.
    edge_orient : list[int]
        Length-12 list. Orientation 0-1 of the edge in that slot.
    """
    corner_pos: list[int] = []
    corner_orient: list[int] = []

    for _slot, (sa, sb, sc) in enumerate(_CORNER_STICKERS):
        ca = int(state[sa[0], sa[1], sa[2]])
        cb = int(state[sb[0], sb[1], sb[2]])
        cc = int(state[sc[0], sc[1], sc[2]])

        piece_idx = _CORNER_COLOR_SET_TO_IDX.get(frozenset([ca, cb, cc]), 0)
        corner_pos.append(piece_idx)

        # Orientation: which rotation brings the U/D sticker face-color to
        # the U/D sticker slot of this corner position.
        # The first sticker in each triplet is the U or D face sticker.
        # Solved U-face color for this *piece*:
        solved_ud_color = _CORNER_SOLVED_COLORS[piece_idx][0]
        if ca == solved_ud_color:
            orient = 0
        elif cb == solved_ud_color:
            orient = 1  # twisted once CW
        else:
            orient = 2  # twisted twice CW
        corner_orient.append(orient)

    edge_pos: list[int] = []
    edge_orient: list[int] = []

    for _slot, (sa, sb) in enumerate(_EDGE_STICKERS):
        ea = int(state[sa[0], sa[1], sa[2]])
        eb = int(state[sb[0], sb[1], sb[2]])

        piece_idx = _EDGE_COLOR_SET_TO_IDX.get(frozenset([ea, eb]), 0)
        edge_pos.append(piece_idx)

        # Orientation: 0 if the first sticker matches the solved first-sticker
        # color of the identified piece, 1 otherwise.
        solved_first_color = _EDGE_SOLVED_COLORS[piece_idx][0]
        orient = 0 if ea == solved_first_color else 1
        edge_orient.append(orient)

    return corner_pos, corner_orient, edge_pos, edge_orient


def _one_hot(index: int, size: int) -> np.ndarray:
    """Return a one-hot float32 vector of length ``size`` with a 1 at ``index``."""
    v = np.zeros(size, dtype=np.float32)
    v[index] = 1.0
    return v


class CubieEncoder(AbstractStateEncoder):
    """Encodes a puzzle state using cubie positions and orientations.

    For 3x3 cubes:
        8 corners × (pos one-hot(8) + orient one-hot(3))  =  8*8 + 8*3  =  88
        12 edges  × (pos one-hot(12) + orient one-hot(2)) = 12*12 + 12*2 = 168
        Total: 256 floats.

    For 2x2 cubes:
        8 corners only (no edges): 8*8 + 8*3 = 88 floats.

    For 4x4, 5x5, and Megaminx:
        Falls back to OneHotEncoder because exact cubie extraction from raw
        facelet arrays is not feasible for these puzzle types.

    Parameters
    ----------
    puzzle_type:
        The AbstractPuzzle subclass this encoder is configured for.
    """

    _SUPPORTED_NAMES = {"2x2", "3x3"}

    def __init__(self, puzzle_type: type[AbstractPuzzle]) -> None:
        self._puzzle_type = puzzle_type
        name = puzzle_type.puzzle_name()

        if name not in self._SUPPORTED_NAMES:
            # Fall back to one-hot for complex puzzles.
            self._fallback: OneHotEncoder | None = OneHotEncoder(puzzle_type)
            self._output_shape = self._fallback.output_shape
            self._output_size_val = self._fallback.output_size
            self._name = name
            return

        self._fallback = None
        self._name = name

        if name == "3x3":
            # 8 corners: pos(8) + orient(3) each → 88
            # 12 edges:  pos(12) + orient(2) each → 168
            self._output_size_val = 8 * 8 + 8 * 3 + 12 * 12 + 12 * 2  # 256
            self._has_edges = True
        else:  # 2x2
            # 8 corners only: pos(8) + orient(3) each → 88
            self._output_size_val = 8 * 8 + 8 * 3  # 88
            self._has_edges = False

        self._output_shape = (self._output_size_val,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the encoded tensor: a 1-D tuple with the flattened size."""
        return self._output_shape

    @property
    def output_size(self) -> int:
        """Total number of float32 values in the encoded representation."""
        return self._output_size_val

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """Encode a raw state array to a float32 cubie vector.

        Parameters
        ----------
        state:
            Raw puzzle state array from ``AbstractPuzzle.state``.

        Returns
        -------
        np.ndarray
            Float32 array of length ``self.output_size``.
        """
        if self._name == "3x3":
            return self._encode_3x3(state)
        if self._name == "2x2":
            return self._encode_2x2(state)
        raise RuntimeError(f"Unexpected puzzle name for cubie encoding: {self._name!r}")

    def _encode_3x3(self, state: np.ndarray) -> np.ndarray:
        """Encode a 3x3 state array using cubie positions and orientations."""
        c_pos, c_orient, e_pos, e_orient = _extract_3x3_cubies(state)
        parts: list[np.ndarray] = []
        for i in range(8):
            parts.append(_one_hot(c_pos[i], 8))
            parts.append(_one_hot(c_orient[i], 3))
        for i in range(12):
            parts.append(_one_hot(e_pos[i], 12))
            parts.append(_one_hot(e_orient[i], 2))
        return np.concatenate(parts)

    def _encode_2x2(self, state: np.ndarray) -> np.ndarray:
        """Encode a 2x2 state using corner cubies only.

        A 2x2 cube has 8 corners with the same sticker layout as the
        corresponding corners of a 3x3. The same extraction logic applies
        (the state is shape (6, 2, 2) with values 0-5).

        For the 2x2, the corner sticker positions differ because the face
        grid is 2x2 not 3x3. We use a dedicated sticker table here.
        """
        # 2x2 corner stickers: (face, row, col)
        # Faces: U=0,D=1,F=2,B=3,L=4,R=5; grid is 2x2 (rows/cols 0-1)
        corner_stickers_2x2: list[
            tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
        ] = [
            # URF: U(1,1), R(0,0), F(0,1)
            ((0, 1, 1), (5, 0, 0), (2, 0, 1)),
            # UFL: U(1,0), F(0,0), L(0,1)
            ((0, 1, 0), (2, 0, 0), (4, 0, 1)),
            # ULB: U(0,0), L(0,0), B(0,1)
            ((0, 0, 0), (4, 0, 0), (3, 0, 1)),
            # UBR: U(0,1), B(0,0), R(0,1)
            ((0, 0, 1), (3, 0, 0), (5, 0, 1)),
            # DFR: D(0,1), F(1,1), R(1,0)
            ((1, 0, 1), (2, 1, 1), (5, 1, 0)),
            # DLF: D(0,0), L(1,1), F(1,0)
            ((1, 0, 0), (4, 1, 1), (2, 1, 0)),
            # DBL: D(1,0), B(1,1), L(1,0)
            ((1, 1, 0), (3, 1, 1), (4, 1, 0)),
            # DRB: D(1,1), R(1,1), B(1,0)
            ((1, 1, 1), (5, 1, 1), (3, 1, 0)),
        ]
        solved_colors_2x2: list[tuple[int, int, int]] = [
            (_FACE_COLOR[a[0]], _FACE_COLOR[b[0]], _FACE_COLOR[c[0]])
            for (a, b, c) in corner_stickers_2x2
        ]
        color_set_to_idx: dict[frozenset, int] = {
            frozenset(colors): idx for idx, colors in enumerate(solved_colors_2x2)
        }

        parts: list[np.ndarray] = []
        for _slot, (sa, sb, sc) in enumerate(corner_stickers_2x2):
            ca = int(state[sa[0], sa[1], sa[2]])
            cb = int(state[sb[0], sb[1], sb[2]])
            cc = int(state[sc[0], sc[1], sc[2]])
            piece_idx = color_set_to_idx.get(frozenset([ca, cb, cc]), 0)
            parts.append(_one_hot(piece_idx, 8))
            solved_ud = solved_colors_2x2[piece_idx][0]
            orient = 0 if ca == solved_ud else (1 if cb == solved_ud else 2)
            parts.append(_one_hot(orient, 3))
        return np.concatenate(parts)

    def encode(self, puzzle: AbstractPuzzle) -> np.ndarray:
        """Encode a single puzzle state as a flat float32 cubie vector.

        Parameters
        ----------
        puzzle:
            The puzzle instance to encode.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``self.output_shape``.
        """
        if self._fallback is not None:
            return self._fallback.encode(puzzle)
        return self._encode_state(puzzle.state)

    def encode_batch(self, puzzles: list[AbstractPuzzle]) -> np.ndarray:
        """Encode a list of puzzles into a 2-D float32 array.

        Parameters
        ----------
        puzzles:
            List of puzzle instances to encode.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(puzzles), self.output_size)``.
        """
        if self._fallback is not None:
            return self._fallback.encode_batch(puzzles)
        rows = [self._encode_state(p.state) for p in puzzles]
        return np.stack(rows, axis=0)
