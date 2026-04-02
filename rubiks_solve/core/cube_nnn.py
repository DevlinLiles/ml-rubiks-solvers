"""
NxN Rubik's cube state representation and move application.

The cube state is a numpy array of shape (6, n, n) with dtype uint8.
Face indices and their solved colors:
  0 = U (Up)    → White  (0)
  1 = D (Down)  → Yellow (1)
  2 = F (Front) → Green  (2)
  3 = B (Back)  → Blue   (3)
  4 = L (Left)  → Orange (4)
  5 = R (Right) → Red    (5)

All operations are IMMUTABLE — apply_move returns a new CubeNNN instance.

Coordinate conventions (standard "white top, green front"):
  - U face: viewed from above, row 0 is the back, row n-1 is the front.
    Column 0 is left, column n-1 is right.
  - F face: viewed from front, row 0 is top, row n-1 is bottom.
    Column 0 is left, column n-1 is right.
  - All other faces follow naturally from this orientation.

Layer depth (0-indexed from the outer face):
  - Layer 0: the physical face stickers + the adjacent strip on the four neighbouring faces.
  - Layer k > 0: only the adjacent strips (no sticker face to rotate).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.core.moves import get_moves

# ---------------------------------------------------------------------------
# Face index constants
# ---------------------------------------------------------------------------
_U, _D, _F, _B, _L, _R = 0, 1, 2, 3, 4, 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_face_cw(face: np.ndarray) -> np.ndarray:
    """Return a 90-degree clockwise rotation of a 2-D face array.

    Args:
        face: 2-D numpy array representing the stickers on a single face.

    Returns:
        New 2-D array with stickers rotated 90 degrees clockwise.
    """
    return np.rot90(face, k=-1)


def _rotate_face_ccw(face: np.ndarray) -> np.ndarray:
    """Return a 90-degree counter-clockwise rotation of a 2-D face array.

    Args:
        face: 2-D numpy array representing the stickers on a single face.

    Returns:
        New 2-D array with stickers rotated 90 degrees counter-clockwise.
    """
    return np.rot90(face, k=1)


def _rotate_face_180(face: np.ndarray) -> np.ndarray:
    """Return a 180-degree rotation of a 2-D face array.

    Args:
        face: 2-D numpy array representing the stickers on a single face.

    Returns:
        New 2-D array with stickers rotated 180 degrees.
    """
    return np.rot90(face, k=2)


# ---------------------------------------------------------------------------
# Strip extraction / insertion helpers
# ---------------------------------------------------------------------------

def _apply_u_cw(state: np.ndarray, layer: int) -> np.ndarray:
    """Apply a clockwise U-face move at the given layer depth.

    U CW (viewed from top):
      new_F[layer, :] = old_L[layer, :]
      new_R[layer, :] = old_F[layer, :]
      new_B[layer, :] = old_R[layer, :]
      new_L[layer, :] = old_B[layer, :]

    Args:
        state: Cube state array of shape (6, n, n).
        layer: Layer depth (0 = outer face, 1 = first inner slice, …).

    Returns:
        New state array with the move applied.
    """
    s = state.copy()
    if layer == 0:
        s[_U] = _rotate_face_cw(state[_U])
    f_row = state[_F][layer, :].copy()
    r_row = state[_R][layer, :].copy()
    b_row = state[_B][layer, :].copy()
    l_row = state[_L][layer, :].copy()
    s[_F][layer, :] = l_row
    s[_R][layer, :] = f_row
    s[_B][layer, :] = r_row
    s[_L][layer, :] = b_row
    return s


def _apply_d_cw(state: np.ndarray, layer: int) -> np.ndarray:
    """Apply a clockwise D-face move at the given layer depth.

    D CW (viewed from bottom — row index from bottom means n-1-layer from top):
      new_F[-(layer+1), :] = old_R[-(layer+1), :]
      new_L[-(layer+1), :] = old_F[-(layer+1), :]
      new_B[-(layer+1), :] = old_L[-(layer+1), :]
      new_R[-(layer+1), :] = old_B[-(layer+1), :]

    Args:
        state: Cube state array of shape (6, n, n).
        layer: Layer depth (0 = outer face, 1 = first inner slice, …).

    Returns:
        New state array with the move applied.
    """
    s = state.copy()
    idx = -(layer + 1)
    if layer == 0:
        s[_D] = _rotate_face_cw(state[_D])
    f_row = state[_F][idx, :].copy()
    r_row = state[_R][idx, :].copy()
    b_row = state[_B][idx, :].copy()
    l_row = state[_L][idx, :].copy()
    s[_F][idx, :] = r_row
    s[_L][idx, :] = f_row
    s[_B][idx, :] = l_row
    s[_R][idx, :] = b_row
    return s


def _apply_f_cw(state: np.ndarray, layer: int) -> np.ndarray:
    """Apply a clockwise F-face move at the given layer depth.

    F CW (viewed from front):
      new_U[-(layer+1), :]  = old_L[:, -(layer+1)] reversed
      new_R[:, layer]       = old_U[-(layer+1), :]
      new_D[layer, :]       = old_R[:, layer] reversed
      new_L[:, -(layer+1)]  = old_D[layer, :]

    Args:
        state: Cube state array of shape (6, n, n).
        layer: Layer depth (0 = outer face, 1 = first inner slice, …).

    Returns:
        New state array with the move applied.
    """
    s = state.copy()
    if layer == 0:
        s[_F] = _rotate_face_cw(state[_F])
    u_row = state[_U][-(layer + 1), :].copy()
    r_col = state[_R][:, layer].copy()
    d_row = state[_D][layer, :].copy()
    l_col = state[_L][:, -(layer + 1)].copy()
    s[_U][-(layer + 1), :] = l_col[::-1]
    s[_R][:, layer]        = u_row
    s[_D][layer, :]        = r_col[::-1]
    s[_L][:, -(layer + 1)] = d_row
    return s


def _apply_b_cw(state: np.ndarray, layer: int) -> np.ndarray:
    """Apply a clockwise B-face move at the given layer depth.

    B CW (viewed from back):
      new_U[layer, :]       = old_R[:, -(layer+1)] reversed
      new_L[:, layer]       = old_U[layer, :]
      new_D[-(layer+1), :]  = old_L[:, layer] reversed
      new_R[:, -(layer+1)]  = old_D[-(layer+1), :]

    Args:
        state: Cube state array of shape (6, n, n).
        layer: Layer depth (0 = outer face, 1 = first inner slice, …).

    Returns:
        New state array with the move applied.
    """
    s = state.copy()
    if layer == 0:
        s[_B] = _rotate_face_cw(state[_B])
    u_row = state[_U][layer, :].copy()
    r_col = state[_R][:, -(layer + 1)].copy()
    d_row = state[_D][-(layer + 1), :].copy()
    l_col = state[_L][:, layer].copy()
    s[_U][layer, :]        = r_col[::-1]
    s[_L][:, layer]        = u_row
    s[_D][-(layer + 1), :] = l_col[::-1]
    s[_R][:, -(layer + 1)] = d_row
    return s


def _apply_l_cw(state: np.ndarray, layer: int) -> np.ndarray:
    """Apply a clockwise L-face move at the given layer depth.

    L CW (viewed from left):
      new_U[:, layer]       = old_B[:, -(layer+1)] reversed
      new_F[:, layer]       = old_U[:, layer]
      new_D[:, layer]       = old_F[:, layer]
      new_B[:, -(layer+1)]  = old_D[:, layer] reversed

    Args:
        state: Cube state array of shape (6, n, n).
        layer: Layer depth (0 = outer face, 1 = first inner slice, …).

    Returns:
        New state array with the move applied.
    """
    s = state.copy()
    if layer == 0:
        s[_L] = _rotate_face_cw(state[_L])
    u_col = state[_U][:, layer].copy()
    f_col = state[_F][:, layer].copy()
    d_col = state[_D][:, layer].copy()
    b_col = state[_B][:, -(layer + 1)].copy()
    s[_U][:, layer]        = b_col[::-1]
    s[_F][:, layer]        = u_col
    s[_D][:, layer]        = f_col
    s[_B][:, -(layer + 1)] = d_col[::-1]
    return s


def _apply_r_cw(state: np.ndarray, layer: int) -> np.ndarray:
    """Apply a clockwise R-face move at the given layer depth.

    R CW (viewed from right):
      new_U[:, -(layer+1)]  = old_F[:, -(layer+1)]
      new_B[:, layer]       = old_U[:, -(layer+1)] reversed
      new_D[:, -(layer+1)]  = old_B[:, layer] reversed
      new_F[:, -(layer+1)]  = old_D[:, -(layer+1)]

    Args:
        state: Cube state array of shape (6, n, n).
        layer: Layer depth (0 = outer face, 1 = first inner slice, …).

    Returns:
        New state array with the move applied.
    """
    s = state.copy()
    if layer == 0:
        s[_R] = _rotate_face_cw(state[_R])
    u_col = state[_U][:, -(layer + 1)].copy()
    f_col = state[_F][:, -(layer + 1)].copy()
    d_col = state[_D][:, -(layer + 1)].copy()
    b_col = state[_B][:, layer].copy()
    s[_U][:, -(layer + 1)] = f_col
    s[_B][:, layer]        = u_col[::-1]
    s[_D][:, -(layer + 1)] = b_col[::-1]
    s[_F][:, -(layer + 1)] = d_col
    return s


# Map face name → CW applicator function
_CW_APPLICATORS = {
    "U": _apply_u_cw,
    "D": _apply_d_cw,
    "F": _apply_f_cw,
    "B": _apply_b_cw,
    "L": _apply_l_cw,
    "R": _apply_r_cw,
}


def _apply_rotation(state: np.ndarray, face: str, direction: int, double: bool) -> np.ndarray:
    """Apply a whole-cube rotation (layer == -1).

    Whole-cube rotations are expressed in terms of face turns on all layers.
    x = R all-layer, y = U all-layer, z = F all-layer.

    Args:
        state: Cube state array of shape (6, n, n).
        face: The axis face ("R" for x, "U" for y, "F" for z).
        direction: +1 CW, -1 CCW.
        double: True for 180-degree turn.

    Returns:
        New state array with the whole-cube rotation applied.
    """
    n = state.shape[1]
    applicator = _CW_APPLICATORS[face]
    result = state
    # Apply the CW move on every layer
    if double:
        # Apply CW twice
        for _ in range(2):
            tmp = result
            for layer in range(n // 2 + n % 2):
                tmp = applicator(tmp, layer)
            result = tmp
    elif direction == +1:
        for layer in range(n // 2 + n % 2):
            result = applicator(result, layer)
    else:
        # CCW = three CW
        for _ in range(3):
            tmp = result
            for layer in range(n // 2 + n % 2):
                tmp = applicator(tmp, layer)
            result = tmp
    return result


# ---------------------------------------------------------------------------
# CubeNNN class
# ---------------------------------------------------------------------------

class CubeNNN(AbstractPuzzle):
    """Generic NxN Rubik's cube with numpy-backed immutable state.

    State is a numpy array of shape (6, n, n) with dtype uint8.
    Color encoding: 0=White(U), 1=Yellow(D), 2=Green(F), 3=Blue(B),
                    4=Orange(L), 5=Red(R).

    All move applications return new instances — self is never mutated.
    """

    def __init__(self, n: int, state: Optional[np.ndarray] = None) -> None:
        """Initialise the cube.

        Args:
            n: Cube dimension (e.g. 3 for a 3x3).
            state: Optional (6, n, n) uint8 array.  If None, a solved state
                   is generated automatically.

        Raises:
            ValueError: If the provided state has the wrong shape or dtype.
        """
        self._n = n
        if state is None:
            self._state: np.ndarray = self._make_solved_state(n)
        else:
            if state.shape != (6, n, n):
                raise ValueError(
                    f"Expected state shape (6, {n}, {n}), got {state.shape}"
                )
            self._state = state.astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_instance(self, new_state: np.ndarray) -> "CubeNNN":
        """Create a new instance of the same class with the given state.

        Subclasses whose ``__init__`` does not accept ``n`` as the first
        positional argument should override this method.  The default
        implementation passes ``(self._n, new_state)`` to the constructor,
        which is correct for ``CubeNNN`` itself.  Fixed-size subclasses
        (``Cube2x2`` etc.) override ``__init__`` to accept only ``state``,
        so they also override this method.

        Args:
            new_state: The (6, n, n) uint8 array for the new instance.

        Returns:
            New puzzle instance of the same type with ``new_state``.
        """
        return self.__class__(self._n, new_state)

    @staticmethod
    def _make_solved_state(n: int) -> np.ndarray:
        """Return a fresh solved-state array for an NxN cube.

        Args:
            n: Cube dimension.

        Returns:
            Array of shape (6, n, n) where face i has all cells equal to i.
        """
        state = np.empty((6, n, n), dtype=np.uint8)
        for i in range(6):
            state[i] = i
        return state

    # ------------------------------------------------------------------
    # AbstractPuzzle interface
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Cube dimension (e.g. 3 for a 3x3)."""
        return self._n

    @property
    def state(self) -> np.ndarray:
        """Canonical state array of shape (6, n, n), dtype uint8.

        Returns a copy so callers cannot accidentally mutate internal state.
        """
        return self._state.copy()

    @property
    def is_solved(self) -> bool:
        """True if every face has all stickers of one uniform color.

        Returns:
            Boolean indicating whether the cube is in its solved state.
        """
        for i in range(6):
            if not np.all(self._state[i] == i):
                return False
        return True

    def apply_move(self, move: Move) -> "CubeNNN":
        """Return a new CubeNNN with the given move applied.

        Handles whole-cube rotations (layer == -1) as well as single-layer
        face turns and inner-slice turns.

        Args:
            move: The move to apply.

        Returns:
            New CubeNNN instance with the move applied.

        Raises:
            ValueError: If the move references an unsupported face or layer.
        """
        state = self._state  # read-only reference; every helper copies

        # Whole-cube rotation
        if move.layer == -1:
            new_state = _apply_rotation(state, move.face, move.direction, move.double)
            return self._new_instance(new_state)

        if move.face not in _CW_APPLICATORS:
            raise ValueError(f"Unknown face: {move.face!r}")

        applicator = _CW_APPLICATORS[move.face]

        if move.double:
            # Two 90-degree CW turns
            new_state = applicator(state, move.layer)
            new_state = applicator(new_state, move.layer)
        elif move.direction == +1:
            new_state = applicator(state, move.layer)
        else:
            # CCW = three CW turns
            new_state = applicator(state, move.layer)
            new_state = applicator(new_state, move.layer)
            new_state = applicator(new_state, move.layer)

        return self._new_instance(new_state)

    def legal_moves(self) -> list[Move]:
        """Return all legal moves for this cube size.

        Returns:
            List of Move objects for this cube's dimension.
        """
        return get_moves(self._n)

    def scramble(self, n_moves: int, rng: np.random.Generator) -> "CubeNNN":
        """Return a new cube reached by applying n_moves random legal moves.

        Avoids trivially inverting the immediately preceding move.

        Args:
            n_moves: Number of random moves to apply.
            rng: NumPy random generator for reproducible scrambles.

        Returns:
            New CubeNNN instance in the scrambled state.
        """
        moves = self.legal_moves()
        cube: CubeNNN = self
        last_move: Optional[Move] = None
        for _ in range(n_moves):
            candidates = [m for m in moves if last_move is None or m != last_move.inverse()]
            idx = rng.integers(len(candidates))
            chosen = candidates[idx]
            cube = cube.apply_move(chosen)
            last_move = chosen
        return cube

    def copy(self) -> "CubeNNN":
        """Return a deep copy of this cube.

        Returns:
            New CubeNNN instance with identical state.
        """
        return self._new_instance(self._state.copy())

    @classmethod
    def solved_state(cls) -> "CubeNNN":
        """Return a new instance in the solved state.

        Subclasses with a fixed n override this; the base class requires
        that _DEFAULT_N is set or raises.

        Returns:
            New CubeNNN in the canonical solved state.

        Raises:
            AttributeError: If the subclass does not define _DEFAULT_N.
        """
        n = getattr(cls, "_DEFAULT_N", None)
        if n is None:
            raise AttributeError(
                "CubeNNN.solved_state() requires _DEFAULT_N to be set on the subclass."
            )
        return cls(n)

    @classmethod
    def move_limit(cls) -> int:
        """Maximum move budget for solving in Half Turn Metric.

        Subclasses must override this.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError("Subclasses must implement move_limit().")

    @classmethod
    def puzzle_name(cls) -> str:
        """Human-readable puzzle name.

        Subclasses must override this.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError("Subclasses must implement puzzle_name().")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self._n}, solved={self.is_solved})"
