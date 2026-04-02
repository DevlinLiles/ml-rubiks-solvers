"""One-hot state encoder for NxN cubes and Megaminx."""
from __future__ import annotations

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.encoding.base import AbstractStateEncoder


def _is_megaminx(puzzle_type: type[AbstractPuzzle]) -> bool:
    """Return True if puzzle_type is Megaminx based on puzzle_name()."""
    return puzzle_type.puzzle_name() == "megaminx"


class OneHotEncoder(AbstractStateEncoder):
    """Encodes puzzle state via one-hot vectors over the color dimension.

    NxN cubes:
        Raw state shape (6, n, n) uint8, values 0-5.
        Each cell is one-hot over 6 colors.
        Intermediate shape: (6, n, n, 6) → flattened to (6*n*n*6,).
        Example: 3x3 → 324 floats.

    Megaminx:
        Raw state shape (12, 11) uint8, values 0-11.
        Each cell is one-hot over 12 colors.
        Intermediate shape: (12, 11, 12) → flattened to (1584,) floats.

    Parameters
    ----------
    puzzle_type:
        The AbstractPuzzle subclass this encoder is configured for.
        Used to determine output shape at construction time.
    """

    def __init__(self, puzzle_type: type[AbstractPuzzle]) -> None:
        self._puzzle_type = puzzle_type
        self._is_megaminx = _is_megaminx(puzzle_type)

        if self._is_megaminx:
            # (12, 11) state, 12 colors
            self._num_colors = 12
            self._raw_shape = (12, 11)
            flat = 12 * 11 * 12
        else:
            # (6, n, n) state for NxN cubes, 6 colors
            solved = puzzle_type.solved_state()
            n = solved.state.shape[1]
            self._n = n
            self._num_colors = 6
            self._raw_shape = (6, n, n)
            flat = 6 * n * n * 6

        self._output_shape: tuple[int, ...] = (flat,)
        self._output_size: int = flat

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the encoded tensor: a 1-D tuple with the flattened size."""
        return self._output_shape

    @property
    def output_size(self) -> int:
        """Total number of float32 values in the encoded representation."""
        return self._output_size

    def _one_hot_flat(self, state: np.ndarray) -> np.ndarray:
        """Convert a raw state array to a flat float32 one-hot array.

        Parameters
        ----------
        state:
            Raw puzzle state of shape ``self._raw_shape`` with dtype uint8.

        Returns
        -------
        np.ndarray
            Flat float32 array of length ``self._output_size``.
        """
        indices = state.astype(np.int32).ravel()  # (N,)
        n_cells = indices.size
        out = np.zeros((n_cells, self._num_colors), dtype=np.float32)
        out[np.arange(n_cells), indices] = 1.0
        return out.ravel()

    def encode(self, puzzle: AbstractPuzzle) -> np.ndarray:
        """Encode a single puzzle state as a flat one-hot float32 array.

        Parameters
        ----------
        puzzle:
            The puzzle instance to encode.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``self.output_shape``.
        """
        return self._one_hot_flat(puzzle.state)

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
        rows = [self._one_hot_flat(p.state) for p in puzzles]
        return np.stack(rows, axis=0)
