"""Abstract state encoder interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle


class AbstractStateEncoder(ABC):
    """Converts puzzle state to a fixed-size numpy/MLX-compatible array.

    Encoders are injected into ML solvers. Swapping encoders (e.g. one-hot
    to cubie) requires only a config change, not solver code changes.
    """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the tensor produced by encode(). Used to size model input layers."""

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Flattened size: product of output_shape. Convenience for FC layer sizing."""

    @abstractmethod
    def encode(self, puzzle: AbstractPuzzle) -> np.ndarray:
        """Encode a single puzzle state. Returns float32 array of shape output_shape."""

    @abstractmethod
    def encode_batch(self, puzzles: list[AbstractPuzzle]) -> np.ndarray:
        """Encode a batch of puzzles. Returns float32 array of shape (N, *output_shape)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_shape={self.output_shape})"
