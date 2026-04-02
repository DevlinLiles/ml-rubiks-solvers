"""Transformer-based solver stub."""
from __future__ import annotations

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.solvers.base import AbstractSolver, SolveResult


class TransformerSolver(AbstractSolver):
    """Stub solver backed by a Transformer sequence model.

    This class is a placeholder.  All method calls raise
    :exc:`NotImplementedError` until the underlying
    :class:`~rubiks_solve.solvers.transformer.model.TransformerSolverModel` is
    implemented and trained.
    """

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Attempt to solve *puzzle* using the Transformer model (not implemented).

        Args:
            puzzle: The puzzle instance to solve.

        Raises:
            NotImplementedError: Always, until the model is implemented.
        """
        raise NotImplementedError("TransformerSolver is not yet implemented.")
