"""PuzzleRouter — dispatch solve requests to registered solvers by puzzle type."""
from __future__ import annotations

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.solvers.base import AbstractSolver, SolveResult


class PuzzleRouter:
    """Route solve requests to the appropriate solver based on puzzle type.

    Each concrete :class:`~rubiks_solve.core.base.AbstractPuzzle` subclass can
    have at most one solver (or :class:`~rubiks_solve.pipeline.chain.SolverChain`
    / :class:`~rubiks_solve.pipeline.ensemble.EnsembleSolver`) registered at a
    time.  Re-registering a type replaces the existing entry.

    Example::

        router = PuzzleRouter()
        router.register(Cube3x3, my_chain)
        router.register(Cube2x2, kociemba_solver)
        result = router.solve(scrambled_cube)
    """

    def __init__(self) -> None:
        self._routes: dict[type[AbstractPuzzle], AbstractSolver] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        puzzle_type: type[AbstractPuzzle],
        solver: AbstractSolver,
    ) -> None:
        """Register *solver* for puzzles of *puzzle_type*.

        If a solver is already registered for *puzzle_type* it is silently
        replaced.

        Args:
            puzzle_type: The exact :class:`~rubiks_solve.core.base.AbstractPuzzle`
                         subclass to match (``isinstance`` is **not** used for
                         routing — only exact type equality).
            solver:      The :class:`~rubiks_solve.solvers.base.AbstractSolver`
                         (or composed pipeline) to use for this type.
        """
        self._routes[puzzle_type] = solver

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Dispatch *puzzle* to the registered solver for its type.

        Args:
            puzzle: The puzzle instance to solve.

        Returns:
            The :class:`~rubiks_solve.solvers.base.SolveResult` produced by the
            registered solver.

        Raises:
            KeyError: If no solver is registered for ``type(puzzle)``.
        """
        solver = self._get_solver(puzzle)
        return solver.solve(puzzle)

    def supports(self, puzzle: AbstractPuzzle) -> bool:
        """Return ``True`` if a solver is registered for ``type(puzzle)``.

        Args:
            puzzle: The puzzle instance to check.

        Returns:
            ``True`` when :meth:`solve` would succeed for this puzzle.
        """
        return type(puzzle) in self._routes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_solver(self, puzzle: AbstractPuzzle) -> AbstractSolver:
        """Look up the solver for *puzzle*, raising :class:`KeyError` if absent.

        Args:
            puzzle: Puzzle instance whose type is used as the routing key.

        Returns:
            The registered :class:`~rubiks_solve.solvers.base.AbstractSolver`.

        Raises:
            KeyError: If no solver has been registered for this puzzle type.
        """
        puzzle_type = type(puzzle)
        if puzzle_type not in self._routes:
            raise KeyError(
                f"No solver registered for puzzle type '{puzzle_type.__name__}'. "
                f"Registered types: {[t.__name__ for t in self._routes]}"
            )
        return self._routes[puzzle_type]
