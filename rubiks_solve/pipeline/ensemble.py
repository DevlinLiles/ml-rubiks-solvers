"""Ensemble solver — runs multiple solvers concurrently and selects the best result."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from enum import Enum
from typing import Any

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.solvers.base import AbstractSolver, SolveResult


class VotingStrategy(Enum):
    """Strategy for selecting the winner among concurrent solver results.

    Attributes:
        SHORTEST_SOLUTION:    Prefer the result with fewest moves.  Ties are
                              broken by wall-clock time (fastest wins).
        FASTEST_SOLVE:        Prefer the result with the lowest wall-clock time.
                              Ties are broken by move count (shortest wins).
        CONFIDENCE_WEIGHTED:  Reserved for future use; currently falls back to
                              ``FASTEST_SOLVE`` behaviour.
    """

    SHORTEST_SOLUTION = "shortest"
    FASTEST_SOLVE = "fastest"
    CONFIDENCE_WEIGHTED = "weighted"


class EnsembleSolver(AbstractSolver):
    """Run multiple solvers concurrently and return the best result.

    GPU/MLX solvers share the process and the MLX compute context so they run
    on threads.  CPU-bound solvers (genetic, MCTS) also run on threads for
    simplicity and uniform cancellation semantics.

    Solvers that do not finish within *timeout_seconds* are cancelled; their
    partial results are discarded.

    Primary sort order and tiebreaking depend on the chosen
    :class:`VotingStrategy`:

    * ``FASTEST_SOLVE``        — wall-clock time ASC, then move count ASC.
    * ``SHORTEST_SOLUTION``    — move count ASC, then wall-clock time ASC.
    * ``CONFIDENCE_WEIGHTED``  — same as ``FASTEST_SOLVE`` (placeholder).

    If *no* solver produces a solved result the unsolved result with the fewest
    moves (primary) and shortest time (secondary) is returned.

    Args:
        puzzle_type:      The :class:`~rubiks_solve.core.base.AbstractPuzzle`
                          subclass this ensemble operates on.
        solvers:          Ordered list of :class:`~rubiks_solve.solvers.base.AbstractSolver`
                          instances.
        strategy:         Selection strategy; defaults to
                          :attr:`VotingStrategy.FASTEST_SOLVE`.
        timeout_seconds:  Maximum seconds to wait for any solver to finish.
        config:           Optional configuration object.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        solvers: list[AbstractSolver],
        strategy: VotingStrategy = VotingStrategy.FASTEST_SOLVE,
        timeout_seconds: float = 30.0,
        config: Any = None,
    ) -> None:
        super().__init__(puzzle_type, config)
        if not solvers:
            raise ValueError("EnsembleSolver requires at least one solver.")
        self._solvers = solvers
        self._strategy = strategy
        self._timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # AbstractSolver interface
    # ------------------------------------------------------------------

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Run all solvers concurrently; return the best result per strategy.

        Returns:
            A :class:`~rubiks_solve.solvers.base.SolveResult` chosen by
            :meth:`_select_best`.  The ``metadata`` dict contains:

            ``"all_results"``
                A :class:`dict` mapping ``solver_name`` to its
                :class:`~rubiks_solve.solvers.base.SolveResult`.

            ``"winner"``
                The ``solver_name`` string of the winning solver.
        """
        wall_start = time.perf_counter()
        results: dict[str, SolveResult] = {}

        # Map future -> solver_name so we can label results as they arrive.
        future_to_name: dict[Future, str] = {}

        with ThreadPoolExecutor(max_workers=len(self._solvers)) as executor:
            for solver in self._solvers:
                future = executor.submit(solver.solve, puzzle)
                future_to_name[future] = solver.solver_name

            remaining = self._timeout_seconds - (time.perf_counter() - wall_start)
            for future in as_completed(future_to_name, timeout=max(remaining, 0)):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    self._logger.warning(
                        "Solver %s raised an exception: %s", name, exc
                    )

        if not results:
            # All solvers timed out or raised — return a sentinel failure.
            total_time = time.perf_counter() - wall_start
            return SolveResult(
                solved=False,
                moves=[],
                solve_time_seconds=total_time,
                iterations=0,
                metadata={"all_results": {}, "winner": None},
            )

        best_result = self._select_best(results)
        winner_name = next(
            name for name, res in results.items() if res is best_result
        )

        # Stamp the overall wall-clock time on the returned result.
        total_time = time.perf_counter() - wall_start
        return SolveResult(
            solved=best_result.solved,
            moves=best_result.moves,
            solve_time_seconds=total_time,
            iterations=best_result.iterations,
            metadata={
                "all_results": results,
                "winner": winner_name,
                **best_result.metadata,
            },
        )

    def _select_best(self, results: dict[str, SolveResult]) -> SolveResult:
        """Return the best :class:`~rubiks_solve.solvers.base.SolveResult` from *results*.

        Solved results are always preferred over unsolved ones.  Among equally-
        solved (or equally-unsolved) results the :attr:`_strategy` determines
        the primary and secondary sort keys.

        Args:
            results: Mapping of solver name to its :class:`SolveResult`.

        Returns:
            The winning :class:`SolveResult` instance.
        """
        solved = {n: r for n, r in results.items() if r.solved}
        pool = solved if solved else results

        if self._strategy == VotingStrategy.SHORTEST_SOLUTION:
            # Primary: fewest moves; tiebreak: fastest time.
            return min(
                pool.values(),
                key=lambda r: (r.move_count, r.solve_time_seconds),
            )
        # FASTEST_SOLVE and CONFIDENCE_WEIGHTED (placeholder).
        # Primary: fastest time; tiebreak: fewest moves.
        return min(
            pool.values(),
            key=lambda r: (r.solve_time_seconds, r.move_count),
        )
