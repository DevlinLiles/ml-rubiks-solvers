"""Multi-stage solver pipeline (SolverChain).

Classical analog: CFOP (cross -> F2L -> OLL -> PLL), each stage a different solver.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.solvers.base import AbstractSolver, SolveResult


@dataclass
class StageConfig:
    """Configuration for a single stage in a SolverChain.

    Attributes:
        solver:       The solver to use for this stage.
        move_budget:  Maximum number of moves this stage is allowed to use.
                      If the puzzle is not solved within the budget the (possibly
                      partially-reduced) state is forwarded to the next stage.
        pass_partial: When True, forward whatever state the stage left behind to
                      the next stage even if it is not solved. When False, forward
                      the *original* state received by this stage so the next stage
                      starts fresh.
    """

    solver: AbstractSolver
    move_budget: int
    pass_partial: bool = True


class SolverChain(AbstractSolver):
    """Multi-stage solver pipeline.

    Each stage receives the puzzle state produced by the previous stage.
    If stage N-1 fully solves the puzzle, the chain stops immediately.
    Otherwise the (possibly partially-reduced) state is forwarded to stage N,
    subject to the ``pass_partial`` flag on stage N-1's :class:`StageConfig`.

    Classical analog: CFOP (cross -> F2L -> OLL -> PLL), each stage a different
    solver.

    Note:
        Stage pairing compatibility is the user's responsibility.  Stage N must
        be able to make progress on whatever state stage N-1 leaves behind.

    Args:
        puzzle_type: The :class:`~rubiks_solve.core.base.AbstractPuzzle` subclass
            this chain operates on.
        stages:      Ordered list of :class:`StageConfig` objects.
        config:      Optional configuration object forwarded to
            :class:`~rubiks_solve.solvers.base.AbstractSolver`.
    """

    def __init__(
        self,
        puzzle_type: type[AbstractPuzzle],
        stages: list[StageConfig],
        config: Any = None,
    ) -> None:
        super().__init__(puzzle_type, config)
        if not stages:
            raise ValueError("SolverChain requires at least one stage.")
        self._stages = stages

    # ------------------------------------------------------------------
    # AbstractSolver interface
    # ------------------------------------------------------------------

    def solve(self, puzzle: AbstractPuzzle) -> SolveResult:
        """Run stages sequentially, aggregating all moves from every stage.

        Each stage's move budget is respected by truncating its solution to at
        most ``move_budget`` moves.  If the truncated sequence does not solve
        the puzzle, the partially-reduced (or original, per ``pass_partial``)
        state is passed to the next stage.

        Returns:
            A :class:`~rubiks_solve.solvers.base.SolveResult` whose ``moves``
            list is the concatenation of all stage move sequences.  The
            ``metadata`` dict contains:

            ``"stage_results"``
                A :class:`list` of :class:`~rubiks_solve.solvers.base.SolveResult`
                — one per stage (stages after a solution was found are absent).

            ``"stage_that_solved"``
                The integer index of the stage that produced a solved state, or
                ``None`` if no stage solved the puzzle.
        """
        wall_start = time.perf_counter()

        all_moves: list = []
        stage_results: list[SolveResult] = []
        stage_that_solved: int | None = None

        current_puzzle = puzzle

        for stage_idx, stage_cfg in enumerate(self._stages):
            result = stage_cfg.solver.solve(current_puzzle)

            # Respect the move budget: take only up to move_budget moves.
            budgeted_moves = result.moves[: stage_cfg.move_budget]

            # Build a truncated result reflecting what we actually used.
            truncated_result = SolveResult(
                solved=result.solved and len(result.moves) <= stage_cfg.move_budget,
                moves=budgeted_moves,
                solve_time_seconds=result.solve_time_seconds,
                iterations=result.iterations,
                metadata=result.metadata,
            )

            # Re-check solved status by actually applying the truncated moves.
            puzzle_after_stage = current_puzzle.apply_moves(budgeted_moves)
            truncated_result = SolveResult(
                solved=puzzle_after_stage.is_solved,
                moves=budgeted_moves,
                solve_time_seconds=result.solve_time_seconds,
                iterations=result.iterations,
                metadata=result.metadata,
            )

            stage_results.append(truncated_result)
            all_moves.extend(budgeted_moves)

            if truncated_result.solved:
                stage_that_solved = stage_idx
                break

            # Decide what state to hand to the next stage.
            if stage_cfg.pass_partial:
                current_puzzle = puzzle_after_stage
            # else: current_puzzle stays as it came into this stage

        total_time = time.perf_counter() - wall_start
        final_solved = stage_that_solved is not None

        return SolveResult(
            solved=final_solved,
            moves=all_moves,
            solve_time_seconds=total_time,
            iterations=sum(r.iterations for r in stage_results),
            metadata={
                "stage_results": stage_results,
                "stage_that_solved": stage_that_solved,
            },
        )
