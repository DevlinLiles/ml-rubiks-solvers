"""Fitness strategies for the genetic algorithm solver.

Each strategy takes a chromosome (list of move indices), the initial puzzle
state, and the list of legal moves, applies the chromosome to the puzzle, and
returns a scalar fitness value where higher is better.
"""
from __future__ import annotations

import numpy as np

from rubiks_solve.core.base import AbstractPuzzle, Move


def _apply_chromosome(
    chromosome: list[int], puzzle: AbstractPuzzle, legal_moves: list[Move]
) -> AbstractPuzzle:
    """Apply a chromosome's moves to a puzzle and return the resulting state.

    Args:
        chromosome: Sequence of indices into ``legal_moves``.
        puzzle: The starting puzzle state (not mutated).
        legal_moves: Ordered list of legal moves for the puzzle type.

    Returns:
        The puzzle state reached after applying all chromosome moves.
    """
    result = puzzle
    for idx in chromosome:
        result = result.apply_move(legal_moves[idx])
    return result


def _count_misplaced(puzzle: AbstractPuzzle) -> int:
    """Return the number of facelets that are not in their solved position.

    This is computed by comparing the current state against the solved state of
    the same puzzle type.

    Args:
        puzzle: The puzzle state to evaluate.

    Returns:
        Integer count of facelets out of place.
    """
    solved = type(puzzle).solved_state()
    return int(np.sum(puzzle.state != solved.state))


def misplaced_facelets_fitness(
    chromosome: list[int], puzzle: AbstractPuzzle, legal_moves: list[Move]
) -> float:
    """Compute fitness as the negative count of misplaced facelets.

    Applies each move index in *chromosome* sequentially to *puzzle* using
    *legal_moves* as the move table, then counts how many facelets differ from
    the solved state.  The count is negated so that maximising fitness
    corresponds to fewer misplaced facelets (0 misplaced → fitness 0.0,
    which is the global maximum).

    Args:
        chromosome: Sequence of indices into ``legal_moves``.
        puzzle: The starting (scrambled) puzzle state.
        legal_moves: Ordered list of legal moves for the puzzle type.

    Returns:
        ``-misplaced_count`` as a float (higher is better).
    """
    result = _apply_chromosome(chromosome, puzzle, legal_moves)
    return -float(_count_misplaced(result))


def weighted_layer_fitness(
    chromosome: list[int], puzzle: AbstractPuzzle, legal_moves: list[Move]
) -> float:
    """Compute fitness with higher weight on bottom-layer facelets.

    Partitions the state array into thirds along the first axis (approximating
    bottom, middle, and top layers for cube puzzles) and applies weights
    [3, 2, 1] respectively so that completing lower layers is rewarded more
    strongly.  Falls back to uniform weighting for puzzles whose state does not
    divide evenly into three groups.

    Args:
        chromosome: Sequence of indices into ``legal_moves``.
        puzzle: The starting (scrambled) puzzle state.
        legal_moves: Ordered list of legal moves for the puzzle type.

    Returns:
        Weighted negative misplacement score as a float (higher is better).
    """
    result = _apply_chromosome(chromosome, puzzle, legal_moves)
    solved = type(puzzle).solved_state()

    current = result.state
    goal = solved.state

    n_faces = current.shape[0]
    third = n_faces // 3
    if third == 0:
        # Too few faces to partition; fall back to unweighted
        return -float(np.sum(current != goal))

    # Layer weights: bottom (indices 0..third-1) = 3, middle = 2, top = 1
    weights = np.ones_like(current, dtype=float)
    weights[:third] = 3.0
    weights[third : 2 * third] = 2.0
    # remaining faces keep weight 1.0

    misplaced_mask = (current != goal).astype(float)
    return -float(np.sum(misplaced_mask * weights))


def solved_bonus_fitness(
    chromosome: list[int], puzzle: AbstractPuzzle, legal_moves: list[Move]
) -> float:
    """Compute fitness as misplaced-facelet count plus a large bonus if solved.

    Identical to :func:`misplaced_facelets_fitness` but adds a bonus of
    ``10 * total_facelets`` when the puzzle is fully solved, making a complete
    solution strongly preferred over any partial solution.

    Args:
        chromosome: Sequence of indices into ``legal_moves``.
        puzzle: The starting (scrambled) puzzle state.
        legal_moves: Ordered list of legal moves for the puzzle type.

    Returns:
        ``-misplaced_count + bonus`` as a float (higher is better).
    """
    result = _apply_chromosome(chromosome, puzzle, legal_moves)
    misplaced = _count_misplaced(result)
    total_facelets = int(result.state.size)
    bonus = 10 * total_facelets if result.is_solved else 0
    return -float(misplaced) + float(bonus)
