"""Genetic operators for variable-length chromosomes.

Chromosomes are lists of integer move indices.  All operators follow DEAP
conventions: crossover functions modify individuals *in-place* and return the
two modified individuals; mutation functions modify the individual *in-place*
and return a one-element tuple containing it.
"""
from __future__ import annotations

import random
from typing import Any


def cx_single_point(ind1: list[int], ind2: list[int]) -> tuple[list[int], list[int]]:
    """Single-point crossover for variable-length sequences.

    Selects a random cut point within each parent independently, then swaps the
    tails.  This preserves the head of each parent up to its cut point and
    replaces the remainder with the other parent's tail, allowing offspring to
    have different lengths from both parents.

    Both individuals are modified in-place, as required by DEAP.

    Args:
        ind1: First parent chromosome (list of move indices).
        ind2: Second parent chromosome (list of move indices).

    Returns:
        The two modified individuals as a tuple ``(ind1, ind2)``.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # Nothing meaningful to cross — return unchanged
        return ind1, ind2

    cut1 = random.randint(1, len(ind1) - 1)
    cut2 = random.randint(1, len(ind2) - 1)

    # Swap tails
    tail1 = ind1[cut1:]
    tail2 = ind2[cut2:]

    # Modify in-place (DEAP requirement)
    del ind1[cut1:]
    ind1 += tail2
    del ind2[cut2:]
    ind2 += tail1

    return ind1, ind2


def mut_random_move(
    individual: list[int], n_moves: int, indpb: float
) -> tuple[list[int]]:
    """Mutation operator for variable-length move-index chromosomes.

    Iterates over each gene position.  With probability *indpb* one of three
    mutations is applied at that position, chosen with equal probability:

    * **flip** — replace the gene with a random move index.
    * **insert** — insert a random move index before the current position.
    * **delete** — remove the gene at the current position.

    At least one gene is always preserved so chromosomes never become empty.
    The individual is modified in-place and returned as a one-element tuple, as
    required by DEAP.

    Args:
        individual: Chromosome to mutate (list of move indices).
        n_moves: Total number of legal moves; indices are drawn from
            ``range(n_moves)``.
        indpb: Per-gene probability of applying a mutation.

    Returns:
        One-element tuple ``(individual,)`` after in-place mutation.
    """
    i = 0
    while i < len(individual):
        if random.random() < indpb:
            op = random.randint(0, 2)
            if op == 0:
                # Flip: replace gene with a random move index
                individual[i] = random.randrange(n_moves)
                i += 1
            elif op == 1:
                # Insert: add a new random gene before position i
                individual.insert(i, random.randrange(n_moves))
                i += 2  # skip the newly inserted gene
            else:
                # Delete: remove gene at position i (guard minimum length)
                if len(individual) > 1:
                    del individual[i]
                    # i stays the same; next gene has shifted into position i
                else:
                    i += 1
        else:
            i += 1

    return (individual,)


def sel_tournament_variable(
    population: list[Any], k: int, tournsize: int
) -> list[Any]:
    """Tournament selection robust to variable fitness scales.

    Draws *tournsize* individuals at random (with replacement) per trial and
    selects the one with the highest fitness value.  Repeats *k* times to
    produce the selected pool.

    Unlike DEAP's built-in ``selTournament``, this function accesses fitness
    via ``ind.fitness.values[0]`` and handles the case where a chromosome may
    have been assigned a ``None`` or invalid fitness by treating missing fitness
    as negative infinity so such individuals are never selected over evaluated
    ones.

    Args:
        population: List of DEAP individuals, each with a ``.fitness.values``
            attribute populated by the toolbox evaluate step.
        k: Number of individuals to select.
        tournsize: Number of contestants per tournament round.

    Returns:
        List of *k* selected individuals (references, not copies).
    """

    def _fitness(ind: Any) -> float:
        try:
            val = ind.fitness.values[0]
            return float(val) if val is not None else float("-inf")
        except (AttributeError, IndexError):
            return float("-inf")

    selected: list[Any] = []
    for _ in range(k):
        contestants = random.choices(population, k=tournsize)
        winner = max(contestants, key=_fitness)
        selected.append(winner)

    return selected
