"""Tests for genetic algorithm operators."""
from __future__ import annotations

import random

import numpy as np

from rubiks_solve.solvers.genetic.operators import cx_single_point, mut_random_move
from rubiks_solve.solvers.genetic.fitness import misplaced_facelets_fitness


# ---------------------------------------------------------------------------
# Crossover tests
# ---------------------------------------------------------------------------

def test_cx_single_point_lengths():
    """Children must have combined length equal to parents' combined length."""
    random.seed(0)
    ind1 = [0, 1, 2, 3, 4]
    ind2 = [5, 6, 7, 8, 9, 10, 11]
    total_before = len(ind1) + len(ind2)
    child1, child2 = cx_single_point(list(ind1), list(ind2))
    total_after = len(child1) + len(child2)
    assert total_after == total_before


def test_cx_single_point_content():
    """Children must only contain genes from the original parents."""
    random.seed(1)
    parent1 = [0, 1, 2, 3]
    parent2 = [4, 5, 6, 7]
    all_genes = set(parent1 + parent2)
    child1, child2 = cx_single_point(list(parent1), list(parent2))
    for gene in child1:
        assert gene in all_genes, f"Child1 gene {gene} not from parents"
    for gene in child2:
        assert gene in all_genes, f"Child2 gene {gene} not from parents"


def test_cx_single_point_short_parent_unchanged():
    """cx_single_point on length-1 parents returns them unchanged."""
    ind1 = [3]
    ind2 = [7]
    child1, child2 = cx_single_point(list(ind1), list(ind2))
    assert child1 == [3]
    assert child2 == [7]


def test_cx_single_point_modifies_in_place():
    """cx_single_point modifies the inputs in-place and returns them."""
    random.seed(2)
    ind1 = [0, 1, 2, 3]
    ind2 = [4, 5, 6, 7]
    ref1, ref2 = ind1, ind2
    c1, c2 = cx_single_point(ind1, ind2)
    assert c1 is ref1
    assert c2 is ref2


def test_cx_single_point_produces_valid_length():
    """After crossover no child should be empty."""
    random.seed(3)
    for _ in range(20):
        ind1 = list(range(random.randint(2, 10)))
        ind2 = list(range(random.randint(2, 10)))
        c1, c2 = cx_single_point(list(ind1), list(ind2))
        assert len(c1) >= 1
        assert len(c2) >= 1


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------

def test_mut_random_move_changes_individual():
    """With indpb=1.0 the chromosome almost certainly changes."""
    random.seed(5)
    original = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    individual = list(original)
    n_moves = 18
    mutated, = mut_random_move(individual, n_moves, indpb=1.0)
    # With indpb=1.0 at least some genes should differ (insertions/deletions change length too)
    changed = list(mutated) != original
    assert changed


def test_mut_respects_move_range():
    """All genes in the mutated chromosome must be valid move indices."""
    random.seed(6)
    n_moves = 18
    individual = [0] * 20
    (mutated,) = mut_random_move(individual, n_moves, indpb=0.5)
    for gene in mutated:
        assert 0 <= gene < n_moves, f"Gene {gene} out of range [0, {n_moves})"


def test_mut_never_empty():
    """Mutation must never produce an empty chromosome."""
    random.seed(7)
    n_moves = 9
    for _ in range(50):
        individual = [random.randrange(n_moves) for _ in range(random.randint(1, 5))]
        (mutated,) = mut_random_move(list(individual), n_moves, indpb=1.0)
        assert len(mutated) >= 1


def test_mut_returns_one_element_tuple():
    """mut_random_move must return a 1-tuple as per DEAP convention."""
    individual = [0, 1, 2]
    result = mut_random_move(individual, n_moves=18, indpb=0.0)
    assert isinstance(result, tuple)
    assert len(result) == 1


def test_mut_zero_probability_no_change():
    """With indpb=0.0 the chromosome must not change."""
    individual = [3, 7, 2, 5, 0]
    original = list(individual)
    (mutated,) = mut_random_move(individual, n_moves=18, indpb=0.0)
    assert mutated == original


# ---------------------------------------------------------------------------
# Fitness tests
# ---------------------------------------------------------------------------

def test_fitness_misplaced_solved():
    """Fitness of the solved state applied by an empty chromosome is 0."""
    from rubiks_solve.core.cube_3x3 import Cube3x3
    solved = Cube3x3.solved_state()
    moves = solved.legal_moves()
    fitness = misplaced_facelets_fitness([], solved, moves)
    assert fitness == 0.0


def test_fitness_misplaced_scrambled():
    """Scrambled state (via identity chromosome) has negative fitness."""
    from rubiks_solve.core.cube_3x3 import Cube3x3
    rng = np.random.default_rng(42)
    scrambled = Cube3x3.solved_state().scramble(10, rng)
    moves = scrambled.legal_moves()
    fitness = misplaced_facelets_fitness([], scrambled, moves)
    assert fitness < 0.0, f"Expected negative fitness for scrambled state, got {fitness}"


def test_fitness_misplaced_solved_chromosome():
    """Applying the inverse of a single-move scramble gives fitness 0."""
    from rubiks_solve.core.cube_3x3 import Cube3x3
    solved = Cube3x3.solved_state()
    moves = solved.legal_moves()
    # Apply move 0 to get a scrambled state
    scrambled = solved.apply_move(moves[0])
    # Inverse move: find the inverse in the move list
    inv = moves[0].inverse()
    inv_idx = next(i for i, m in enumerate(moves) if m.name == inv.name)
    fitness = misplaced_facelets_fitness([inv_idx], scrambled, moves)
    assert fitness == 0.0, f"Expected 0.0 fitness after inverse move, got {fitness}"


def test_fitness_higher_is_better():
    """A chromosome that partially solves has higher fitness than one that does nothing."""
    from rubiks_solve.core.cube_3x3 import Cube3x3
    solved = Cube3x3.solved_state()
    moves = solved.legal_moves()
    # Scramble with move 0
    scrambled = solved.apply_move(moves[0])
    inv = moves[0].inverse()
    inv_idx = next(i for i, m in enumerate(moves) if m.name == inv.name)
    fitness_undone = misplaced_facelets_fitness([inv_idx], scrambled, moves)
    fitness_nothing = misplaced_facelets_fitness([], scrambled, moves)
    assert fitness_undone > fitness_nothing
