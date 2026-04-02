"""Solver implementations for rubiks-solve.

Available solvers:

* :class:`~rubiks_solve.solvers.genetic.solver.GeneticSolver` — evolutionary
  algorithm using DEAP.
* :class:`~rubiks_solve.solvers.mcts.solver.MCTSSolver` — Monte Carlo Tree
  Search.
* :class:`~rubiks_solve.solvers.transformer.solver.TransformerSolver` — stub
  for a future Transformer-based model (raises :exc:`NotImplementedError`).
"""
from rubiks_solve.solvers.genetic.solver import GeneticConfig, GeneticSolver
from rubiks_solve.solvers.mcts.solver import MCTSConfig, MCTSSolver
from rubiks_solve.solvers.transformer.solver import TransformerSolver

__all__ = [
    "GeneticSolver",
    "GeneticConfig",
    "MCTSSolver",
    "MCTSConfig",
    "TransformerSolver",
]
