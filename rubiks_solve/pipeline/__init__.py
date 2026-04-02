"""Pipeline package — multi-stage, ensemble, and routing solver infrastructure."""
from rubiks_solve.pipeline.chain import SolverChain, StageConfig
from rubiks_solve.pipeline.ensemble import EnsembleSolver, VotingStrategy
from rubiks_solve.pipeline.router import PuzzleRouter

__all__ = [
    "SolverChain",
    "StageConfig",
    "EnsembleSolver",
    "VotingStrategy",
    "PuzzleRouter",
]
