"""Gymnasium environments and reward functions for puzzle solving."""
from rubiks_solve.env.reward import DenseReward, PDTReward, SparseReward

try:
    from rubiks_solve.env.base_env import AbstractPuzzleEnv, RewardFn
    from rubiks_solve.env.cube_env import CubeEnv
    from rubiks_solve.env.wrappers import ScrambleDepthCurriculum
except ImportError:
    pass  # gymnasium not available (e.g. DGX without gym install)

__all__ = [
    "AbstractPuzzleEnv",
    "CubeEnv",
    "DenseReward",
    "PDTReward",
    "RewardFn",
    "ScrambleDepthCurriculum",
    "SparseReward",
]
