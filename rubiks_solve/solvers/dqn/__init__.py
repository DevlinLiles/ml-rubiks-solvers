"""DQN-based solver package: Dueling DQN with experience replay."""
from rubiks_solve.solvers.dqn.replay_buffer import ReplayBuffer, Transition

try:
    from rubiks_solve.solvers.dqn.model import DuelingDQN
    from rubiks_solve.solvers.dqn.trainer import DQNTrainer, DQNTrainerConfig
    from rubiks_solve.solvers.dqn.solver import DQNSolver, DQNConfig
except ImportError:
    pass  # MLX not available (e.g. DGX/Linux); use the *_torch modules directly.

__all__ = [
    "ReplayBuffer",
    "Transition",
    "DuelingDQN",
    "DQNTrainer",
    "DQNTrainerConfig",
    "DQNSolver",
    "DQNConfig",
]
