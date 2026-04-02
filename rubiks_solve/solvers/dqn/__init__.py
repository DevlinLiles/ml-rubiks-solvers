"""DQN-based solver package: Dueling DQN with experience replay."""
from rubiks_solve.solvers.dqn.replay_buffer import ReplayBuffer, Transition
from rubiks_solve.solvers.dqn.model import DuelingDQN
from rubiks_solve.solvers.dqn.trainer import DQNTrainer, DQNTrainerConfig
from rubiks_solve.solvers.dqn.solver import DQNSolver, DQNConfig

__all__ = [
    "ReplayBuffer",
    "Transition",
    "DuelingDQN",
    "DQNTrainer",
    "DQNTrainerConfig",
    "DQNSolver",
    "DQNConfig",
]
