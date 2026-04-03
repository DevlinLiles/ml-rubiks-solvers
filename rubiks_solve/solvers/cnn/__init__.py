"""CNN-based solver package: value network + beam search."""
try:
    from rubiks_solve.solvers.cnn.model import CubeValueNet
    from rubiks_solve.solvers.cnn.trainer import CNNTrainer, CNNTrainerConfig
    from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig
except ImportError:
    pass  # MLX not available (e.g. DGX/Linux); use the *_torch modules directly.

__all__ = [
    "CubeValueNet",
    "CNNTrainer",
    "CNNTrainerConfig",
    "CNNSolver",
    "CNNConfig",
]
