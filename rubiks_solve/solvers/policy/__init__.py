"""Policy-network-based solver package: imitation learning + greedy rollout."""
try:
    from rubiks_solve.solvers.policy.model import CubePolicyNet
    from rubiks_solve.solvers.policy.trainer import PolicyTrainer, PolicyTrainerConfig
    from rubiks_solve.solvers.policy.solver import PolicyNetworkSolver, PolicyConfig
except ImportError:
    pass  # MLX not available (e.g. DGX/Linux); use the *_torch modules directly.

__all__ = [
    "CubePolicyNet",
    "PolicyTrainer",
    "PolicyTrainerConfig",
    "PolicyNetworkSolver",
    "PolicyConfig",
]
