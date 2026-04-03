"""PyTorch Dueling DQN — DGX Spark backend equivalent of model.py."""
from __future__ import annotations

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """Dueling DQN with shared trunk, value stream, and advantage stream (PyTorch).

    Q(s,a) = V(s) + A(s,a) - mean_a'(A(s,a'))

    Architecture mirrors the MLX version in model.py.

    Args:
        input_size: Flattened encoder output size.
        n_actions:  Number of discrete legal moves.
    """

    def __init__(self, input_size: int, n_actions: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            nn.Linear(input_size, 4096), nn.ReLU(),
            nn.Linear(4096, 2048), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_stream(h)                          # (..., 1)
        a = self.advantage_stream(h)                      # (..., n_actions)
        return v + a - a.mean(dim=-1, keepdim=True)       # (..., n_actions)
