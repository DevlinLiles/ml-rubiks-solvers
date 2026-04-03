"""PyTorch policy network — DGX Spark backend equivalent of model.py."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CubePolicyNet(nn.Module):
    """Fully-connected policy network outputting log-probabilities (PyTorch).

    Architecture mirrors the MLX version in model.py:
        Linear(input_size -> 4096) -> ReLU
        Linear(4096 -> 2048)       -> ReLU
        Linear(2048 -> n_actions)  -> log_softmax

    Args:
        input_size:  Flattened encoder output size.
        n_actions:   Number of legal moves.
        hidden_dims: Hidden layer widths. Defaults to [4096, 2048].
    """

    def __init__(
        self,
        input_size: int,
        n_actions: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [4096, 2048]

        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims

        dims = [input_size] + hidden_dims
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(x), dim=-1)
