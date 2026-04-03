"""PyTorch CNN value network — DGX Spark backend equivalent of model.py."""
from __future__ import annotations

import torch
from torch import nn


class CubeValueNet(nn.Module):
    """Fully-connected value network estimating distance-to-solved (PyTorch).

    Architecture mirrors the MLX version in model.py:
        Linear(input_size -> 4096) -> BatchNorm1d -> ReLU
        Linear(4096 -> 2048)       -> BatchNorm1d -> ReLU
        Linear(2048 -> 512)        -> BatchNorm1d -> ReLU
        Linear(512 -> 1)

    Args:
        input_size:  Flattened encoder output size.
        hidden_dims: Sequence of hidden layer widths. Defaults to [4096, 2048, 512].
    """

    def __init__(
        self,
        input_size: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [4096, 2048, 512]

        self.input_size = input_size
        self.hidden_dims = hidden_dims

        dims = [input_size] + hidden_dims
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: return scalar value estimate for each input state."""
        return self.net(x)
