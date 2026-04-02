"""CNN value network for estimating distance-to-solved from encoded cube state."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class CubeValueNet(nn.Module):
    """Fully-connected value network that estimates distance-to-solved.

    The network maps a flat encoded puzzle state to a single scalar value.
    Lower values indicate states closer to solved. The model is trained with
    MSE loss against scramble-depth labels produced offline.

    Architecture (default hidden_dims=[4096, 2048, 512]):
        Linear(input_size -> 4096) -> BatchNorm -> ReLU
        Linear(4096 -> 2048)       -> BatchNorm -> ReLU
        Linear(2048 -> 512)        -> BatchNorm -> ReLU
        Linear(512 -> 1)

    Args:
        input_size:   Flattened encoder output size (encoder.output_size).
        hidden_dims:  Sequence of hidden layer widths. Defaults to [4096, 2048, 512].
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

        # Build alternating Linear / BatchNorm layers for each hidden dimension.
        dims = [input_size] + hidden_dims
        linears: list[nn.Linear] = []
        batchnorms: list[nn.BatchNorm] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            linears.append(nn.Linear(in_dim, out_dim))
            batchnorms.append(nn.BatchNorm(out_dim))

        self.linears = linears
        self.batchnorms = batchnorms

        # Final output projection to a single scalar value.
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, input_size)`` or ``(input_size,)``.
               Values should be float32.

        Returns:
            Scalar value estimates of shape ``(batch, 1)`` or ``(1,)``.
        """
        for linear, bn in zip(self.linears, self.batchnorms):
            x = linear(x)
            x = bn(x)
            x = nn.relu(x)

        x = self.output_layer(x)
        return x
