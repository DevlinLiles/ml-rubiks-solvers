"""Policy network: predicts a probability distribution over legal moves."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn  # pylint: disable=consider-using-from-import


class CubePolicyNet(nn.Module):
    """Fully-connected policy network that outputs move log-probabilities.

    The network is trained via imitation learning on expert (BFS/optimal)
    solutions or via policy-gradient methods.  It outputs log-softmax values
    so that cross-entropy loss can be computed directly without an extra
    log call.

    Architecture (default hidden_dims=[4096, 2048]):
        Linear(input_size -> 4096) -> ReLU
        Linear(4096 -> 2048)       -> ReLU
        Linear(2048 -> n_actions)  -> log_softmax

    Args:
        input_size:   Flattened encoder output size (``encoder.output_size``).
        n_actions:    Number of legal moves for the target puzzle type.
        hidden_dims:  Sequence of hidden layer widths.
                      Defaults to ``[4096, 2048]``.
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

        # Build hidden layers.
        dims = [input_size] + hidden_dims
        layers: list[nn.Linear] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))

        self.hidden_layers = layers

        # Output projection.
        self.output_layer = nn.Linear(hidden_dims[-1], n_actions)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass returning log-probabilities over actions.

        Args:
            x: Encoded state tensor of shape ``(batch, input_size)`` or
               ``(input_size,)``.  Must be float32.

        Returns:
            Log-probability array of shape ``(batch, n_actions)`` or
            ``(n_actions,)``, suitable for use with cross-entropy loss.
        """
        for layer in self.hidden_layers:
            x = nn.relu(layer(x))

        logits = self.output_layer(x)  # (..., n_actions)
        log_probs = nn.log_softmax(logits, axis=-1)
        return log_probs
