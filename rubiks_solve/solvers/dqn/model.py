"""Dueling DQN architecture with separate value and advantage streams."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network for puzzle solving.

    Decomposes Q-values into a state-value stream V(s) and an advantage
    stream A(s, a).  The final Q-values are computed as:

        Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))

    Subtracting the mean advantage improves training stability by making
    the advantage identifiable (the value baseline absorbs the mean).

    Architecture:
        Shared trunk:
            Linear(input_size -> 4096) -> ReLU
            Linear(4096 -> 2048)       -> ReLU

        Value stream:
            Linear(2048 -> 512) -> ReLU
            Linear(512 -> 1)

        Advantage stream:
            Linear(2048 -> 512) -> ReLU
            Linear(512 -> n_actions)

    Args:
        input_size: Flattened encoder output (``encoder.output_size``).
        n_actions:  Number of discrete actions (legal moves).
    """

    def __init__(self, input_size: int, n_actions: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.n_actions = n_actions

        # --- Shared trunk ---
        self.fc1 = nn.Linear(input_size, 4096)
        self.fc2 = nn.Linear(4096, 2048)

        # --- Value stream ---
        self.value_fc = nn.Linear(2048, 512)
        self.value_out = nn.Linear(512, 1)

        # --- Advantage stream ---
        self.advantage_fc = nn.Linear(2048, 512)
        self.advantage_out = nn.Linear(512, n_actions)

    def __call__(self, x: mx.array) -> mx.array:
        """Compute Q-values for all actions.

        Args:
            x: Encoded state tensor of shape ``(batch, input_size)`` or
               ``(input_size,)``.  Should be float32.

        Returns:
            Q-value tensor of shape ``(batch, n_actions)`` or ``(n_actions,)``.
        """
        # Shared trunk.
        h = nn.relu(self.fc1(x))
        h = nn.relu(self.fc2(h))

        # Value stream.
        v = nn.relu(self.value_fc(h))
        v = self.value_out(v)  # (..., 1)

        # Advantage stream.
        a = nn.relu(self.advantage_fc(h))
        a = self.advantage_out(a)  # (..., n_actions)

        # Combine: Q = V + A - mean(A)
        # keepdims ensures broadcasting works for both batched and single inputs.
        a_mean = mx.mean(a, axis=-1, keepdims=True)
        q = v + a - a_mean  # (..., n_actions)
        return q
