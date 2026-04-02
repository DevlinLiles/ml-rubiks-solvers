"""Fixed-capacity circular replay buffer for DQN experience replay."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """A single environment transition stored in the replay buffer.

    Attributes:
        state:      Encoded state before the action (float32 array).
        action:     Index of the action taken.
        reward:     Scalar reward received.
        next_state: Encoded state after the action (float32 array).
        done:       True if the episode terminated after this transition.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular buffer of :class:`Transition` objects.

    When the buffer is full the oldest entries are overwritten in a FIFO
    manner.  Sampling is done without replacement to reduce correlation
    between consecutive gradient updates.

    Args:
        capacity: Maximum number of transitions held in memory.
                  Defaults to 100 000.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._buffer: list[Transition | None] = [None] * capacity
        self._head: int = 0   # Next write position.
        self._size: int = 0   # Current number of valid entries.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer, overwriting the oldest entry if full.

        Args:
            transition: The experience tuple to store.
        """
        self._buffer[self._head] = transition
        self._head = (self._head + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        """Sample a random mini-batch of transitions (without replacement).

        Args:
            batch_size: Number of transitions to sample.
            rng:        NumPy random generator for reproducible sampling.

        Returns:
            List of ``batch_size`` :class:`Transition` objects.

        Raises:
            ValueError: If ``batch_size`` exceeds the current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {self._size}."
            )
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]  # type: ignore[index]

    def __len__(self) -> int:
        """Return the number of transitions currently stored."""
        return self._size

    @property
    def capacity(self) -> int:
        """Maximum number of transitions the buffer can hold."""
        return self._capacity

    def is_ready(self, min_size: int) -> bool:
        """Return True if the buffer contains at least *min_size* transitions.

        Args:
            min_size: Minimum required buffer population before training starts.
        """
        return self._size >= min_size
