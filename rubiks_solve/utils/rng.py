"""Random number generation utilities for reproducible experiments."""
from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a seeded numpy Generator backed by PCG64.

    Args:
        seed: Integer seed for reproducibility.  ``None`` seeds from OS entropy.

    Returns:
        A new :class:`numpy.random.Generator` instance.
    """
    return np.random.default_rng(np.random.PCG64(seed))


def make_mlx_key(seed: int) -> "mx.array":
    """Create a seeded MLX PRNG key via ``mx.random.key``.

    Args:
        seed: Non-negative integer seed.

    Returns:
        An ``mx.array`` suitable for use with MLX random functions.
    """
    import mlx.core as mx

    return mx.random.key(seed)


_global_rng: np.random.Generator | None = None


def get_global_rng() -> np.random.Generator:
    """Return the module-level RNG, initialising it with OS entropy if needed.

    Returns:
        The module-level :class:`numpy.random.Generator`.
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = make_rng(None)
    return _global_rng


def set_global_seed(seed: int) -> None:
    """Set both numpy and MLX global seeds for reproducibility.

    Resets the module-level :data:`_global_rng` and calls
    ``mlx.core.random.seed`` so that all downstream random operations
    produce deterministic outputs.

    Args:
        seed: Integer seed applied to both backends.
    """
    global _global_rng
    _global_rng = make_rng(seed)

    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass
