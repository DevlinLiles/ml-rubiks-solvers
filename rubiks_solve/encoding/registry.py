"""Registry of available state encoders."""
from __future__ import annotations

from rubiks_solve.core.base import AbstractPuzzle
from rubiks_solve.encoding.base import AbstractStateEncoder
from rubiks_solve.encoding.cubie import CubieEncoder
from rubiks_solve.encoding.one_hot import OneHotEncoder

ENCODER_REGISTRY: dict[str, type[AbstractStateEncoder]] = {
    "one_hot": OneHotEncoder,
    "cubie": CubieEncoder,
}
"""Mapping of encoder name strings to their AbstractStateEncoder subclasses.

Supported keys:
    ``"one_hot"``  – :class:`~rubiks_solve.encoding.one_hot.OneHotEncoder`
    ``"cubie"``    – :class:`~rubiks_solve.encoding.cubie.CubieEncoder`
"""


def get_encoder(name: str, puzzle_type: type[AbstractPuzzle]) -> AbstractStateEncoder:
    """Instantiate and return a state encoder by name.

    Parameters
    ----------
    name:
        Registry key, e.g. ``"one_hot"`` or ``"cubie"``.
    puzzle_type:
        The AbstractPuzzle subclass the encoder will operate on.

    Returns
    -------
    AbstractStateEncoder
        A freshly constructed encoder instance.

    Raises
    ------
    KeyError
        If ``name`` is not present in :data:`ENCODER_REGISTRY`.
    """
    try:
        encoder_cls = ENCODER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(ENCODER_REGISTRY))
        raise KeyError(
            f"Unknown encoder {name!r}. Available encoders: {available}"
        ) from None
    return encoder_cls(puzzle_type)
