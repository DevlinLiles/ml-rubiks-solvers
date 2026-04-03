"""Transformer-based puzzle solver model (stub — not yet implemented)."""
# Architecture placeholder for a Transformer-based puzzle solver.
# Future work: sequence model over move history to predict next move.


class TransformerSolverModel:
    """Placeholder for a Transformer-based puzzle-solving model.

    Intended future design: an autoregressive sequence model that takes the
    current puzzle state (and optionally a history of previous moves) as input
    and predicts a distribution over the next move to apply.

    All methods raise :exc:`NotImplementedError` until the architecture is
    implemented.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("TransformerSolverModel is not yet implemented.")

    def forward(self, *args, **kwargs):
        """Forward pass (not implemented)."""
        raise NotImplementedError("TransformerSolverModel is not yet implemented.")

    def predict(self, *args, **kwargs):
        """Predict the next move for a given puzzle state (not implemented)."""
        raise NotImplementedError("TransformerSolverModel is not yet implemented.")
