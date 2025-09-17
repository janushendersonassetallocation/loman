"""Default transformer configuration for serialization."""

from .transformer import NdArrayTransformer, Transformer


def default_transformer(*args, **kwargs):
    """Create a default transformer with NdArray support."""
    t = Transformer(*args, **kwargs)
    t.register(NdArrayTransformer())
    return t
