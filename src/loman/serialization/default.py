"""Default transformer configuration for serialization."""

from .transformer import NdArrayTransformer, Transformer


def default_transformer(*args, **kwargs):
    t = Transformer(*args, **kwargs)
    t.register(NdArrayTransformer())
    return t
