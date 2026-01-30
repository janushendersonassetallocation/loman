"""Default transformer configuration for serialization."""

from typing import Any

from .transformer import NdArrayTransformer, Transformer


def default_transformer(*args: Any, **kwargs: Any) -> Transformer:
    """Create a default transformer with NdArray support."""
    t = Transformer(*args, **kwargs)
    t.register(NdArrayTransformer())
    return t
