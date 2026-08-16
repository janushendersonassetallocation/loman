"""Default transformer configuration for serialization.

.. deprecated::
    :func:`default_transformer` predates
    :func:`~loman.serialization.computation.default_computation_transformer` and
    is not used on the :class:`~loman.computeengine.Computation` path. It
    registers only ndarray support, so a Transformer built with it cannot encode
    a DataFrame, a Series, a callable or a NodeKey. Use
    :func:`~loman.serialization.computation.default_computation_transformer`, or
    construct a bare :class:`~loman.serialization.transformer.Transformer` and
    register what you need.
"""

import warnings
from typing import Any

from .transformer import NdArrayTransformer, Transformer


def default_transformer(*args: Any, **kwargs: Any) -> Transformer:
    """Create a default transformer with NdArray support.

    .. deprecated::
        Use
        :func:`~loman.serialization.computation.default_computation_transformer`
        instead. This function will be removed in a future release.
    """
    warnings.warn(
        "default_transformer is deprecated and will be removed in a future release. "
        "Use loman.serialization.computation.default_computation_transformer instead, "
        "or build a Transformer and register the transformers you need.",
        DeprecationWarning,
        stacklevel=2,
    )
    t = Transformer(*args, **kwargs)
    t.register(NdArrayTransformer())
    return t
