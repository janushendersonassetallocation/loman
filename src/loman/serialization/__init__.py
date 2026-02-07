"""Serialization utilities for Loman computations."""

from .default import default_transformer
from .transformer import (
    CustomTransformer,
    MissingObject,
    NdArrayTransformer,
    Transformable,
    Transformer,
    UnrecognizedTypeError,
    UntransformableTypeError,
)

# Backward compatibility aliases
UnrecognizedTypeException = UnrecognizedTypeError
UntransformableTypeException = UntransformableTypeError

__all__ = [
    "CustomTransformer",
    "MissingObject",
    "NdArrayTransformer",
    "Transformable",
    "Transformer",
    "UnrecognizedTypeException",
    "UntransformableTypeException",
    "default_transformer",
]
