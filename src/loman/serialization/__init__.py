"""Serialization utilities for Loman computations."""

from .archive import (
    ARCHIVE_EXTENSIONS,
    ArchiveSerializer,
    has_parquet_support,
    is_archive_path,
)
from .computation import FORMAT_VERSION, MIN_SUPPORTED_VERSION, ComputationSerializer
from .default import default_transformer
from .transformer import (
    CustomTransformer,
    DateTimeTransformer,
    DillFunctionTransformer,
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
    "ARCHIVE_EXTENSIONS",
    "FORMAT_VERSION",
    "MIN_SUPPORTED_VERSION",
    "ArchiveSerializer",
    "ComputationSerializer",
    "CustomTransformer",
    "DateTimeTransformer",
    "DillFunctionTransformer",
    "MissingObject",
    "NdArrayTransformer",
    "Transformable",
    "Transformer",
    "UnrecognizedTypeException",
    "UntransformableTypeException",
    "default_transformer",
    "has_parquet_support",
    "is_archive_path",
]
