"""Stable helpers referenced by the serialized-format golden files.

The golden files under ``tests/data/formats/`` store functions as
``module`` + ``qualname`` references.  Those references are resolved by import
when the file is read back, so **the names in this module must not change**.
Renaming or removing anything here breaks the backward-compatibility corpus,
which is exactly the promise the corpus exists to protect.

Add new helpers freely; never rename or delete an existing one.
"""

from __future__ import annotations


def add_one(x):
    """Return ``x + 1``."""
    return x + 1


def double(x):
    """Return ``2 * x``."""
    return 2 * x


def add(x, y):
    """Return ``x + y``."""
    return x + y


def raise_value_error():
    """Raise a ValueError, to capture a node in ERROR state."""
    msg = "deliberate golden-file error"
    raise ValueError(msg)
