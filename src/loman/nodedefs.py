"""Node-definition primitives shared by the engine and the block builders.

A leaf module: it imports nothing from ``loman`` beyond :mod:`loman.compat`,
which is itself a leaf. That is the whole point of it. :mod:`loman.util`
needs this vocabulary on every call --- ``C`` to mark an argument as a
constant, ``identity_function`` to link one node to another, ``_bind_self``
to bind a feature's callback --- and :mod:`loman.computeengine` needs the same
names while importing :mod:`loman.util` at module level. Defining them here
lets both sides import them normally instead of deferring the import into
function bodies to survive a cycle.
"""

import types
from dataclasses import dataclass
from typing import Any

from .compat import get_signature


@dataclass()
class ConstantValue:
    """Container for constant values in computations."""

    value: object


C = ConstantValue


def _bind_self(f: Any, obj: object, ignore_self: bool) -> Any:
    """Bind a callback to the definition object when its first parameter is 'self'.

    Anything that is not callable, including ``None`` and a plain node name, is
    returned unchanged.

    Asking for ``self`` when there is no definition object to bind to is a
    contradiction, so it is reported here rather than as the bare
    ``TypeError: instance must not be None`` that binding would otherwise raise.
    """
    if not callable(f) or not ignore_self:
        return f
    signature = get_signature(f)
    if len(signature.kwd_params) > 0 and signature.kwd_params[0] == "self":
        if obj is None:
            name = getattr(f, "__qualname__", repr(f))
            msg = (
                f"Cannot bind 'self' for {name}: no definition object was supplied. "
                "Pass one, drop the 'self' parameter, or use ignore_self=False."
            )
            raise ValueError(msg)
        return types.MethodType(f, obj)
    return f


def identity_function(x: Any) -> Any:
    """Return the input value unchanged."""
    return x
