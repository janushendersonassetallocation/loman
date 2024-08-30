import inspect

import sys
from dataclasses import dataclass, field

from typing import List


@dataclass
class _Signature:
    kwd_params: List[str] = field()
    default_params: List[str] = field()
    has_var_args: bool = field()
    has_var_kwds: bool = field()


def get_signature(func):
    sig = inspect.signature(func)
    pk = inspect._ParameterKind
    has_var_args = False
    has_var_kwds = False
    all_keyword_params = []
    default_params = []
    for param_name, param in sig.parameters.items():
        if param.kind == pk.VAR_POSITIONAL:
            has_var_args = True
        elif param.kind == pk.VAR_KEYWORD:
            has_var_kwds = True
        elif param.kind in (pk.POSITIONAL_OR_KEYWORD, pk.KEYWORD_ONLY):
            all_keyword_params.append(param_name)
            if param.default != inspect._empty:
                default_params.append(param_name)
        else:
            raise NotImplementedError(f"Unexpected param kind: {param.kind}")
    return _Signature(all_keyword_params, default_params, has_var_args, has_var_kwds)
