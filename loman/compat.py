import inspect
from collections import namedtuple

import sys

_Signature = namedtuple('_Signature', ['kwd_params', 'default_params', 'has_var_args', 'has_var_kwds'])


def get_signature(func):
    sig = inspect.signature(func)
    pk = inspect._ParameterKind
    has_var_args = any(p.kind == pk.VAR_POSITIONAL for p in sig.parameters.values())
    has_var_kwds = any(p.kind == pk.VAR_KEYWORD for p in sig.parameters.values())
    all_keyword_params = [param_name for param_name, param in sig.parameters.items()
                          if param.kind in (pk.POSITIONAL_OR_KEYWORD, pk.KEYWORD_ONLY)]
    default_params = [param_name for param_name, param in sig.parameters.items()
                          if param.kind in (pk.POSITIONAL_OR_KEYWORD, pk.KEYWORD_ONLY) and param.default != inspect._empty]
    return _Signature(all_keyword_params, default_params, has_var_args, has_var_kwds)
