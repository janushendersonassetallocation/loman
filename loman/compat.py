import inspect
from collections import namedtuple

import sys

import six

_Signature = namedtuple('_Signature', ['kwd_params', 'default_params', 'has_var_args', 'has_var_kwds'])

if six.PY3:
    if sys.version_info >= (3, 5):
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
    elif sys.version_info >= (3, 4):
        def get_signature(func):
            sig = inspect.signature(func)
            has_var_args = any(p.kind == inspect._VAR_POSITIONAL for p in sig.parameters.values())
            has_var_kwds = any(p.kind == inspect._VAR_KEYWORD for p in sig.parameters.values())
            all_keyword_params = [param_name for param_name, param in sig.parameters.items()
                                  if param.kind in (inspect._POSITIONAL_OR_KEYWORD, inspect._KEYWORD_ONLY)]
            default_params = [param_name for param_name, param in sig.parameters.items()
                                  if param.kind in (inspect._POSITIONAL_OR_KEYWORD, inspect._KEYWORD_ONLY) and param.default != inspect._empty]
            return _Signature(all_keyword_params, default_params, has_var_args, has_var_kwds)
    else:
        raise Exception("Only Python3 >=3.4 is supported")
elif six.PY2:
    def get_signature(func):
        argspec = inspect.getargspec(func)
        has_var_args = argspec.varargs is not None
        has_var_kwds = argspec.keywords is not None
        all_keyword_params = argspec.args
        if argspec.defaults is None:
            default_params = []
        else:
            n_default_params = len(argspec.defaults)
            default_params = argspec.args[-n_default_params:]
        return _Signature(all_keyword_params, default_params, has_var_args, has_var_kwds)
else:
    raise Exception("Only Pythons 2 and 3 supported")