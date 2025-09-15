from .transformer import Transformer, NdArrayTransformer


def default_transformer(*args, **kwargs):
    t = Transformer(*args, **kwargs)
    t.register(NdArrayTransformer())
    return t