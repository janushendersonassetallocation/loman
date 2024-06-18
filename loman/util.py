import types
import itertools


def apply1(f, xs, *args, **kwds):
    if isinstance(xs, types.GeneratorType):
        return (f(x, *args, **kwds) for x in xs)
    if isinstance(xs, list):
        return [f(x, *args, **kwds) for x in xs]
    return f(xs, *args, **kwds)


def as_iterable(xs):
    if isinstance(xs, (types.GeneratorType, list, set)):
        return xs
    return (xs,)


def apply_n(f, *xs, **kwds):
    for p in itertools.product(*[as_iterable(x) for x in xs]):
        f(*p, **kwds)

class AttributeView:
    def __init__(self, get_attribute_list, get_attribute, get_item=None):
        self.get_attribute_list = get_attribute_list
        self.get_attribute = get_attribute
        self.get_item = get_item
        if self.get_item is None:
            self.get_item = get_attribute

    def __dir__(self):
        return self.get_attribute_list()

    def __getattr__(self, attr):
        try:
            return self.get_attribute(attr)
        except KeyError:
            raise AttributeError(attr)

    def __getitem__(self, key):
        return self.get_item(key)

    def __getstate__(self):
        return {
            'get_attribute_list': self.get_attribute_list,
            'get_attribute': self.get_attribute,
            'get_item': self.get_item
        }

    def __setstate__(self, state):
        self.get_attribute_list = state['get_attribute_list']
        self.get_attribute = state['get_attribute']
        self.get_item = state['get_item']

    @staticmethod
    def from_dict(d, use_apply1=True):
        if use_apply1:
            def get_attribute(xs):
                return apply1(d.get, xs)
        else:
            get_attribute = d.get
        return AttributeView(d.keys, get_attribute)