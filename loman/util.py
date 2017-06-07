class AttributeView(object):
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
            raise AttributeError()

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
