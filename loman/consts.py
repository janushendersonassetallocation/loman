from enum import Enum


class States(Enum):
    """Possible states for a computation node"""
    PLACEHOLDER = 0
    UNINITIALIZED = 1
    STALE = 2
    COMPUTABLE = 3
    UPTODATE = 4
    ERROR = 5
    PINNED = 6


class NodeAttributes:
    VALUE = 'value'
    STATE = 'state'
    FUNC = 'func'
    GROUP = 'group'
    TAG = 'tag'
    STYLE = 'style'
    ARGS = 'args'
    KWDS = 'kwds'
    TIMING = 'timing'
    EXECUTOR = 'executor'
    CONVERTER = 'converter'


class EdgeAttributes:
    PARAM = 'param'


class SystemTags:
    SERIALIZE = '__serialize__'
    EXPANSION = '__expansion__'


class NodeTransformations:
    CONTRACT = '__contract__'
    COLLAPSE = '__collapse__'