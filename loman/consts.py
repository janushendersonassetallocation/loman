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


class NodeAttributes(object):
    VALUE = 'value'
    STATE = 'state'
    FUNC = 'func'
    GROUP = 'group'
    TAG = 'tag'
    ARGS = 'args'
    KWDS = 'kwds'
    TIMING = 'timing'
    EXECUTOR = 'executor'


class EdgeAttributes(object):
    PARAM = 'param'


class SystemTags(object):
    SERIALIZE = '__serialize__'
    EXPANSION = '__expansion__'