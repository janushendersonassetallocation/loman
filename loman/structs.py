from dataclasses import dataclass
from typing import Union, List, Iterable

from loman.path_parser import Path, to_path
from loman.util import as_iterable

InputName = Union[str, Path, 'NodeKey', object]
InputNames = List[InputName]
Name = Union[str, 'NodeKey', object]
Names = List[Name]


@dataclass(frozen=True, repr=False)
class NodeKey:
    path: Path
    obj: object = None

    def __post_init__(self):
        if self.path.is_absolute_path:
            object.__setattr__(self, 'path', Path(self.path.parts, False))  # Bypass immutability

    @classmethod
    def from_name(cls, name: InputName):
        if isinstance(name, str):
            return cls(to_path(name), None)
        elif isinstance(name, Path):
            return cls(name, None)
        elif isinstance(name, NodeKey):
            return name
        elif isinstance(name, object):
            return cls(Path.root(), name)
        else:
            raise ValueError(f"Unexpected error creating node key for name {name}")

    def __str__(self):
        if self.obj is None:
            return self.path.str_no_absolute()
        else:
            return f'{self.path.str_no_absolute()}: {self.obj}'

    @property
    def name(self) -> Name:
        if self.obj is None:
            return self.path.str_no_absolute()
        elif self.path.is_root and not isinstance(self.obj, str):
            return self.obj
        else:
            return self

    @property
    def label(self) -> str:
        if self.obj is None:
            return self.path.last_part
        else:
            return str(self.obj)

    @property
    def group_path(self) -> Path:
        if self.obj is None:
            return self.path.parent()
        else:
            return self.path

    def drop_root(self, root):
        path = self.path.drop_root(root)
        if path is None:
            return None
        return NodeKey(path, self.obj)

    def join(self, *parts):
        if len(parts) == 0:
            return self
        if self.obj is not None:
            raise Exception('Cannot join a node key with an object path')
        last_part = parts[-1]
        try:
            to_path(last_part)
            return NodeKey(self.path.join(*parts), None)
        except ValueError:
            return NodeKey(self.path.join(*parts[:-1]), last_part)

    def is_descendent_of(self, other: 'NodeKey'):
        if other.obj is None:
            return self.path.is_descendent_of(other.path)
        else:
            return self.path == other.path

    def parent(self):
        if self.obj is None:
            return NodeKey(self.path.parent())
        else:
            return NodeKey(self.path)

    def prepend_path(self, path: Path):
        return NodeKey(path.join(self.path), self.obj)

    def __repr__(self):
        path_str = str(self.path)
        quoted_path_str = repr(path_str)
        if self.obj is None:
            return f'{self.__class__.__name__}({quoted_path_str})'
        return f'{self.__class__.__name__}({quoted_path_str}, {self.obj})'

    def __eq__(self, other):
        return self.path.parts == other.path.parts and self.obj == other.obj


def names_to_node_keys(names: Union[InputName, InputNames]) -> List[NodeKey]:
    return [NodeKey.from_name(name) for name in as_iterable(names)]


def node_keys_to_names(node_keys: Iterable[NodeKey]) -> List[Name]:
    return [node_key.name for node_key in node_keys]
