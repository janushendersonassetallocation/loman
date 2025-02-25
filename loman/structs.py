from dataclasses import dataclass
from typing import Union, List, Iterable

from loman.path_parser import Path, to_path
from loman.util import as_iterable

InputName = Union[str, Path, 'NodeKey', object]
InputNames = List[InputName]
Name = Union[str, 'NodeKey', object]
Names = List[Name]


@dataclass(frozen=True)
class NodeKey:
    path: Path
    obj: object

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


def names_to_node_keys(names: Union[InputName, InputNames]) -> List[NodeKey]:
    return [NodeKey.from_name(name) for name in as_iterable(names)]


def node_keys_to_names(node_keys: Iterable[NodeKey]) -> List[Name]:
    return [node_key.name for node_key in node_keys]
