from __future__ import annotations

from dataclasses import dataclass, field

import json
import re
from typing import List, Union, Iterable, Tuple


PART = re.compile(r'([^/]*)/?')


def quote_path_part(part: str) -> str:
    if '/' in part:
        return json.dumps(part)
    else:
        return part


def to_path(path: PathType) -> Path:
    if isinstance(path, str):
        return Path.scan(path)
    elif isinstance(path, Path):
        return path
    elif isinstance(path, tuple):
        return Path(path, False)
    elif isinstance(path, list):
        return Path(tuple(path), False)
    else:
        raise ValueError(f"Unexpected type for path {path}: {path.__class__}")


def to_paths(paths: PathsType) -> List[Path]:
    if isinstance(paths, (str, Path)):
        return [to_path(paths)]
    else:
        try:
            return [to_path(path) for path in paths]
        except TypeError:
            raise ValueError(f"Unexpected type for paths {paths}: {paths.__class__}")


def path_join(*paths: PathsType) -> Path:
    parts_all = []
    paths = to_paths(paths)
    for path in paths:
        parts_all.extend(path.parts)
    is_absolute_path = paths[0].is_absolute_path if len(paths) > 0 else False
    return Path(tuple(parts_all), is_absolute_path=is_absolute_path)


def path_common_parent(path1: PathType, path2: PathType):
    path1 = to_path(path1)
    path2 = to_path(path2)
    parts = []
    for p1, p2 in zip(path1.parts, path2.parts):
        if p1 != p2:
            break
        parts.append(p1)
    return Path(tuple(parts), path1.is_absolute_path)


@dataclass(frozen=True)
class Path:
    parts: Tuple[str, ...] = field()
    is_absolute_path: bool = field(default=True)

    def __post_init__(self):
        if not isinstance(self.parts, Tuple):
            raise ValueError("parts must be a Tuple")

    @staticmethod
    def _scan(path_str: str, end: int) -> Path:
        parts = []
        parts_append = parts.append

        is_absolute_path = False
        while path_str[end:end + 1] == '/':
            is_absolute_path = True
            end = end + 1
        while True:
            nextchar = path_str[end:end + 1]
            if nextchar == '':
                break
            if nextchar == '"':
                part, end = json.decoder.scanstring(path_str, end + 1)
                parts_append(part)
                nextchar = path_str[end:end + 1]
                assert nextchar == '' or nextchar == '/'
                if nextchar != '':
                    end = end + 1
            else:
                chunk = PART.match(path_str, end)
                end = chunk.end()
                part, = chunk.groups()
                parts_append(part)

        assert end == len(path_str)

        return Path(tuple(parts), is_absolute_path)

    @staticmethod
    def scan(path_str: str) -> Path:
        """Breaks a path string up into its component parts"""
        return Path._scan(path_str, 0)

    def str_no_absolute(self):
        return '/'.join([quote_path_part(part) for part in self.parts])

    def __str__(self) -> str:
        s = self.str_no_absolute()
        if self.is_absolute_path:
            s = '/' + s
        return s

    def parent(self) -> Path:
        if len(self.parts) == 0:
            raise PathNotFound()
        return Path(self.parts[:-1], self.is_absolute_path)

    def join(self, *parts: List[PathType]) -> Path:
        return path_join(self, *parts)

    def is_descendent_of(self, other: Path):
        n_self_parts = len(self.parts)
        n_other_parts = len(other.parts)
        return n_self_parts > n_other_parts and self.parts[:n_other_parts] == other.parts

    @classmethod
    def root(cls):
        return cls(tuple(), True)

    @property
    def is_root(self):
        return len(self.parts) == 0


class PathNotFound(Exception):
    pass


PathType = Union[str, Tuple[str], List[str], Path]
PathsType = Union[PathType, Iterable[PathType]]
