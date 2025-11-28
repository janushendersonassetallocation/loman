"""Node key implementation for computation graph navigation."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Union

from loman.util import as_iterable

Name = Union[str, "NodeKey", object]
Names = list[Name]


class PathNotFoundError(Exception):
    """Exception raised when a node path cannot be found."""

    pass


def quote_part(part: object) -> str:
    """Quote a node key part for safe representation in paths."""
    if isinstance(part, str):
        if "/" in part:
            return json.dumps(part)
        else:
            return part
    return str(part)


@dataclass(frozen=True, repr=False)
class NodeKey:
    """Immutable key for identifying nodes in the computation graph hierarchy."""

    parts: tuple

    def __str__(self):
        """Return string representation using path notation."""
        return "/".join([quote_part(part) for part in self.parts])

    @property
    def name(self) -> Name:
        """Get the name of this node (last part of the path)."""
        if len(self.parts) == 0:
            return ""
        elif len(self.parts) == 1:
            return self.parts[0]
        elif all(isinstance(part, str) for part in self.parts):
            return "/".join(quote_part(part) for part in self.parts)
        else:
            return self

    @property
    def label(self) -> str:
        """Get the label for this node (for display purposes)."""
        if len(self.parts) == 0:
            return ""
        return str(self.parts[-1])

    def drop_root(self, root: Optional["Name"]) -> Optional["NodeKey"]:
        """Remove a root prefix from this node key if it matches."""
        if root is None:
            return self
        root = to_nodekey(root)
        n_root_parts = len(root.parts)
        if self.is_descendent_of(root):
            parts = self.parts[n_root_parts:]
            return NodeKey(parts)
        else:
            return None

    def join(self, *others: Name) -> "NodeKey":
        """Join this node key with other names to create a new node key."""
        result = self
        for other in others:
            if other is None:
                continue
            other = to_nodekey(other)
            result = result.join_parts(*other.parts)
        return result

    def join_parts(self, *parts) -> "NodeKey":
        """Join this node key with raw parts to create a new node key."""
        if len(parts) == 0:
            return self
        return NodeKey(self.parts + tuple(parts))

    def __add__(self, other: Name) -> "NodeKey":
        """Join this node key with other to create a new node key."""
        return self.join(other)

    def is_descendent_of(self, other: "NodeKey") -> bool:
        """Check if this node key is a descendant of another node key."""
        n_self_parts = len(self.parts)
        n_other_parts = len(other.parts)
        return n_self_parts > n_other_parts and self.parts[:n_other_parts] == other.parts

    @property
    def parent(self) -> "NodeKey":
        """Get the parent node key."""
        if len(self.parts) == 0:
            raise PathNotFoundError()
        return NodeKey(self.parts[:-1])

    def prepend(self, nk: "NodeKey") -> "NodeKey":
        """Prepend another node key to this one."""
        return nk.join_parts(*self.parts)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        path_str = str(self)
        quoted_path_str = repr(path_str)
        return f"{self.__class__.__name__}({quoted_path_str})"

    def __eq__(self, other) -> bool:
        """Check equality with another NodeKey."""
        if other is None:
            return False
        if not isinstance(other, NodeKey):
            return NotImplemented
        return self.parts == other.parts

    _ROOT = None

    @classmethod
    def root(cls) -> "NodeKey":
        """Get the root node key."""
        if cls._ROOT is None:
            cls._ROOT = cls(())
        return cls._ROOT

    @property
    def is_root(self):
        """Check if this is the root node key."""
        return len(self.parts) == 0

    @staticmethod
    def common_parent(nodekey1: Name, nodekey2: Name):
        """Find the common parent of two node keys."""
        nodekey1 = to_nodekey(nodekey1)
        nodekey2 = to_nodekey(nodekey2)
        parts = []
        for p1, p2 in zip(nodekey1.parts, nodekey2.parts):
            if p1 != p2:
                break
            parts.append(p1)
        return NodeKey(tuple(parts))

    def ancestors(self) -> list["NodeKey"]:
        """Get all ancestor node keys from root to parent."""
        result = []
        x = self
        while True:
            result.append(x)
            if x.is_root:
                break
            x = x.parent
        return result


def names_to_node_keys(names: Name | Names) -> list[NodeKey]:
    """Convert names to NodeKey objects."""
    return [to_nodekey(name) for name in as_iterable(names)]


def node_keys_to_names(node_keys: Iterable[NodeKey]) -> list[Name]:
    """Convert NodeKey objects back to names."""
    return [node_key.name for node_key in node_keys]


PART = re.compile(r"([^/]*)/?")


def _parse_nodekey(path_str: str, end: int) -> NodeKey:
    parts = []
    parts_append = parts.append

    while path_str[end : end + 1] == "/":
        end = end + 1
    while True:
        nextchar = path_str[end : end + 1]
        if nextchar == "":
            break
        if nextchar == '"':
            part, end = json.decoder.scanstring(path_str, end + 1)
            parts_append(part)
            nextchar = path_str[end : end + 1]
            assert nextchar == "" or nextchar == "/"
            if nextchar != "":
                end = end + 1
        else:
            chunk = PART.match(path_str, end)
            end = chunk.end()
            (part,) = chunk.groups()
            parts_append(part)

    assert end == len(path_str)

    return NodeKey(tuple(parts))


def parse_nodekey(path_str: str) -> NodeKey:
    """Parse a string representation into a NodeKey."""
    return _parse_nodekey(path_str, 0)


def to_nodekey(name: Name) -> NodeKey:
    """Convert a name to a NodeKey object."""
    if isinstance(name, str):
        return parse_nodekey(name)
    elif isinstance(name, NodeKey):
        return name
    elif isinstance(name, object):
        return NodeKey((name,))
    else:
        raise ValueError(f"Unexpected error creating node key for name {name}")


def nodekey_join(*names: Name) -> NodeKey:
    """Join multiple names into a single NodeKey."""
    return NodeKey.root().join(*names)


def _match_pattern_recursive(pattern: NodeKey, target: NodeKey, p_idx: int, t_idx: int) -> bool:
    """Recursively match pattern parts against target parts.

    Args:
        pattern: The pattern NodeKey to match against
        target: The target NodeKey to match
        p_idx: Current index in pattern parts
        t_idx: Current index in target parts

    Returns:
        bool: True if pattern matches target, False otherwise
    """
    if p_idx == len(pattern.parts) and t_idx == len(target.parts):
        return True
    if p_idx == len(pattern.parts):
        return False
    if t_idx == len(target.parts):
        return all(p == "**" for p in pattern.parts[p_idx:])

    if pattern.parts[p_idx] == "**":
        return _match_pattern_recursive(pattern, target, p_idx + 1, t_idx) or _match_pattern_recursive(
            pattern, target, p_idx, t_idx + 1
        )
    elif pattern.parts[p_idx] == "*":
        return _match_pattern_recursive(pattern, target, p_idx + 1, t_idx + 1)
    else:
        if pattern.parts[p_idx] == target.parts[t_idx]:
            return _match_pattern_recursive(pattern, target, p_idx + 1, t_idx + 1)
        return False


def is_pattern(nodekey: NodeKey) -> bool:
    """Check if a node key contains wildcard patterns."""
    return any("*" in part or "**" in part for part in nodekey.parts)


def match_pattern(pattern: NodeKey, target: NodeKey) -> bool:
    """Match a pattern against a target NodeKey.

    Supports wildcards:
    * - matches exactly one part
    ** - matches zero or more parts

    Args:
        pattern: The pattern to match against
        target: The target to match

    Returns:
        bool: True if pattern matches target, False otherwise
    """
    return _match_pattern_recursive(pattern, target, 0, 0)
