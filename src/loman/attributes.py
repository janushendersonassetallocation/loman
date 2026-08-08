"""Tag, style and metadata accessors for computation nodes.

These are the annotations a node carries alongside its state and value: tags
(including the system tags that drive serialization and expansion), the
GraphViz style name, and free-form metadata.
"""

from collections.abc import Iterable
from typing import Any, overload

from .consts import NodeAttributes
from .exception import NonExistentNodeException
from .nodekey import Name, Names, NodeKey, node_keys_to_names, to_nodekey
from .query import QueryMixin
from .util import apply1, apply_n


class AttributeMixin(QueryMixin):
    """Reading and writing node tags, styles and metadata."""

    def _set_tag_one(self, name: Name, tag: str) -> None:
        """Set a single tag on a single node."""
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.TAG].add(tag)
        self._tag_map[tag].add(node_key)

    def set_tag(self, name: Name | Names, tag: str | Iterable[str]) -> None:
        """Set tags on a node or nodes. Ignored if tags are already set.

        :param name: Node or nodes to set tag for
        :param tag: Tag to set
        """
        apply_n(self._set_tag_one, name, tag)

    def _clear_tag_one(self, name: Name, tag: str) -> None:
        """Clear a single tag from a single node."""
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.TAG].discard(tag)
        self._tag_map[tag].discard(node_key)

    def clear_tag(self, name: Name | Names, tag: str | Iterable[str]) -> None:
        """Clear tag on a node or nodes. Ignored if tags are not set.

        :param name: Node or nodes to clear tags for
        :param tag: Tag to clear
        """
        apply_n(self._clear_tag_one, name, tag)

    def _tag_one(self, name: Name) -> set[str]:
        """Get the tags of a single node."""
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        tags: set[str] = node[NodeAttributes.TAG]
        return tags

    @overload
    def tags(self, name: Name) -> set[str]: ...

    @overload
    def tags(self, name: Names) -> list[set[str]]: ...

    def tags(self, name: Name | Names) -> set[str] | list[set[str]]:
        """Get the tags associated with a node.

            >>> from loman import Computation
            >>> comp = Computation()
            >>> comp.add_node('a', tags=['foo', 'bar'])
            >>> sorted(comp.t.a)
            ['__serialize__', 'bar', 'foo']

        :param name: Name or names of the node to get the tags of
        :return:
        """
        return apply1(self._tag_one, name)

    def nodes_by_tag(self, tag: str | Iterable[str]) -> set[Name]:
        """Get the names of nodes with a particular tag or tags.

        :param tag: Tag or tags for which to retrieve nodes
        :return: Names of the nodes with those tags
        """
        nodes: set[NodeKey] = set()
        tags_to_check: Iterable[str] = [tag] if isinstance(tag, str) else tag
        for tag1 in tags_to_check:
            nodes1 = self._tag_map.get(tag1)
            if nodes1 is not None:
                nodes.update(nodes1)
        return {n.name for n in nodes}

    def _get_tags_for_state(self, tag: str) -> set[Name]:
        """Get node names that have a specific tag."""
        return set(node_keys_to_names(self._tag_map[tag]))

    def _set_style_one(self, name: Name, style: str) -> None:
        """Set style on a single node."""
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.STYLE] = style

    def set_style(self, name: Name | Names, style: str) -> None:
        """Set styles on a node or nodes.

        :param name: Node or nodes to set style for
        :param style: Style to set
        """
        apply_n(self._set_style_one, name, style)

    def _clear_style_one(self, name: Name) -> None:
        """Clear style from a single node."""
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.STYLE] = None

    def clear_style(self, name: Name | Names) -> None:
        """Clear style on a node or nodes.

        :param name: Node or nodes to clear styles for
        """
        apply_n(self._clear_style_one, name)

    def _style_one(self, name: Name) -> str | None:
        """Get the style of a single node."""
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        style: str | None = node.get(NodeAttributes.STYLE)
        return style

    @overload
    def styles(self, name: Name) -> str | None: ...

    @overload
    def styles(self, name: Names) -> list[str | None]: ...

    def styles(self, name: Name | Names) -> str | None | list[str | None]:
        """Get the tags associated with a node.

            >>> from loman import Computation
            >>> comp = Computation()
            >>> comp.add_node('a', style='dot')
            >>> comp.style.a
            'dot'

        :param name: Name or names of the node to get the tags of
        :return:
        """
        return apply1(self._style_one, name)

    def metadata(self, name: Name) -> dict[str, Any]:
        """Get metadata for a node."""
        node_key = to_nodekey(name)
        if self.tree_has_path(name):
            if node_key not in self._metadata:
                self._metadata[node_key] = {}
            result: dict[str, Any] = self._metadata[node_key]
            return result
        else:
            msg = f"Node {node_key} does not exist."
            raise NonExistentNodeException(msg)
