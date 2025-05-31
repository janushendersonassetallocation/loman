import os
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
from typing import Optional, List, Union

import matplotlib as mpl
import networkx as nx
import numpy as np
import pydotplus
from matplotlib.colors import Colormap

import loman
from .nodekey import NodeKey, Name, to_nodekey, match_pattern, is_pattern
from .consts import NodeAttributes, States, NodeTransformations
from .graph_utils import contract_node

@dataclass
class Node:
    nodekey: NodeKey
    original_nodekey: NodeKey
    data: dict


class NodeFormatter(ABC):
    def calibrate(self, nodes: List[Node]) -> None:
        pass

    @abstractmethod
    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        pass

    @staticmethod
    def create(cmap: Optional[Union[dict, Colormap]] = None, colors: str = 'state', shapes: Optional[str] = None):
        node_formatters = [StandardLabel(), StandardGroup()]

        if isinstance(shapes, str):
            shapes = shapes.lower()
        if shapes == 'type':
            node_formatters.append(ShapeByType())
        elif shapes is None:
            pass
        else:
            raise ValueError(f"{shapes} is not a valid loman shapes parameter for visualization")

        colors = colors.lower()
        if colors == 'state':
            node_formatters.append(ColorByState(cmap))
        elif colors == 'timing':
            node_formatters.append(ColorByTiming(cmap))
        else:
            raise ValueError(f"{colors} is not a valid loman colors parameter for visualization")

        node_formatters.append(StandardStylingOverrides())
        node_formatters.append(RectBlocks())

        return CompositeNodeFormatter(node_formatters)


class ColorByState(NodeFormatter):
    DEFAULT_STATE_COLORS = {
        None: '#ffffff',  # xkcd white
        States.PLACEHOLDER: '#f97306',  # xkcd orange
        States.UNINITIALIZED: '#0343df',  # xkcd blue
        States.STALE: '#ffff14',  # xkcd yellow
        States.COMPUTABLE: '#9dff00',  # xkcd bright yellow green
        States.UPTODATE: '#15b01a',  # xkcd green
        States.ERROR: '#e50000',  # xkcd red
        States.PINNED: '#bf77f6'  # xkcd light purple
    }

    def __init__(self, state_colors=None):
        if state_colors is None:
            state_colors = self.DEFAULT_STATE_COLORS.copy()
        self.state_colors = state_colors

    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        states = [node.data.get(NodeAttributes.STATE, None) for node in nodes]
        if len(nodes) == 1:
            state = states[0]
        else:
            if any(state == States.ERROR for state in states):
                state = States.ERROR
            elif any(state == States.STALE for state in states):
                state = States.STALE
            else:
                state0 = states[0]
                if all(state == state0 for state in states):
                    state = state0
                else:
                    state = None
        return {
            'style': 'filled',
            'fillcolor': self.state_colors[state]
        }


class ColorByTiming(NodeFormatter):
    def __init__(self, cmap: Optional[Colormap] = None):
        if cmap is None:
            cmap = mpl.colors.LinearSegmentedColormap.from_list('blend', ['#15b01a', '#ffff14', '#e50000'])
        self.cmap = cmap
        self.min_duration = np.nan
        self.max_duration = np.nan

    def calibrate(self, nodes: List[Node]) -> None:
        durations = []
        for node in nodes:
            timing = node.data.get(NodeAttributes.TIMING)
            if timing is not None:
                durations.append(timing.duration)
        self.max_duration = max(durations)
        self.min_duration = min(durations)

    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        if len(nodes) == 1:
            data = nodes[0].data
            timing_data = data.get(NodeAttributes.TIMING)
            if timing_data is None:
                col = '#FFFFFF'
            else:
                duration = timing_data.duration
                norm_duration: float = (duration - self.min_duration) / max(1e-8, self.max_duration - self.min_duration)
                col = mpl.colors.rgb2hex(self.cmap(norm_duration))
            return {
                'style': 'filled',
                'fillcolor': col
            }


class ShapeByType(NodeFormatter):
    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        if len(nodes) == 1:
            data = nodes[0].data
            value = data.get(NodeAttributes.VALUE)
            if value is None:
                return
            if isinstance(value, np.ndarray):
                return {'shape': 'rect'}
            elif isinstance(value, pd.DataFrame):
                return {'shape': 'box3d'}
            elif np.isscalar(value):
                return {'shape': 'ellipse'}
            elif isinstance(value, (list, tuple)):
                return {'shape': 'ellipse', 'peripheries': 2}
            elif isinstance(value, dict):
                return {'shape': 'house', 'peripheries': 2}
            elif isinstance(value, loman.Computation):
                return {'shape': 'hexagon'}
            else:
                return {'shape': 'diamond'}


class RectBlocks(NodeFormatter):
    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        if is_composite:
            return {'shape': 'rect', 'peripheries': 2}


class StandardLabel(NodeFormatter):
    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        return {'label': name.label}


def get_group_path(name: NodeKey, data: dict) -> NodeKey:
    name_group_path = name.parent
    attribute_group = data.get(NodeAttributes.GROUP)
    attribute_group_path = None if attribute_group is None else NodeKey((attribute_group,))

    group_path = name_group_path.join(attribute_group_path)
    return group_path


class StandardGroup(NodeFormatter):
    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        if len(nodes) == 1:
            data = nodes[0].data
            group_path = get_group_path(name, data)
        else:
            group_path = name.parent
        if group_path.is_root:
            return None
        return {'_group': group_path}


class StandardStylingOverrides(NodeFormatter):
    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        if len(nodes) == 1:
            data = nodes[0].data
            style = data.get(NodeAttributes.STYLE)
            if style is None:
                return
            if style == 'small':
                return {'width': 0.3, 'height': 0.2, 'fontsize': 8}
            elif style == 'dot':
                return {'shape': 'point', 'width': 0.1, 'peripheries': 1}


@dataclass
class CompositeNodeFormatter(NodeFormatter):
    formatters: List[NodeFormatter] = field(default_factory=list)

    def calibrate(self, nodes: List[Node]) -> None:
        for formatter in self.formatters:
            formatter.calibrate(nodes)

    def format(self, name: NodeKey, nodes: List[Node], is_composite: bool) -> Optional[dict]:
        d = {}
        for formatter in self.formatters:
            format_attrs = formatter.format(name, nodes, is_composite)
            if format_attrs is not None:
                d.update(format_attrs)
        return d


@dataclass
class GraphView:
    computation: 'loman.Computation'
    root: Optional[Name] = None
    node_formatter: Optional[NodeFormatter] = None
    node_transformations: Optional[dict] = None
    collapse_all: bool = True

    graph_attr: Optional[dict] = None
    node_attr: Optional[dict] = None
    edge_attr: Optional[dict] = None

    struct_dag: Optional[nx.DiGraph] = None
    viz_dag: Optional[nx.DiGraph] = None
    viz_dot: Optional[pydotplus.Dot] = None

    def __post_init__(self):
        self.refresh()

    @staticmethod
    def get_sub_block(dag, root, node_transformations: dict):
        d_transform_to_nodes = defaultdict(list)
        for nk, transform in node_transformations.items():
            d_transform_to_nodes[transform].append(nk)

        dag_out = nx.DiGraph()

        d_original_to_mapped = {}
        s_collapsed = set()

        for nk_original in dag.nodes():
            nk_mapped = nk_original.drop_root(root)
            if nk_mapped is None:
                continue
            nk_highest_collapse = nk_original
            is_collapsed = False
            for nk_collapse in d_transform_to_nodes[NodeTransformations.COLLAPSE]:
                if nk_highest_collapse.is_descendent_of(nk_collapse):
                    nk_highest_collapse = nk_collapse
                    is_collapsed = True
            nk_mapped = nk_highest_collapse.drop_root(root)
            if nk_mapped is None:
                continue
            d_original_to_mapped[nk_original] = nk_mapped
            if is_collapsed:
                s_collapsed.add(nk_mapped)

        for nk_mapped in d_original_to_mapped.values():
            dag_out.add_node(nk_mapped)

        for nk_u, nk_v in dag.edges():
            nk_mapped_u = d_original_to_mapped.get(nk_u)
            nk_mapped_v = d_original_to_mapped.get(nk_v)
            if nk_mapped_u is None or nk_mapped_v is None or nk_mapped_u == nk_mapped_v:
                continue
            dag_out.add_edge(nk_mapped_u, nk_mapped_v)

        for nk in d_transform_to_nodes[NodeTransformations.CONTRACT]:
            contract_node(dag_out, d_original_to_mapped[nk])
            del d_original_to_mapped[nk]

        d_mapped_to_original = defaultdict(list)
        for nk_original, nk_mapped in d_original_to_mapped.items():
            if nk_mapped in dag_out.nodes:
                d_mapped_to_original[nk_mapped].append(nk_original)

        s_collapsed.intersection_update(dag_out.nodes)

        return dag_out, d_mapped_to_original, s_collapsed

    def _initialize_transforms(self):
        node_transformations = {}
        if self.collapse_all:
            self._apply_default_collapse_transforms(node_transformations)
        self._apply_custom_transforms(node_transformations)
        return node_transformations

    def _apply_default_collapse_transforms(self, node_transformations):
        for n in self.computation.get_tree_descendents(self.root):
            nk = to_nodekey(n)
            if not self.computation.has_node(nk):
                node_transformations[nk] = NodeTransformations.COLLAPSE

    def _apply_custom_transforms(self, node_transformations):
        if self.node_transformations is not None:
            for rule_name, transform in self.node_transformations.items():
                include_ancestors = transform == NodeTransformations.EXPAND
                rule_nk = to_nodekey(rule_name)
                if is_pattern(rule_nk):
                    apply_nodes = set()
                    for n in self.computation.get_tree_descendents(self.root):
                        nk = to_nodekey(n)
                        if match_pattern(rule_nk, nk):
                            apply_nodes.add(nk)
                else:
                    apply_nodes = {rule_nk}
                    node_transformations[rule_nk] = transform
                if include_ancestors:
                    for nk in apply_nodes:
                        for nk1 in nk.ancestors():
                            if nk1.is_root or nk1 == self.root:
                                break
                            node_transformations[nk1] = NodeTransformations.EXPAND
                for rule_nk in apply_nodes:
                    node_transformations[rule_nk] = transform
        return node_transformations

    def _create_visualization_dag(self, original_nodes, composite_nodes):
        node_formatter = self.node_formatter
        if node_formatter is None:
            node_formatter = NodeFormatter.create()
        return create_viz_dag(self.struct_dag, self.computation.dag, node_formatter, original_nodes, composite_nodes)

    def _create_dot_graph(self):
        return to_pydot(self.viz_dag, self.graph_attr, self.node_attr, self.edge_attr)

    def refresh(self):
        node_transformations = self._initialize_transforms()
        self.struct_dag, original_nodes, composite_nodes = self.get_sub_block(self.computation.dag, self.root, node_transformations)
        self.viz_dag = self._create_visualization_dag(original_nodes, composite_nodes)
        self.viz_dot = self._create_dot_graph()

    def svg(self) -> Optional[str]:
        if self.viz_dot is None:
            return None
        return self.viz_dot.create_svg().decode('utf-8')

    def view(self):
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(self.viz_dot.create_pdf())
            os.startfile(f.name)

    def _repr_svg_(self):
        return self.svg()


def create_viz_dag(struct_dag, comp_dag, node_formatter: NodeFormatter, original_nodes: dict, composite_nodes: set) -> nx.DiGraph:
    if node_formatter is not None:
        nodes = []
        for nodekey in struct_dag.nodes:
            for original_nodekey in original_nodes[nodekey]:
                data = comp_dag.nodes[original_nodekey]
                node = Node(nodekey, original_nodekey, data)
                nodes.append(node)
        node_formatter.calibrate(nodes)

    viz_dag = nx.DiGraph()
    node_index_map = {}
    for i, nodekey in enumerate(struct_dag.nodes):
        short_name = f'n{i}'
        attr_dict = None

        if node_formatter is not None:
            nodes = []
            for original_nodekey in original_nodes[nodekey]:
                data = comp_dag.nodes[original_nodekey]
                node = Node(nodekey, original_nodekey, data)
                nodes.append(node)
            is_composite = nodekey in composite_nodes
            attr_dict = node_formatter.format(nodekey, nodes, is_composite)
        if attr_dict is None:
            attr_dict = {}

        attr_dict = {k: v for k, v in attr_dict.items() if v is not None}

        viz_dag.add_node(short_name, **attr_dict)
        node_index_map[nodekey] = short_name

    for name1, name2 in struct_dag.edges():
        short_name_1 = node_index_map[name1]
        short_name_2 = node_index_map[name2]

        group_path1 = get_group_path(name1, struct_dag.nodes[name1])
        group_path2 = get_group_path(name2, struct_dag.nodes[name2])
        group_path = NodeKey.common_parent(group_path1, group_path2)

        attr_dict = {}
        if not group_path.is_root:
            #group_path = None
            attr_dict['_group'] = group_path

        viz_dag.add_edge(short_name_1, short_name_2, **attr_dict)

    return viz_dag


def to_pydot(viz_dag, graph_attr=None, node_attr=None, edge_attr=None) -> pydotplus.Dot:
    root = NodeKey.root()

    node_groups = {}
    for name, data in viz_dag.nodes(data=True):
        group = data.get('_group', root)
        node_groups.setdefault(group, []).append(name)

    edge_groups = {}
    for name1, name2, data in viz_dag.edges(data=True):
        group = data.get('_group', root)
        edge_groups.setdefault(group, []).append((name1, name2))

    subgraphs = {root: create_root_graph(graph_attr, node_attr, edge_attr)}

    for group, names in node_groups.items():
        c = subgraphs[root] if group is root else create_subgraph(group)

        for name in names:
            node = pydotplus.Node(name)
            for k, v in viz_dag.nodes[name].items():
                if not k.startswith("_"):
                    node.set(k, v)
            c.add_node(node)

        subgraphs[group] = c

    groups = list(subgraphs.keys())
    for group in groups:
        group1 = group
        while True:
            if group1.is_root:
                break
            group1 = group1.parent
            if group1 in subgraphs:
                break
            subgraphs[group1] = create_subgraph(group1)

    for group, subgraph in subgraphs.items():
        if group.is_root:
            continue
        parent = group
        while True:
            parent = parent.parent
            if parent in subgraphs or parent.is_root:
                break
        subgraphs[parent].add_subgraph(subgraph)

    for group, edges in edge_groups.items():
        c = subgraphs[group]
        for name1, name2 in edges:
            edge = pydotplus.Edge(name1, name2)
            c.add_edge(edge)

    return subgraphs[root]


def create_root_graph(graph_attr, node_attr, edge_attr):
    root_graph = pydotplus.Dot()
    if graph_attr is not None:
        for k, v in graph_attr.items():
            root_graph.set(k, v)
    if node_attr is not None:
        root_graph.set_node_defaults(**node_attr)
    if edge_attr is not None:
        root_graph.set_edge_defaults(**edge_attr)
    return root_graph

def create_subgraph(group: NodeKey):
    c = pydotplus.Subgraph('cluster_' + str(group))
    c.obj_dict['attributes']['label'] = str(group)
    return c
