import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd
from typing import Optional, List, Union

import matplotlib as mpl
import networkx as nx
import numpy as np
import pydotplus
from matplotlib.colors import Colormap

import loman
from .path_parser import path_join, Path, path_common_parent
from .structs import NodeKey, InputName
from .consts import NodeAttributes, States, NodeTransformations
from .graph_utils import contract_node


class NodeFormatter(ABC):
    def calibrate(self, nodes) -> None:
        pass

    @abstractmethod
    def format(self, name: NodeKey, data) -> Optional[dict]:
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

    def format(self, name: NodeKey, data):
        return {
            'style': 'filled',
            'fillcolor': self.state_colors[data.get(NodeAttributes.STATE, None)]
        }


class ColorByTiming(NodeFormatter):
    def __init__(self, cmap: Optional[Colormap] = None):
        if cmap is None:
            cmap = mpl.colors.LinearSegmentedColormap.from_list('blend', ['#15b01a', '#ffff14', '#e50000'])
        self.cmap = cmap
        self.min_duration = np.nan
        self.max_duration = np.nan

    def calibrate(self, nodes):
        durations = []
        for name, data in nodes:
            timing = data.get(NodeAttributes.TIMING)
            if timing is not None:
                durations.append(timing.duration)
        self.max_duration = max(durations)
        self.min_duration = min(durations)

    def format(self, name: NodeKey, data):
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
    def format(self, name: NodeKey, data):
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


class StandardLabel(NodeFormatter):
    def format(self, name: NodeKey, data):
        return {'label': name.label}


def get_group_path(name: NodeKey, data: dict) -> Path:
    name_group_path = name.group_path
    attribute_group = data.get(NodeAttributes.GROUP)
    attribute_group_path = None if attribute_group is None else Path((attribute_group,))

    group_path = path_join(name_group_path, attribute_group_path)
    return group_path


class StandardGroup(NodeFormatter):
    def format(self, name: NodeKey, data):
        group_path = get_group_path(name, data)
        if group_path.is_root:
            return None
        return {'_group': group_path}


class StandardStylingOverrides(NodeFormatter):
    def format(self, name, data):
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

    def calibrate(self, nodes):
        for formatter in self.formatters:
            formatter.calibrate(nodes)

    def format(self, name, data):
        d = {}
        for formatter in self.formatters:
            format_attrs = formatter.format(name, data)
            if format_attrs is not None:
                d.update(format_attrs)
        return d


@dataclass
class GraphView:
    computation: 'loman.Computation'
    root: Optional[InputName] = None
    node_formatter: Optional[NodeFormatter] = None
    node_transformations: Optional[dict] = None

    graph_attr: Optional[dict] = None
    node_attr: Optional[dict] = None
    edge_attr: Optional[dict] = None

    struct_dag: Optional[nx.DiGraph] = None
    viz_dag: Optional[nx.DiGraph] = None
    viz_dot: Optional[pydotplus.Dot] = None

    def __post_init__(self):
        self.refresh()

    @staticmethod
    def get_sub_block(dag, root, node_transformations):
        dag_out = nx.DiGraph()

        nodekey_map = {}

        nodes_to_contract = []

        for nodekey, data in dag.nodes(data=True):
            new_nodekey = nodekey.drop_root(root)
            if new_nodekey is None:
                continue
            dag_out.add_node(new_nodekey, **data)
            nodekey_map[nodekey] = new_nodekey
            if node_transformations.get(nodekey) == NodeTransformations.CONTRACT:
                nodes_to_contract.append(new_nodekey)

        for nodekey_u, nodekey_v, data in dag.edges(data=True):
            new_nodekey_u = nodekey_map.get(nodekey_u)
            new_nodekey_v = nodekey_map.get(nodekey_v)
            if new_nodekey_u is None or new_nodekey_v is None:
                continue
            dag_out.add_edge(new_nodekey_u, new_nodekey_v, **data)

        contract_node(dag_out, nodes_to_contract)

        return dag_out

    def refresh(self):
        node_transformations = {NodeKey.from_name(k): v for k, v in self.node_transformations.items()} if self.node_transformations is not None else {}
        self.struct_dag = self.get_sub_block(self.computation.dag, self.root, node_transformations)

        node_formatter = self.node_formatter
        if node_formatter is None:
            node_formatter = NodeFormatter.create()
        self.viz_dag = create_viz_dag(self.struct_dag, node_formatter)

        self.viz_dot = to_pydot(self.viz_dag, self.graph_attr, self.node_attr, self.edge_attr)

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


def create_viz_dag(comp_dag, node_formatter: Optional[NodeFormatter] = None) -> nx.DiGraph:
    if node_formatter is not None:
        node_formatter.calibrate(comp_dag.nodes(data=True))

    viz_dag = nx.DiGraph()
    node_index_map = {}
    for i, (name, data) in enumerate(comp_dag.nodes(data=True)):
        short_name = f'n{i}'
        attr_dict = None

        if node_formatter is not None:
            attr_dict = node_formatter.format(name, data)
        if attr_dict is None:
            attr_dict = {}

        attr_dict = {k: v for k, v in attr_dict.items() if v is not None}

        viz_dag.add_node(short_name, **attr_dict)
        node_index_map[name] = short_name

    for name1, name2 in comp_dag.edges():
        short_name_1 = node_index_map[name1]
        short_name_2 = node_index_map[name2]

        group_path1 = get_group_path(name1, comp_dag.nodes[name1])
        group_path2 = get_group_path(name2, comp_dag.nodes[name2])
        group_path = path_common_parent(group_path1, group_path2)

        attr_dict = {}
        if not group_path.is_root:
            #group_path = None
            attr_dict['_group'] = group_path

        viz_dag.add_edge(short_name_1, short_name_2, **attr_dict)

    return viz_dag


def to_pydot(viz_dag, graph_attr=None, node_attr=None, edge_attr=None) -> pydotplus.Dot:
    root = Path(tuple(), False)

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
            group1 = group1.parent()
            if group1 in subgraphs:
                break
            subgraphs[group1] = create_subgraph(group1)

    for group, subgraph in subgraphs.items():
        if group.is_root:
            continue
        parent = group
        while True:
            parent = parent.parent()
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

def create_subgraph(group: Path):
    c = pydotplus.Subgraph('cluster_' + str(group))
    c.obj_dict['attributes']['label'] = str(group)
    return c
