from typing import Optional

import matplotlib as mpl
import networkx as nx
import numpy as np
import pydotplus
from matplotlib.colors import Colormap

from loman.consts import NodeAttributes, States


class NodeFormatter:
    def calibrate(self, nodes):
        pass

    def format(self, name, data):
        pass


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

    def format(self, name, data):
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

    def format(self, name, data):
        timing_data = data.get(NodeAttributes.TIMING)
        if timing_data is None:
            col = '#FFFFFF'
        else:
            duration = timing_data.duration
            norm_duration = (duration - self.min_duration) / (self.max_duration - self.min_duration)
            col = mpl.colors.rgb2hex(self.cmap(norm_duration))
        return {
            'style': 'filled',
            'fillcolor': col
        }


class StandardLabel(NodeFormatter):
    def format(self, name, data):
        return {
            'label': str(name)
        }


class StandardGroup(NodeFormatter):
    def format(self, name, data):
        group = data.get(NodeAttributes.GROUP)
        if group is None:
            return None
        return {
            '_group': group
        }


class StandardStylingOverrides(NodeFormatter):
    def format(self, name, data):
        tags = data.get(NodeAttributes.TAG)
        if 'style_small' in tags:
            return {'width': 0.3, 'height': 0.2, 'fontsize': 8}
        elif 'style_dot' in tags:
            return {'shape': 'point', 'width': 0.1}


def get_node_formatters(cmap, colors):
    node_formatters = [StandardLabel(), StandardGroup()]
    colors = colors.lower()
    if colors == 'state':
        node_formatters.append(ColorByState(cmap))
    elif colors == 'timing':
        node_formatters.append(ColorByTiming(cmap))
    else:
        raise ValueError(f"{colors} is not a valid loman colors parameter for visualization")
    node_formatters.append(StandardStylingOverrides())
    return node_formatters


def create_viz_dag(comp_dag, colors='state', cmap=None):
    node_formatters = get_node_formatters(cmap, colors)

    for node_formatter in node_formatters:
        node_formatter.calibrate(comp_dag.nodes(data=True))

    viz_dag = nx.DiGraph()
    node_index_map = {}
    for i, (name, data) in enumerate(comp_dag.nodes(data=True)):
        short_name = f'n{i}'
        attr_dict = {}

        for node_formatter in node_formatters:
            format_attrs = node_formatter.format(name, data)
            if format_attrs is not None:
                attr_dict.update(format_attrs)

        attr_dict = {k: v for k, v in attr_dict.items() if v is not None}

        viz_dag.add_node(short_name, **attr_dict)
        node_index_map[name] = short_name

    for name1, name2 in comp_dag.edges():
        short_name_1 = node_index_map[name1]
        short_name_2 = node_index_map[name2]

        group1 = comp_dag.nodes[name1].get(NodeAttributes.GROUP)
        group2 = comp_dag.nodes[name2].get(NodeAttributes.GROUP)
        group = group1 if group1 == group2 else None

        attr_dict = {'_group': group}

        viz_dag.add_edge(short_name_1, short_name_2, **attr_dict)

    return viz_dag


def to_pydot(viz_dag, graph_attr=None, node_attr=None, edge_attr=None):
    node_groups = {}
    for name, data in viz_dag.nodes(data=True):
        group = data.get('_group')
        node_groups.setdefault(group, []).append(name)

    edge_groups = {}
    for name1, name2, data in viz_dag.edges(data=True):
        group = data.get('_group')
        edge_groups.setdefault(group, []).append((name1, name2))

    viz_dot = pydotplus.Dot()

    if graph_attr is not None:
        for k, v in graph_attr.items():
            viz_dot.set(k, v)

    if node_attr is not None:
        viz_dot.set_node_defaults(**node_attr)

    if edge_attr is not None:
        viz_dot.set_edge_defaults(**edge_attr)

    for group, names in node_groups.items():
        if group is None:
            continue
        c = pydotplus.Subgraph('cluster_' + str(group))

        for name in names:
            node = pydotplus.Node(name)
            for k, v in viz_dag.nodes[name].items():
                if not k.startswith("_"):
                    node.set(k, v)
            c.add_node(node)

        for name1, name2 in edge_groups.get(group, []):
            edge = pydotplus.Edge(name1, name2)
            c.add_edge(edge)

        c.obj_dict['label'] = str(group)
        viz_dot.add_subgraph(c)

    for name in node_groups.get(None, []):
        node = pydotplus.Node(name)
        for k, v in viz_dag.nodes[name].items():
            if not k.startswith("_"):
                node.set(k, v)
        viz_dot.add_node(node)

    for name1, name2 in edge_groups.get(None, []):
        edge = pydotplus.Edge(name1, name2)
        viz_dot.add_edge(edge)

    return viz_dot
