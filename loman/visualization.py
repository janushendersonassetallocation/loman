import matplotlib as mpl
import networkx as nx
import pydotplus
import six

from loman.consts import NodeAttributes, States

state_colors = {
    None: '#ffffff',                    # xkcd white
    States.PLACEHOLDER: '#f97306',      # xkcd orange
    States.UNINITIALIZED: '#0343df',    # xkcd blue
    States.STALE: '#ffff14',            # xkcd yellow
    States.COMPUTABLE: '#9dff00',       # xkcd bright yellow green
    States.UPTODATE: '#15b01a',         # xkcd green
    States.ERROR: '#e50000',            # xkcd red
    States.PINNED: '#bf77f6'            # xkcd light purple
}


def create_viz_dag(comp_dag, colors='state', cmap=None):
    colors = colors.lower()
    if colors == 'state':
        if cmap is None:
            cmap = state_colors
    elif colors == 'timing':
        if cmap is None:
            cmap = mpl.colors.LinearSegmentedColormap.from_list('blend', ['#15b01a', '#ffff14', '#e50000'])
        timings = nx.get_node_attributes(comp_dag, NodeAttributes.TIMING)
        max_duration = max(timing.duration for timing in six.itervalues(timings) if hasattr(timing, 'duration'))
        min_duration = min(timing.duration for timing in six.itervalues(timings) if hasattr(timing, 'duration'))
    else:
        raise ValueError('{} is not a valid loman colors parameter for visualization'.format(colors))

    viz_dag = nx.DiGraph()
    node_index_map = {}
    for i, (name, data) in enumerate(comp_dag.nodes(data=True)):
        short_name = "n{}".format(i)
        attr_dict = {
            'label': name,
            'style': 'filled',
            '_group': data.get(NodeAttributes.GROUP)
        }

        if colors == 'state':
            attr_dict['fillcolor'] = cmap[data.get(NodeAttributes.STATE, None)]
        elif colors == 'timing':
            timing_data = data.get(NodeAttributes.TIMING)
            if timing_data is None:
                col = '#FFFFFF'
            else:
                duration = timing_data.duration
                norm_duration = (duration - min_duration) / (max_duration - min_duration)
                col = mpl.colors.rgb2hex(cmap(norm_duration))
            attr_dict['fillcolor'] = col

        viz_dag.add_node(short_name, **attr_dict)
        node_index_map[name] = short_name
    for name1, name2 in comp_dag.edges():
        short_name_1 = node_index_map[name1]
        short_name_2 = node_index_map[name2]

        group1 = comp_dag.node[name1].get(NodeAttributes.GROUP)
        group2 = comp_dag.node[name2].get(NodeAttributes.GROUP)
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
        for k, v in six.iteritems(graph_attr):
            viz_dot.set(k, v)

    if node_attr is not None:
        viz_dot.set_node_defaults(**node_attr)

    if edge_attr is not None:
        viz_dot.set_edge_defaults(**edge_attr)

    for group, names in six.iteritems(node_groups):
        if group is None:
            continue
        c = pydotplus.Subgraph('cluster_' + str(group))

        for name in names:
            node = pydotplus.Node(name)
            for k, v in six.iteritems(viz_dag.node[name]):
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
        for k, v in six.iteritems(viz_dag.node[name]):
            if not k.startswith("_"):
                node.set(k, v)
        viz_dot.add_node(node)

    for name1, name2 in edge_groups.get(None, []):
        edge = pydotplus.Edge(name1, name2)
        viz_dot.add_edge(edge)

    return viz_dot