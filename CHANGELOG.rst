Change Log
==========

`0.1.2`_ (Unreleased)
---------------------

* Add node function decorator

`0.1.1`_ (2017-04-25)
---------------------

* Support for Python 3.4 and 3.5
* Method and attribute-style accessors support lists of nodes
* Added support for node-tagging
* Compute method can optionally throw exceptions, for easier interactive debugging
* ``get_inputs`` method and ``i`` attribute-style access to get list of inputs to a node
* ``add_node`` takes optional inspect parameter to avoid inspection for performance
* ``add_node`` takes optional group to render graph layout with subgraphs
* ``draw_graphviz`` renamed to ``draw``
* ``draw_nx`` removed
* ``get_df`` renamed to ``to_df``
* ``get_value_dict`` renamed to ``to_dict``
* BUGFIX: implementation of _get_calc_nodes used by compute fixed
* BUGFIX: args parameters do not create spurious nodes
* BUGFIX: default function parameters do not cause placeholder node to be created
* BUGFIX: node states correctly updated when calling add_node with value parameter

`0.1.0`_ (2017-04-05)
---------------------

* Added documentation: Introduction, Quickstart and Strategies for Use
* Added docstrings to Computation methods
* Added logging
* Added ``v`` and ``s`` fields for attribute-style access to values and states of nodes
* BUGFIX: Detect cycles in ``compute_all``

`0.0.1`_ (2017-03-24)
---------------------

* Computation object with ``add_node``, ``insert``, ``compute``, ``compute_all``, ``state``, ``value``, ``set_stale`` methods
* Computation object can be drawn with ``draw_graphviz`` method
* Nodes can be updated in place
* Computation handles exceptions in node computation, storing exception and traceback
* Can specify mapping between function parameters and input nodes
* Convenience methods: ``add_named_tuple_expansion``, ``add_map_node``, ``get_df``, ``get_value_dict``, ``insert_from``, ``insert_multi``
* Convenience method
* Computation objects can be serialized
* Computation objects can be shallow-copied with ``copy``
* Unit tests
* Runs under Python 2.7, 3.6