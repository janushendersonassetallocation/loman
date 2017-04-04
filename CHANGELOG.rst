Change Log
==========

`0.1.0`_ (unreleased)
---------------------

* Added documentation: Introduction, Quickstart and Strategies for Use
* Added docstrings to core Computation methods
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