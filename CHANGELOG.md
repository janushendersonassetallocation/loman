# Change Log

## [unreleased]
- Added type hints on ComputationFactory

## [0.5.2] (TBD)

- Added support for pattern matching in node transformations, including wildcard patterns
- Add nested attribute views, so comp.v.foo.bar.baz is equivalent to comp.v['foo/bar/baz']
- Set COLLAPSE as default node transformation, and added EXPANDED NodeTransformation type (ancestors of expanded nodes are automatically expanded)
- Added `collapse_all` flag to GraphView to support backward compatibility
- Cleaned up GraphView.refresh implementation
- Moved Path functionality to NodeKey

## [0.5.1] (2025-05-21)

- Add root parameter to Computation.draw to support viewing sub-blocks.
- Add NodeTransformations, including a new COLLAPSE node transformation
- Modify add_node so that argument names of supplied function will look up within same block, rather than root block
- Add links parameter to Computation.add_block
- Add keep_values parameter to Computation.add_block
- Blocks show state if all blocks have same state (or error or stale if any do)
- BUGFIX: Linking a node to itself is a no-op
- BUGFIX: Inserting to a placeholder node raises a specific exception
- BUGFIX: Composite blocks retain sub-block on collapsing

## [0.5.0] (2025-04-10)

- Add support for blocks (Computation.add_block)
- Add support for links (Computation.link)
- Nodes keyed using NodeKey with paths to support nested blocks
- Visualization modified to support grouping elements in same block
- BUGFIX: Fix calc nodes with no parameters
- Switched to use Python build front-end

## [0.4.1] (2024-11-29)

- If first parameter of a `@calc_node` is called `self`, then it can be used to call non-calc_node methods of the class. (Can be disabled with `@ComputationFactory(ignore_self=False)` or `@calc_node(ignore_self=False)` ).
- Add support for convertors to force input and calc node values to a particular type/form
- Add support for serializing nodes that are computations
- create_viz_dag now takes a list of node_formatters, which apply arbitrary formatting to the visualization node based on the lomnan node (Included NodeFormatters are `ColorByState`, `ColorByTiming`, `ShapeByType`, `StandardLabel`, `StandardGroup`, `StandardStylingOverrides`)
- Fix ReadTheDocs build
- Convert documentation from reStructuredText to MyST Markdown

## [0.4.0] (2024-08-22)

- Removed Python 2 support
- Changed test framework from nose to pytest
- Add `compute_and_get_value`, `x` attribute-style access to compute value of a node and get it in one step
- Replace namedtuples with dataclasses
- BUGFIX: Fix equality testing on Computation.insert
- Use DataFrame.equals and Series.equals to test equality in Computation.insert
- BUGFIX: Fix handling of groups in rendering functions

## [0.3.0] (2019-10-24)

- Added `get_original_inputs` to see source inputs of entire computation or a given set of nodes
- Added `get_outputs`, `o` attribute-style access to get list of nodes fed by a particular node
- Added `get_final_outputs` to get end nodes of a computation or a given set of nodes
- Added `restrict` method to remove nodes unnecessary to calculate a given set of outputs
- Added `rename_node` method to rename a node, while ensuring that nodes which use it as an input continue to do so
- Added `repoint` method allowing all nodes which use a given node as an input to use an alternative node instead
- Documented `get_inputs` and `i` attribute-style accessor

## [0.2.1] (2017-12-29)

- Added class-style definitions of computations

## [0.2.0] (2017-12-05)

- Added support for multithreading when calculating nodes
- Update to use networkx 2.0
- Added `print_errors` method
- Added `force` parameter to `insert` method to allow no recalculation if value is not updated
- BUGFIX: Fix behavior when calculation node overwritten with value node

## [0.1.3] (2017-07-02)

- Methods set_tag and clear_tag support lists or generators of tags. Method nodes_by_tag can retrieve a list of nodes with a specific tag.
- Remove set_tags and clear_tags.
- Add node computation timing data, accessible through tim attribute-style access or get_timing method.
- compute method can accept a list of nodes to compute.
- Loman now uses pydotplus for visualization. Internally, visualization has two steps: converting a Computation to a networkx visualization DAG, and then converting that to a pydotplus Dot object.
- Added view method - creates and opens a temporary pdf visualization.
- draw and view methods can show timing information with colors='timing' option

## [0.1.2] (2017-04-28)

- Add @node function decorator
- Add ConstantValue (with alias C) to provide constant values to function parameters without creating a placeholder node for that constant
- BUGFIX: Visualizing computations was broken in v0.1.1!

## [0.1.1] (2017-04-25)

- Support for Python 3.4 and 3.5
- Method and attribute-style accessors support lists of nodes
- Added support for node-tagging
- Compute method can optionally throw exceptions, for easier interactive debugging
- `get_inputs` method and `i` attribute-style access to get list of inputs to a node
- `add_node` takes optional inspect parameter to avoid inspection for performance
- `add_node` takes optional group to render graph layout with subgraphs
- `draw_graphviz` renamed to `draw`
- `draw_nx` removed
- `get_df` renamed to `to_df`
- `get_value_dict` renamed to `to_dict`
- BUGFIX: implementation of \_get_calc_nodes used by compute fixed
- BUGFIX: args parameters do not create spurious nodes
- BUGFIX: default function parameters do not cause placeholder node to be created
- BUGFIX: node states correctly updated when calling add_node with value parameter

## [0.1.0] (2017-04-05)

- Added documentation: Introduction, Quickstart and Strategies for Use
- Added docstrings to Computation methods
- Added logging
- Added `v` and `s` fields for attribute-style access to values and states of nodes
- BUGFIX: Detect cycles in `compute_all`

## [0.0.1] (2017-03-24)

- Computation object with `add_node`, `insert`, `compute`, `compute_all`, `state`, `value`, `set_stale` methods
- Computation object can be drawn with `draw_graphviz` method
- Nodes can be updated in place
- Computation handles exceptions in node computation, storing exception and traceback
- Can specify mapping between function parameters and input nodes
- Convenience methods: `add_named_tuple_expansion`, `add_map_node`, `get_df`, `get_value_dict`, `insert_from`, `insert_multi`
- Convenience method
- Computation objects can be serialized
- Computation objects can be shallow-copied with `copy`
- Unit tests
- Runs under Python 2.7, 3.6
