# Change Log

## [unreleased]
- `FanOut` rejects a target the block template never mentions, since that is usually a typo that would add a dead node to every block; pass `create=True` to feed such a node deliberately. The low-level `add_fan_out` stays permissive, having no template to check against
- Added `Positional`, wrapping an aggregator that takes its values positionally so a `combine` does not need a lambda at every call site
- Documented which repeated-block names are relative to `base_path`: a fan-in `result` is a verbatim outer node, while an `InputValue` is placed under `base_path`
- BUGFIX: a fan-out targeting a node an earlier feature planned raised a bare `KeyError` from inside the builder; it now reports the duplicate write
- CI: the Graphviz install now retries transient package-feed failures, falls back between
  Chocolatey and winget, and fails loudly when `dot` is still missing rather than letting
  the test run report it as a hundred unrelated failures
- Allow `Computation.compute` to compute one or more blocks
- Added type hints on ComputationFactory
- BUGFIX: `compute_and_get_value` sets error state on exception
- Added utilities for repeated blocks and keyed fan-in/fan-out computations
- Added `repeated_blocks` to declare repeated blocks within a `@ComputationFactory` class
- Repeated blocks are now described by an ordered list of features (`FanOut`, `FanIn`, `IdNode`, `InputValue`), replacing the separate `fan_out` and `fan_in` arguments
- Features describe nodes rather than creating them, so custom wiring patterns can be added by implementing `BlockFeature.plan` without giving up validate-before-mutate
- Fan-out sources may be a function of the key, so each repeated block can read from a different node
- `IdNode` gives each repeated block a node holding its own key
- Fan-out transforms are now the generated node's own function rather than a wrapped constant
- BUGFIX: a callable passed where a node name is expected raises `TypeError` instead of creating a node keyed by the function
- BUGFIX: constant node arguments (`C(...)`) are now serialized. Previously they were silently dropped, so a reloaded graph looked up to date but raised `TypeError` from the missing argument as soon as the node was recalculated. Serialization format version is now 2; version 1 files still load. A constant that cannot be encoded now raises `SerializationError` naming the node instead of being dropped. Pass `ComputationSerializer(on_unserializable_constant="drop")` to restore the previous lenient behaviour while an existing codebase is fixed: the constant is omitted and an `UnserializableConstantWarning` is emitted, rather than the save failing
- Added a Marimo example of a large repeated instrument-block computation
- Added non-mutating graph validation and execution planning APIs with DataFrame diagnostics
- Added `Computation.subscribe` and `ComputationEvent` for observing batched changes to a
  computation. A computation with no subscribers pays no measurable cost: state propagation
  merges keys set-to-set rather than materialising them, so hashing does not grow. A callback
  with an object behind it is held weakly, whether the method is written in Python or in C,
  so `comp.subscribe(events.append)` no longer retains `events` for the life of the
  computation; owners that support no weak reference, such as `list`, still fall back to a
  strong one
- Added the `loman[ui]` extra, providing `Computation.widget()`: an interactive notebook graph
  that follows its computation, with node inspection, drill-down into blocks, scalar input
  editing and compute controls
- `GraphView` now exposes `original_nodes`, `composite_nodes` and `node_index_map`, so callers
  can map rendered nodes back to computation nodes
- The widget's detail panel renders DataFrames, Series, arrays and nested data, with editable
  cells on tabular input nodes, a tail window onto long frames, and a "Show full" button that
  hands the node to the notebook to render with its own tools
- The widget matches the background of the page it is embedded in, and takes its light or dark
  theme from that rather than from the operating system setting. Where the host publishes a
  shadcn-style palette, as marimo does, the widget wears that palette directly
- Clicking a block in the widget opens it where it stands; alt-clicking isolates it, so the
  view shows only that block's top layer and the breadcrumb leads back out
- The widget's graph has no background of its own, so it sits on the host's and follows its
  theme: Graphviz is told `bgcolor="transparent"` and its ink is retinted, while node fills
  keep the state colours and node labels are inked against the fill they land on
- The widget's detail panel opens on a click and closes on Escape, so the graph has the
  full width whenever nothing is being inspected
- The widget keeps its recent layouts, so navigating back out of a block does not re-run
  Graphviz over a picture it has already drawn
- BUGFIX: clicking an open block's title did not close it. Pressing anywhere in the graph
  entered the panning state immediately, which made the canvas inert to hit-testing before
  the button came back up; panning now waits for the pointer to actually move. The title bar
  is also the close target now, rather than the width of the word
- BUGFIX: the widget's "Show full" resolved the node by its string label, so it returned the
  wrong value where a node named `1` and a node named `"1"` both exist. It now carries the
  node key, and `full_view_name` reports the name with its original type, as
  `selected_name` does
- Added `fit_on_render` to scale the graph to the pane on every render

## [0.5.3] (2025-06-20)

- ComputationFactories can have blocks added directly to them
- Add support for viewing node function source code
- Added support for node, block and computation metadata
- Added a custom json serializer for future use
- Various bug fixes

## [0.5.2] (2025-05-28)
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
