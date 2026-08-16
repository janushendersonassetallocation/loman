# Change Log

## [unreleased]
- FEAT: The widget can now build the graph, not only follow it. `comp.widget(buildable=True)` adds a node form to the toolbar and an Edit/Rename/Delete section to the detail panel, so nodes can be added, redefined, renamed and deleted without leaving the picture. An input node is a name and optionally a scalar; a calculation node is a name, a list of inputs and a Python expression, where each input becomes both a parameter of the compiled function and an edge in the graph. Everything maps onto the API that already existed --- `add_node`, `rename_node`, `delete_node` --- so a node built in the widget is not a second class of node
- FEAT: Names in the node form are read against the block in focus, so a name typed while inside `market` lands inside `market`, and a leading `/` names a node from the top of the computation. That is what lets a node in a block depend on one outside it, and it is why there is no "new block" button: a block is a naming convention, so adding `market/spot` and `market/vol` makes the block appear around them
- FEAT: Building the graph is opt-in through `buildable=True`, off by default and separate from `editable`, because defining a calculation node compiles an expression written in the browser and runs it in the kernel. `namespace=globals()` is what puts the notebook's own imports in scope for it; without a namespace only builtins are. The function's globals stay pointed at the live mapping, so an import made after a node was built is visible to it, and defining a node does not bind its name in the caller's namespace
- FEAT: A node built in the widget can still show its own source. Its text is registered where `inspect` reads from, so `get_source` and the panel's Source section show the expression that was typed rather than reporting that the source is unavailable --- which is what a lambda, and the previous shape of this feature, would have done. What such a node cannot do is round-trip through `save()`: a function compiled from a text box has no importable path, so it saves with `UnserializableFunctionWarning` like any other non-importable function
- FEAT: The node form only offers Edit on a node it could put back the way it found it. A function written in Python has no expression to show, and positional or constant arguments have no field, so those nodes are described rather than offered for editing --- offering would have meant offering to replace them with something else. Rename and Delete are offered on every node, and deleting one that others still depend on reports the PLACEHOLDER Loman leaves behind rather than claiming the node has gone
- FIX: Calc nodes declared on a `@ComputationFactory` class now round-trip, so a saved computation can be reloaded and recomputed. They are methods bound to the definition object, which has no importable path --- after decoration the class's name refers to the factory function --- so the function used to be dropped and the reloaded graph could never update again. It is now stored as the class and method name and rebuilt by constructing a fresh definition object. State set in `__init__` is therefore reconstructed; state mutated on `self` at run time is not, and a definition class that cannot be constructed without arguments still falls back to being stored without a function
- CHANGE: Serialization format version restarts at 1. The shapes written by pre-release versions are still read on a best-effort basis; they were never covered by a published guarantee, so continuing their numbering would have implied one
- FIX: Saving to a path whose parent directory does not exist now says so, naming the directory. Both containers write to a sibling temporary first, so the error previously named a `.tmp` file the caller never asked for
- FIX: Two concurrent saves through the same `ComputationSerializer` corrupted each other. Per-save state lived on the serializer instance, so one save's blob writer replaced another's mid-flight; measured, 11 of 12 threads failed with "Can't write to [a closed zipfile]". The state now lives in a `ContextVar`, and a serializer is safe to build once and share
- FIX: A directory save that failed part way destroyed the previous container. The blobs directory was cleared before anything was written, so an unserializable value left the old manifest pointing at deleted files and nothing loadable on disk — losing a good checkpoint because of an operation that did not succeed. The new container is now built alongside the old one and swapped in only once complete
- FIX: A node that failed and then left ERROR state — because one of its inputs was replaced — could not be saved at all. The error encoding keyed off the node's state rather than the value's type, so the still-present `Error` took the generic path, where an exception has no encoding, and the whole save raised. Introduced by retaining values for out-of-date nodes
- FIX: A node whose function cannot be encoded now emits `UnserializableFunctionWarning` instead of failing silently. The value is still saved; what is lost is the ability to recalculate, and a reloaded graph would otherwise look complete while never updating again. This is how a computation built with `@ComputationFactory` behaves: its calc nodes are bound methods with no importable path, so they cannot be stored — the values round-trip but the nodes cannot recompute
- COMPAT: No public name, method or parameter has been removed or made required. Every parameter added is keyword-only with a default, existing call patterns are unchanged, and `tests/test_api_compat.py` locks that surface so a future removal has to be a decision rather than a side effect. `write_dill_old` stays too, deprecated: it is unsafe to call concurrently, but deleting a public method without a release that says so breaks callers silently
- COMPAT: Files written by earlier releases still load and still recompute, including version 1 and version 2 documents. The reverse does not hold and cannot: a simple graph written by this release still loads under an older one, but a value using a version 3 encoding — any DataFrame or Series, since indexes are now encoded as indexes — raises `UnrecognizedTypeError` there. It fails loudly rather than loading something subtly wrong. Upgrade readers before writers if the two are deployed separately
- COMPAT: Three encodings changed for types that were previously handled wrongly, so the same value now produces different JSON. `np.float64` was silently written as a bare float and now keeps its type; an `IntEnum` member was written as a bare int and now round-trips as the enum; an ERROR node's exception was always rebuilt as a plain `Exception` and now comes back as its own builtin type. Reading old files is unaffected — only what gets written is different
- CHORE: The minimum pandas version is now declared honestly as 2.0. Anyone on pandas 1.x was already broken, since recording datetime resolution needs APIs introduced in 2.0
- FEAT: A node's values can be stored somewhere other than the saved file. Mark it with `add_node(..., store='warehouse')` and supply the implementation at both ends: `comp.save('run.loman', stores={'warehouse': MyStore()})` and `Computation.load(..., stores={...})`. A store is two methods — `write_blob(key, data)` and `read_blob(key)` — and inherits compression, deduplication, checksums and the blob table, so an S3 or database backend needs no loman internals. Override `key_for` to control the key layout
- FEAT: A saved file records a store's *name*, never its configuration, so a bucket, endpoint or credential never reaches the file. The consequence is that a file with external values cannot resolve them alone: the matching store is supplied by whoever loads it, and the error names the missing store and the node when it is not
- FEAT: A node routed to a store that was not supplied is an error at save time rather than a silent fall back to writing the value into the file. Believing data went to a bucket when it is sitting in the archive is worse than a failed save
- FEAT: One save can span several stores — most values in the container, some nodes elsewhere — and the store named on a node is a default that a profile override replaces, so the same computation saves to a bucket in production and to a plain container in a test
- FEAT: Because an external store is independent of the container, the single JSON document can now carry out-of-line values: a manifest you can read in a text editor, describing data held elsewhere
- FIX: `tag:` selectors in a `SerializationProfile` matched nothing. `settings_for` took a set of tags but the only caller never passed any, so a selector like `{'tag:bulky': {...}}` silently did nothing. The node's tags now reach it
- CHORE: Corrected the declared pandas requirement from `>=0.19.2` to `>=2.0`, which is the real floor: recording each datetime value's resolution uses `Timestamp.as_unit` and `DatetimeIndex.unit`, both introduced in pandas 2.0. Below that, a microsecond index would be reread as nanoseconds — wrong timestamps rather than an import error
- COMPAT: pandas 2 and pandas 3 are both supported and now tested. A container written under one loads correctly under the other, because each value's resolution is recorded rather than assumed; `tests/fixtures/pandas2.loman` is a committed pandas 2 file asserted to load exactly, and a separate CI job runs the serialization suite against the minimum supported pandas
- FEAT: Added `Computation.save` and `Computation.load`, writing a `.loman` file: a zip holding a `manifest.json` that describes the graph plus large values stored beside it as binary. The manifest still records every value's shape, dtype, column names and index type, so a saved graph can be inspected without decoding any of the data. A 100k x 10 float DataFrame saves in 8.8 MB and 0.01s, against 22.4 MB and 0.62s for the equivalent JSON document
- FEAT: The profile and the container are independent choices. `profile="readable"` keeps every value inline; `profile="efficient"` (the default) writes large ones out of line. `container` is `"zip"` (the default, a `.loman` file), `"dir"` (the same layout on disk) or `"json"` (one document). Both are inferred from the path, and `load` detects the container from the file itself. Only `efficient` with `"json"` is impossible, and it raises pointing at `container="zip"`. Prefer `container="dir"` when checkpointing repeatedly: updating one value in a zip rewrites the whole archive
- FEAT: Blob compression is named by the caller, never inferred. `compression` on a `SerializationProfile` takes `"none"` or a codec and optional level; the default is `"zstd:1"`. After compressing, whichever of the compressed and raw payloads is smaller is what gets stored, so nothing is ever written larger than it started and no read pays a decompression step for nothing — a byte comparison on data already in hand, with no threshold to tune
- CHORE: `zstandard` is now a required dependency rather than part of the `efficient` extra, which is what makes compressing by default defensible: zstd rejects incompressible data at about 1 GB/s against zlib's 43 MB/s, and compresses real data better and faster besides. The `efficient` extra now contains only `pyarrow`, for parquet frame storage
- CHANGE: An earlier `"auto"` compression mode, which sampled the first 256 KiB of a blob and extrapolated, was removed rather than tuned. On a payload whose character changes part way through — routine in market data — it was wrong in both directions, in one arrangement projecting a 3.6% saving against an actual 36.8% and silently storing the blob raw. It existed only to avoid paying zlib's cost to find out; with zstd there is nothing left for it to save
- FEAT: A `CustomTransformer` can now write bytes out of line by calling `transformer.offer_blob(nbytes=...)` and `transformer.put_blob(...)` inside its existing `to_dict`. No signature changed, so transformers written against earlier releases keep working — they never call `offer_blob`, so they always inline, exactly as before
- CHORE: Added the `loman[efficient]` extra (`pyarrow`, `zstandard`) for parquet frame storage and zstd compression. Neither is required: without them values are stored as `.npy` and compressed with `zlib`, both already available. A frame pyarrow cannot represent falls back to the default encoding rather than failing the save
- FIX: A DataFrame or Series indexed by dates could not be serialized at all — `pandas.Timestamp` had no transformer, so any `DatetimeIndex` raised `SerializationError`. Also now supported: `datetime`, `date`, `time`, `timedelta`, `pandas.Timedelta`, `NaT`, numpy scalars, `set`, `frozenset`, `bytes`, `bytearray` and `Decimal`
- FIX: A dict with non-string keys silently came back with string keys — `{1: 'a'}` reloaded as `{'1': 'a'}`. Such a dict is now written as a list of encoded key/value pairs. A dict whose keys are all plain strings is still written as a JSON object, so the common case stays readable
- FIX: A `MultiIndex` came back as a flat `Index` of tuples, and a DataFrame's `columns` were never passed through the transformer. Indexes are now encoded as indexes, which also means the default `RangeIndex` of a large frame is four numbers rather than one per row
- FIX: Non-finite floats were written as bare `NaN` / `Infinity` tokens, which Python reads back and stricter JSON parsers reject. They are now tagged, and `allow_nan=False` is set, so an invalid document is structurally impossible rather than merely unlikely
- FIX: `STALE` and `COMPUTABLE` nodes were saved without the values they still held, discarding exactly the intermediates that make a saved graph worth inspecting. A node is now saved whenever it holds a value. A node whose value is legitimately `None` is also distinguished from one with no value
- FIX: A node's `group`, `style`, `executor` and `converter` were reset to `None` on load, and metadata and timing were never written at all, so a reloaded graph rendered and ran differently from the one saved. All now round-trip
- FIX: An ERROR node's exception was always rebuilt as a bare `Exception`, so `except ValueError` no longer matched after a round-trip. Builtin exception types are now reconstructed as themselves. Others become `loman.exception.DeserializedError`, carrying the original type name and module: rebuilding an arbitrary exception would mean importing whatever module the file names
- FEAT: Added `allow_code=False` to `Computation.load` and `read_json`. Loading normally restores node functions, which imports the modules the file names or unpickles a dill blob out of it — both run code the file chose. With `allow_code=False` callables are skipped and values, structure, states and tags still load. This is a mitigation, not a security boundary; the format is not safe against a hostile file and never was
- CHANGE: Serialization format version is now 3. Version 1 and 2 files still load and still recompute
- FIX: `Transformer.register_*` raised `AssertionError` on a duplicate registration, which vanished under `python -O` and turned a duplicate into a silent overwrite. It now raises `DuplicateRegistrationError`, a `ValueError` subclass
- FEAT: Added `SimpleTransformer`, building a transformer from a `to_dict`/`from_dict` pair, and `ComputationSerializer.register`. The custom-type example in the documentation described both and neither existed, so that example raised `TypeError` as written
- CHANGE: Removed the deprecated `write_dill_old`, which deleted `__getstate__` and `__setstate__` from the class object for the duration of a write, making it unsafe to call from more than one thread
- FIX: `add_fan_in` accepted a result that one of its own sources already depended on, producing a cyclic graph that only surfaced later during planning or compute. It now rejects it before mutating, as `add_fan_out` always did
- FIX: Asking for `self` binding with no definition object now explains the contradiction, instead of surfacing a bare `TypeError: instance must not be None` from inside the binding call
- CHANGE: `FanOut` rejects a target the block template never mentions, since that is usually a typo that would add a dead node to every block; pass `create=True` to feed such a node deliberately. The low-level `add_fan_out` stays permissive, having no template to check against
- FEAT: `InputValue` takes the same `create` flag as `FanOut`, and `IdNode` takes it with a default of `True`, since creating the node is its purpose; pass `create=False` there to have a misspelled name rejected at definition time rather than surfacing later as an uninitialized input
- FEAT: Added `Positional`, wrapping an aggregator that takes its values positionally so a `combine` does not need a lambda at every call site
- FIX: A fan-out targeting a node an earlier feature planned raised a bare `KeyError` from inside the builder; it now reports the duplicate write
- FEAT: Allow `Computation.compute` to compute one or more blocks
- FEAT: Added utilities for repeated blocks and keyed fan-in/fan-out computations
- FEAT: Added `repeated_blocks` to declare repeated blocks within a `@ComputationFactory` class
- CHANGE: Repeated blocks are now described by an ordered list of features (`FanOut`, `FanIn`, `IdNode`, `InputValue`), replacing the separate `fan_out` and `fan_in` arguments
- CHANGE: Features describe nodes rather than creating them, so custom wiring patterns can be added by implementing `BlockFeature.plan` without giving up validate-before-mutate
- FEAT: Fan-out sources may be a function of the key, so each repeated block can read from a different node
- FEAT: `IdNode` gives each repeated block a node holding its own key
- CHANGE: Fan-out transforms are now the generated node's own function rather than a wrapped constant
- FIX: A callable passed where a node name is expected raises `TypeError` instead of creating a node keyed by the function
- FIX: Constant node arguments (`C(...)`) are now serialized. Previously they were silently dropped, so a reloaded graph looked up to date but raised `TypeError` from the missing argument as soon as the node was recalculated. Serialization format version is now 2; version 1 files still load. A constant that cannot be encoded now raises `SerializationError` naming the node instead of being dropped. Pass `ComputationSerializer(on_unserializable_constant="drop")` to restore the previous lenient behaviour while an existing codebase is fixed: the constant is omitted and an `UnserializableConstantWarning` is emitted, rather than the save failing. A version 1 file is not repaired by loading it under this release: its constants were never written, so a graph saved before this change must be saved again to gain them. Conversely, a version 2 file loads under earlier releases without error — they ignore the new fields, giving the same dropped constants as before — so the two versions can be read either way round while a codebase moves over
- FEAT: Added non-mutating graph validation and execution planning APIs with DataFrame diagnostics
- FEAT: Added `Computation.subscribe` and `ComputationEvent` for observing batched changes to a
  computation. A computation with no subscribers pays no measurable cost: state propagation
  merges keys set-to-set rather than materialising them, so hashing does not grow. A callback
  with an object behind it is held weakly, whether the method is written in Python or in C,
  so `comp.subscribe(events.append)` no longer retains `events` for the life of the
  computation; owners that support no weak reference, such as `list`, still fall back to a
  strong one
- FEAT: Added the `loman[ui]` extra, providing `Computation.widget()`: an interactive notebook graph
  that follows its computation, with node inspection, drill-down into blocks, scalar input
  editing and compute controls
- FEAT: `GraphView` now exposes `original_nodes`, `composite_nodes` and `node_index_map`, so callers
  can map rendered nodes back to computation nodes
- FEAT: The widget's detail panel renders DataFrames, Series, arrays and nested data, with editable
  cells on tabular input nodes, a tail window onto long frames, and a "Show full" button that
  hands the node to the notebook to render with its own tools
- FEAT: The widget matches the background of the page it is embedded in, and takes its light or dark
  theme from that rather than from the operating system setting. Where the host publishes a
  shadcn-style palette, as marimo does, the widget wears that palette directly
- FEAT: Clicking a block in the widget opens it where it stands; alt-clicking isolates it, so the
  view shows only that block's top layer and the breadcrumb leads back out
- FEAT: The widget's graph has no background of its own, so it sits on the host's and follows its
  theme: Graphviz is told `bgcolor="transparent"` and its ink is retinted, while node fills
  keep the state colours and node labels are inked against the fill they land on
- FEAT: The widget's detail panel opens on a click and closes on Escape, so the graph has the
  full width whenever nothing is being inspected
- FEAT: The widget keeps its recent layouts, so navigating back out of a block does not re-run
  Graphviz over a picture it has already drawn
- FIX: Clicking an open block's title did not close it. Pressing anywhere in the graph
  entered the panning state immediately, which made the canvas inert to hit-testing before
  the button came back up; panning now waits for the pointer to actually move. The title bar
  is also the close target now, rather than the width of the word
- FIX: The widget's "Show full" resolved the node by its string label, so it returned the
  wrong value where a node named `1` and a node named `"1"` both exist. It now carries the
  node key, and `full_view_name` reports the name with its original type, as
  `selected_name` does
- FEAT: Added `fit_on_render` to scale the graph to the pane on every render

## [0.6.0] (2026-04-26)
- FEAT: Added JSON serialization as the replacement for the dill-based format. `Computation.write_json`
  and `Computation.read_json` are backed by a `ComputationSerializer` built on the Transformer
  framework, with transformers for enums, importable callables, NodeKeys, numpy arrays, and pandas
  DataFrames and Series. The default profile stores functions by reference, so a saved graph no
  longer embeds a dill blob and is portable across Python versions; lambdas and closures raise
  `SerializationError` instead. Pass the dill profile to keep serializing them as before, with the
  same portability caveats. `write_dill` and `read_dill` still work but are deprecated and emit a
  `DeprecationWarning` — see "Migrating from write_dill to write_json" for the move

## [0.5.5] (2026-04-12)
- CHORE: No changes to the library; released for tooling and template synchronisation only

## [0.5.4] (2026-04-10)
- FEAT: Added type hints on ComputationFactory
- FIX: `compute_and_get_value` sets error state on exception
- FIX: `ComputationFactory` preserves the decorated class's metadata rather than replacing it
- CHORE: Improved type safety across the codebase

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
- FIX: Linking a node to itself is a no-op
- FIX: Inserting to a placeholder node raises a specific exception
- FIX: Composite blocks retain sub-block on collapsing

## [0.5.0] (2025-04-10)

- Add support for blocks (Computation.add_block)
- Add support for links (Computation.link)
- Nodes keyed using NodeKey with paths to support nested blocks
- Visualization modified to support grouping elements in same block
- FIX: Fix calc nodes with no parameters
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
- FIX: Fix equality testing on Computation.insert
- Use DataFrame.equals and Series.equals to test equality in Computation.insert
- FIX: Fix handling of groups in rendering functions

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
- FIX: Fix behavior when calculation node overwritten with value node

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
- FIX: Visualizing computations was broken in v0.1.1!

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
- FIX: Implementation of \_get_calc_nodes used by compute fixed
- FIX: args parameters do not create spurious nodes
- FIX: Default function parameters do not cause placeholder node to be created
- FIX: Node states correctly updated when calling add_node with value parameter

## [0.1.0] (2017-04-05)

- Added documentation: Introduction, Quickstart and Strategies for Use
- Added docstrings to Computation methods
- Added logging
- Added `v` and `s` fields for attribute-style access to values and states of nodes
- FIX: Detect cycles in `compute_all`

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
