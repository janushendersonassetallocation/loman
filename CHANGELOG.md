# Change Log

## [unreleased]
- CHORE: Broke the import cycle between `loman.computeengine` and `loman.util`. The utility layer reached back into the engine from inside seven function bodies --- `C`, `ConstantValue`, `identity_function` and `_bind_self` --- because importing them at module level would have closed a loop, and a second cycle ran through `loman.nodekey`, which imported `as_iterable` from `util` while `util` deferred its own import of `to_nodekey`. That shared vocabulary now lives in two leaf modules, `loman.nodedefs` and `loman.iterables`, so every dependency `util` has is visible at the top of the file and a symbol the engine happens to define is no longer trapped there. No name in `loman`'s public surface moved; the internal ones that did are `loman.util.as_iterable` and `loman.util.apply_n`, now in `loman.iterables`, and `loman.computeengine._bind_self`, now in `loman.nodedefs`
- FEAT: The widget can now build the graph, not only follow it. `comp.widget(buildable=True)` adds a node form to the toolbar and an Edit/Rename/Delete section to the detail panel, so nodes can be added, redefined, renamed and deleted without leaving the picture. An input node is a name and optionally a scalar; a calculation node is a name, a list of inputs and a Python expression, where each input becomes both a parameter of the compiled function and an edge in the graph. Everything maps onto the API that already existed --- `add_node`, `rename_node`, `delete_node` --- so a node built in the widget is not a second class of node
- FEAT: Names in the node form are read against the block in focus, so a name typed while inside `market` lands inside `market`, and a leading `/` names a node from the top of the computation. That is what lets a node in a block depend on one outside it, and it is why there is no "new block" button: a block is a naming convention, so adding `market/spot` and `market/vol` makes the block appear around them
- FEAT: Building the graph is opt-in through `buildable=True`, off by default and separate from `editable`, because defining a calculation node compiles an expression written in the browser and runs it in the kernel. `namespace=globals()` is what puts the notebook's own imports in scope for it; without a namespace only builtins are. The function's globals stay pointed at the live mapping, so an import made after a node was built is visible to it, and defining a node does not bind its name in the caller's namespace
- FEAT: A node built in the widget can still show its own source. Its text is registered where `inspect` reads from, so `get_source` and the panel's Source section show the expression that was typed rather than reporting that the source is unavailable --- which is what a lambda, and the previous shape of this feature, would have done. What such a node cannot do is round-trip through `save()`: a function compiled from a text box has no importable path, so it saves with `UnserializableFunctionWarning` like any other non-importable function
- FEAT: The node form only offers Edit on a node it could put back the way it found it. A function written in Python has no expression to show, and positional or constant arguments have no field, so those nodes are described rather than offered for editing --- offering would have meant offering to replace them with something else. Rename and Delete are offered on every node, and deleting one that others still depend on reports the PLACEHOLDER Loman leaves behind rather than claiming the node has gone
- FIX: Calc nodes declared on a `@ComputationFactory` class now round-trip, so a saved computation can be reloaded and recomputed. They are methods bound to the definition object, which has no importable path --- after decoration the class's name refers to the factory function --- so the function used to be dropped and the reloaded graph could never update again. It is now stored as the class and method name and rebuilt by constructing a fresh definition object. State set in `__init__` is therefore reconstructed; state mutated on `self` at run time is not, and a definition class that cannot be constructed without arguments still falls back to being stored without a function
- CHANGE: Serialization format version restarts at 1. The shapes written by pre-release versions are still read on a best-effort basis; they were never covered by a published guarantee, so continuing their numbering would have implied one

## [0.7.0] (2026-08-16)

### Serialization

- FEAT: Added `Computation.save` and `Computation.load`, writing a `.loman` container: a zip holding a `manifest.json` that describes the graph, with large values stored beside it as binary. The manifest records every value's shape, dtype, column names and index type, so a saved graph can be inspected without decoding any of the data. A 100k x 10 float DataFrame saves in 8.8 MB and 0.01s, against 22.4 MB and 0.62s for the equivalent JSON document
- FEAT: Profile and container are independent choices. `profile="readable"` keeps every value inline, `profile="efficient"` (the default) writes large ones out of line; `container` is `"zip"` (a `.loman` file), `"dir"` (the same layout on disk) or `"json"` (a single document). Both are inferred from the path, and `load` detects the container from the file itself. Only `efficient` with `"json"` is impossible, and it raises pointing at `container="zip"`. Prefer `container="dir"` when checkpointing repeatedly, since updating one value in a zip rewrites the whole archive
- FEAT: A node's values can be stored outside the saved file. `add_node(..., store='warehouse')` routes them to a store supplied at both ends — `save(..., stores={'warehouse': MyStore()})` and `load(..., stores={...})` — where a store is two methods, `write_blob` and `read_blob`, and inherits compression, deduplication, checksums and the blob table, so an S3 or database backend needs no loman internals. One save can span several stores, and a profile override replaces the store named on a node, so the same computation writes to a bucket in production and to a plain container in a test
- FEAT: A saved file records a store's *name*, never its configuration, so a bucket, endpoint or credential never reaches the file: the matching store is supplied by whoever loads it, and the error names the missing store and node when it is not. A node routed to a store that was not supplied fails the save rather than falling back to writing the value inline — believing data went to a bucket when it is sitting in the archive is worse than a failed save. Because stores are independent of the container, the single JSON document can now carry out-of-line values: a manifest readable in a text editor, describing data held elsewhere
- FEAT: Blob compression is named by the caller, never inferred — `compression` on a `SerializationProfile` takes `"none"` or a codec and optional level, defaulting to `"zstd:1"` — and whichever of the compressed and raw payloads is smaller is what gets stored, so nothing is written larger than it started. An earlier `"auto"` mode that sampled the first 256 KiB and extrapolated was removed rather than tuned: on a payload whose character changes part way through it was wrong in both directions
- FEAT: A `CustomTransformer` can write bytes out of line by calling `transformer.offer_blob(nbytes=...)` and `transformer.put_blob(...)` inside its existing `to_dict`. No signature changed, so transformers written against earlier releases keep working — they never call `offer_blob`, so they always inline, exactly as before
- FEAT: Added `SimpleTransformer`, building a transformer from a `to_dict`/`from_dict` pair, and `ComputationSerializer.register`. The custom-type example in the documentation described both and neither existed, so that example raised `TypeError` as written
- FEAT: Added `allow_code=False` to `Computation.load` and `read_json`. Loading normally restores node functions, which imports the modules the file names or unpickles a dill blob out of it — both run code the file chose. With `allow_code=False` callables are skipped and values, structure, states and tags still load. This is a mitigation, not a security boundary; the format is not safe against a hostile file and never was
- FIX: The value model is now lossless where it was lossy. A `DatetimeIndex` could not be serialized at all, and `datetime`, `date`, `time`, `timedelta`, `Timedelta`, `NaT`, numpy scalars, `set`, `frozenset`, `bytes`, `bytearray` and `Decimal` are now supported too. A dict with non-string keys came back with string keys; a `MultiIndex` came back as a flat `Index` of tuples and a DataFrame's `columns` never went through the transformer, so indexes are now encoded as indexes — which also makes a large frame's default `RangeIndex` four numbers rather than one per row; and non-finite floats, written as bare `NaN` / `Infinity` tokens that stricter parsers reject, are now tagged, with `allow_nan=False` set so an invalid document is structurally impossible
- FIX: A saved graph now reloads as the graph that was saved. `STALE` and `COMPUTABLE` nodes were saved without the values they still held, discarding exactly the intermediates that make a saved graph worth inspecting, and a node whose value is legitimately `None` was indistinguishable from one with no value; `group`, `style`, `executor` and `converter` were reset to `None`, and metadata and timing were never written at all; constant node arguments (`C(...)`) were silently dropped, so a reloaded graph looked up to date but raised `TypeError` as soon as the node recalculated — an unencodable constant now raises `SerializationError` naming the node, with `ComputationSerializer(on_unserializable_constant="drop")` restoring the lenient behaviour behind a warning while a codebase is fixed
- FIX: An ERROR node's exception was always rebuilt as a bare `Exception`, so `except ValueError` no longer matched after a round-trip. Builtin exception types are now reconstructed as themselves; others become `loman.exception.DeserializedError`, carrying the original type name and module, since rebuilding an arbitrary exception would mean importing whatever module the file names. A node that failed and then left ERROR state, because one of its inputs was replaced, also could not be saved at all: the error encoding keyed off the node's state rather than the value's type, and the whole save raised
- FIX: Calc nodes declared on a `@ComputationFactory` class now round-trip, so a saved computation can be reloaded and recomputed. They are methods bound to the definition object, which has no importable path — after decoration the class's name refers to the factory function — so the function used to be dropped and the reloaded graph could never update again. It is now stored as the class and method name and rebuilt by constructing a fresh definition object: state set in `__init__` is reconstructed, state mutated on `self` at run time is not, and a definition class that cannot be constructed without arguments still falls back to being stored without a function
- FIX: A node whose function cannot be encoded now emits `UnserializableFunctionWarning` instead of failing silently. The value is still saved; what is lost is the ability to recalculate, and a reloaded graph would otherwise look complete while never updating again
- FIX: Saving is now safe to interrupt and safe to share. A directory save that failed part way destroyed the previous container, because the blobs directory was cleared before anything was written — the new container is now built alongside the old one and swapped in only once complete. Two concurrent saves through the same `ComputationSerializer` corrupted each other, since per-save state lived on the instance; measured, 11 of 12 threads failed with "Can't write to [a closed zipfile]". That state now lives in a `ContextVar`, and a serializer is safe to build once and share
- FIX: Saving to a path whose parent directory does not exist now says so, naming the directory. Both containers write to a sibling temporary first, so the error previously named a `.tmp` file the caller never asked for
- FIX: `tag:` selectors in a `SerializationProfile` matched nothing, because `settings_for` took a set of tags and the only caller never passed any. The node's tags now reach it
- FIX: `Transformer.register_*` raised `AssertionError` on a duplicate registration, which vanished under `python -O` and turned a duplicate into a silent overwrite. It now raises `DuplicateRegistrationError`, a `ValueError` subclass
- FIX: `write_json` and `read_json` open files as UTF-8 rather than the platform locale encoding, as RFC 8259 requires. Latent today, since `ensure_ascii=True` keeps written files pure ASCII, but real as soon as that is turned off
- CHANGE: The serialization format version restarts at 1. The shapes written by pre-release versions are still read on a best-effort basis; they were never covered by a published guarantee, so continuing their numbering would have implied one
- COMPAT: Files written by earlier releases still load and still recompute. Three encodings changed for types that were previously handled wrongly, so the same value now produces different JSON: `np.float64` was silently written as a bare float and now keeps its type, an `IntEnum` member was written as a bare int and now round-trips as the enum, and an ERROR node's exception is now its own builtin type. Reading old files is unaffected — only what gets written differs — but a file written by this release and read by an older one can raise `UnrecognizedTypeError` on any DataFrame or Series, since indexes are now encoded as indexes. It fails loudly rather than loading something subtly wrong: upgrade readers before writers if the two are deployed separately
- COMPAT: No public name, method or parameter has been removed or made required. Every parameter added is keyword-only with a default, existing call patterns are unchanged, and `tests/test_api_compat.py` locks that surface so a future removal has to be a decision rather than a side effect. `write_dill_old` stays too, deprecated: it is unsafe to call concurrently, but deleting a public method without a release that says so breaks callers silently
- COMPAT: pandas 2 and pandas 3 are both supported and now tested. A container written under one loads correctly under the other, because each value's resolution is recorded rather than assumed; `tests/fixtures/pandas2.loman` is a committed pandas 2 file asserted to load exactly, and a separate CI job runs the serialization suite against the minimum supported pandas
- CHORE: The declared pandas floor is corrected from `>=0.19.2` to `>=2.0`, which is the real one: recording each datetime value's resolution uses `Timestamp.as_unit` and `DatetimeIndex.unit`, both introduced in pandas 2.0. Anyone on pandas 1.x was already broken — below that floor a microsecond index would be reread as nanoseconds, wrong timestamps rather than an import error
- CHORE: `zstandard` is now a required dependency rather than part of the `efficient` extra, which is what makes compressing by default defensible: zstd rejects incompressible data at about 1 GB/s against zlib's 43 MB/s, and compresses real data better and faster besides. The `efficient` extra now contains only `pyarrow`, for parquet frame storage; without it values are stored as `.npy` and compressed with `zlib`, and a frame pyarrow cannot represent falls back to the default encoding rather than failing the save

### Notebook widget

- FEAT: Added the `loman[ui]` extra, providing `Computation.widget()`: an interactive notebook graph that follows its computation, with node inspection, drill-down into blocks, scalar input editing and compute controls. The detail panel renders DataFrames, Series, arrays and nested data, with editable cells on tabular input nodes, a tail window onto long frames, and a "Show full" button that hands the node to the notebook to render with its own tools
- FEAT: The widget can now build the graph, not only follow it. `comp.widget(buildable=True)` adds a node form to the toolbar and an Edit/Rename/Delete section to the detail panel, so nodes can be added, redefined, renamed and deleted without leaving the picture. An input node is a name and optionally a scalar; a calculation node is a name, a list of inputs and a Python expression, where each input becomes both a parameter of the compiled function and an edge in the graph. Everything maps onto the API that already existed — `add_node`, `rename_node`, `delete_node` — so a node built in the widget is not a second class of node
- FEAT: Names in the node form are read against the block in focus, so a name typed while inside `market` lands inside `market`, and a leading `/` names a node from the top of the computation. That is what lets a node in a block depend on one outside it, and it is why there is no "new block" button: a block is a naming convention, so adding `market/spot` and `market/vol` makes the block appear around them
- FEAT: Building the graph is opt-in through `buildable=True`, off by default and separate from `editable`, because defining a calculation node compiles an expression written in the browser and runs it in the kernel. `namespace=globals()` puts the notebook's own imports in scope for it; without a namespace only builtins are. The function's globals stay pointed at the live mapping, so an import made after a node was built is visible to it, and defining a node does not bind its name in the caller's namespace
- FEAT: A node built in the widget can still show its own source: its text is registered where `inspect` reads from, so `get_source` and the panel's Source section show the expression that was typed. What such a node cannot do is round-trip through `save()` — a function compiled from a text box has no importable path, so it saves with `UnserializableFunctionWarning` like any other non-importable function. The form likewise only offers Edit on a node it could put back the way it found it: a function written in Python has no expression to show, and positional or constant arguments have no field, so those nodes are described rather than offered for editing. Rename and Delete are offered on every node, and deleting one that others still depend on reports the PLACEHOLDER Loman leaves behind rather than claiming the node has gone
- FEAT: The widget takes its appearance from the page it is embedded in rather than from the operating system: it matches the host background, follows its light or dark theme, and where the host publishes a shadcn-style palette, as marimo does, wears that palette directly. The graph has no background of its own — Graphviz is told `bgcolor="transparent"` and its ink is retinted, while node fills keep the state colours and labels are inked against the fill they land on
- FEAT: Clicking a block opens it where it stands and alt-clicking isolates it, with a breadcrumb leading back out; the detail panel opens on a click and closes on Escape, so the graph has the full width whenever nothing is being inspected; and recent layouts are kept, so navigating back out of a block does not re-run Graphviz over a picture it has already drawn. Added `fit_on_render` to scale the graph to the pane on every render
- FEAT: `GraphView` now exposes `original_nodes`, `composite_nodes` and `node_index_map`, so callers can map rendered nodes back to computation nodes
- FIX: Clicking an open block's title did not close it, because pressing anywhere in the graph entered the panning state immediately and made the canvas inert to hit-testing before the button came back up; panning now waits for the pointer to actually move, and the title bar is the close target rather than the width of the word
- FIX: The widget's "Show full" resolved the node by its string label, so it returned the wrong value where a node named `1` and a node named `"1"` both exist. It now carries the node key, and `full_view_name` reports the name with its original type, as `selected_name` does

### Repeated blocks

- FEAT: Added utilities for repeated blocks and keyed fan-in/fan-out computations, and `repeated_blocks` to declare them within a `@ComputationFactory` class
- CHANGE: Repeated blocks are described by an ordered list of features — `FanOut`, `FanIn`, `IdNode`, `InputValue` — replacing the separate `fan_out` and `fan_in` arguments. Features describe nodes rather than creating them, so custom wiring patterns can be added by implementing `BlockFeature.plan` without giving up validate-before-mutate
- FEAT: Fan-out sources may be a function of the key, so each repeated block can read from a different node, and `IdNode` gives each block a node holding its own key. Fan-out transforms are now the generated node's own function rather than a wrapped constant. Added `Positional`, wrapping an aggregator that takes its values positionally so a `combine` does not need a lambda at every call site
- CHANGE: `FanOut` rejects a target the block template never mentions, since that is usually a typo that would add a dead node to every block; pass `create=True` to feed such a node deliberately. `InputValue` takes the same flag, and `IdNode` takes it defaulting to `True` since creating the node is its purpose. The low-level `add_fan_out` stays permissive, having no template to check against
- FIX: `add_fan_in` accepted a result that one of its own sources already depended on, producing a cyclic graph that only surfaced later during planning or compute. It now rejects it before mutating, as `add_fan_out` always did
- FIX: A fan-out targeting a node an earlier feature planned raised a bare `KeyError` from inside the builder; it now reports the duplicate write. A callable passed where a node name is expected raises `TypeError` instead of creating a node keyed by the function, and asking for `self` binding with no definition object explains the contradiction rather than surfacing a bare `TypeError` from inside the binding call

### Computation engine

- FEAT: Added `Computation.subscribe` and `ComputationEvent` for observing batched changes to a computation. A computation with no subscribers pays no measurable cost: state propagation merges keys set-to-set rather than materialising them, so hashing does not grow. A callback with an object behind it is held weakly, whether the method is written in Python or in C, so `comp.subscribe(events.append)` no longer retains `events` for the life of the computation; owners that support no weak reference, such as `list`, still fall back to a strong one
- FEAT: Added non-mutating graph validation and execution planning APIs with DataFrame diagnostics
- FEAT: `Computation.compute` can compute one or more blocks

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
