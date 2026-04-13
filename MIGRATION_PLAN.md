# Dill → JSON Serialization Migration

## Context

Loman uses `dill` for serialization (`computeengine.py`). This was necessary because nodes
store Python callables (functions, lambdas, closures), and standard pickle cannot handle
these. Dill extends pickle to serialize arbitrary callables.

However, dill carries significant problems:

| Problem | Detail |
|---------|--------|
| **Security** | `dill.load()` can execute arbitrary code on deserialization |
| **Portability** | Binary format is sensitive to Python version and library versions |
| **Transparency** | Binary format is not human-readable or auditable |
| **Dependency** | Adds an external dependency that exists solely for this one purpose |

A `Transformer` framework already exists in `src/loman/serialization/` that converts objects
to JSON-serializable dicts without executing arbitrary code. It supports primitives, tuples,
lists, dicts, dataclasses, attrs classes, numpy arrays, and custom types via plugin. The
migration extends this framework to cover the remaining types needed for full graph
serialization.

---

## Intended Use Cases for Serialization

From the loman documentation, the intended uses are:

1. **Post-mortem inspection** — A scheduled batch process serializes its full state (inputs,
   intermediates, results) so that when failures occur operators can load and inspect exactly
   what happened.

2. **Error recovery** — After loading a serialized computation, operators can insert corrected
   data and recalculate only the affected nodes rather than re-running the full pipeline.

3. **Broadcasting calibrated state** — A central process (e.g. interest rate curve
   calibration) serializes its computed state and distributes it to consumer processes.

4. **Interactive experimentation** — A serialized computation is loaded in a notebook as a
   starting point for exploration without re-running expensive upstream calculations.

> **Note:** The serialization format is not stabilized and should *not* be relied on for
> long-term storage. It is intended for operational inspection and short-term handoff.

---

## Design Decisions

### Function handling

Only **importable** functions can be serialized — functions that can be retrieved at
deserialization time via `importlib.import_module(module).__dict__[qualname]`. Lambdas,
closures, and interactively-defined functions raise `SerializationError`. This is a deliberate
constraint: if a node's function cannot be referenced by name, the caller must either:

- Move the function to a named module-level definition, or
- Set `serialize=False` on that node (its value will still be serialized if it is UPTODATE)

### ERROR state nodes

ERROR nodes are serialized with the exception class name, message, and formatted traceback
stored as strings. This directly supports the primary use case of post-mortem inspection — you
want to see *what* failed when you reload the graph. The exception cannot be re-raised (the
live object is not preserved), but all diagnostic information is retained.

### PINNED state

PINNED nodes are serialized and restored as PINNED. The value is stored identically to UPTODATE
nodes. Preserving PINNED is important for correctness: a PINNED node blocks staleness
propagation from its dependencies, and this behavior must be the same after a roundtrip.

### Edge encoding

Each edge in the DAG carries a `param` attribute: a tuple of `(_ParameterType, name_or_index)`.
This encodes whether the upstream node feeds a positional argument (ARG, int index) or a keyword
argument (KWD, string name). The serialized edge format preserves both:

```json
{"source": "a", "target": "b", "param_type": "kwd", "param_name": "x"}
{"source": "a", "target": "b", "param_type": "arg", "param_index": 0}
```

### Node names

All node identifiers are `NodeKey` objects internally. Single-part keys (e.g.
`NodeKey(("a",))`) are serialized as plain strings (`"a"`). Hierarchical keys (e.g.
`NodeKey(("foo", "b"))`) are serialized as path strings (`"foo/b"`), using the same
format as `str(NodeKey(...))`.

### Nested Computation as node value

Deferred. Serializing a `Computation` stored as the value of a node in another
`Computation` requires recursive serialization and a `ComputationTransformer`. This
is out of scope for this migration.

### Attributes not serialized

The following are not serialized, matching the current dill behaviour:

- `_metadata` — per-node metadata dict
- `executor_map` / `default_executor` — executor configuration
- `converter` — per-node value converter functions

### Deprecation strategy

Two-step: deprecation warnings in one release, full removal in the next. This gives users
one release cycle to migrate.

---

## JSON Output Format

```json
{
  "version": "1.0",
  "nodes": {
    "a": {
      "state": "UPTODATE",
      "value": {"type": "int", "value": 1}
    },
    "b": {
      "state": "UPTODATE",
      "value": {"type": "int", "value": 2},
      "func": {"module": "mymodule", "qualname": "add_one"},
      "args": [],
      "kwds": {"x": "a"},
      "tags": ["__serialize__"]
    },
    "c": {
      "state": "ERROR",
      "error": {
        "type": "ValueError",
        "message": "something went wrong",
        "traceback": "Traceback (most recent call last):\n  ..."
      }
    },
    "d": {
      "state": "UNINITIALIZED"
    }
  },
  "edges": [
    {"source": "a", "target": "b", "param_type": "kwd", "param_name": "x"}
  ]
}
```

---

## Phase 1: Acceptance Tests (TDD Red)

**Goal:** Write all acceptance tests before any implementation. All tests fail until
implementation phases complete. By the end of Phase 4 all 22 tests must be green.

**File:** `tests/test_serialization.py` (append to existing file)

### Module-level helper functions

Required because JSON serialization can only reference importable functions. These are
defined at module level in the test file so they have stable `__module__` and `__qualname__`.

```python
def _json_add_one(x):
    return x + 1

def _json_double(x):
    return 2 * x

def _json_add(x, y):
    return x + y
```

### Transformer acceptance tests (7 tests)

| Test | Validates |
|------|-----------|
| `test_enum_transformer_states` | `States.UPTODATE` → dict → `States.UPTODATE` roundtrip |
| `test_func_ref_transformer_importable` | Module-level function → `{module, qualname}` → same function |
| `test_func_ref_transformer_lambda_raises` | Lambda raises `ValueError` in `to_dict` |
| `test_func_ref_transformer_closure_raises` | Closure raises `ValueError` in `to_dict` |
| `test_pandas_dataframe_transformer` | `pd.DataFrame` with mixed dtypes roundtrips correctly |
| `test_pandas_series_transformer` | `pd.Series` roundtrips correctly |
| `test_nodekey_transformer` | `NodeKey(("a", "b"))` → dict → `NodeKey(("a", "b"))` |

### Computation JSON API acceptance tests (15 tests)

| Test | Validates |
|------|-----------|
| `test_json_roundtrip_basic` | 4-node graph using `_json_*` helpers; nodes, states, values match after roundtrip |
| `test_json_roundtrip_skip_flag` | Node with `serialize=False` is `UNINITIALIZED` after roundtrip |
| `test_json_no_serialize_input` | Input node with `serialize=False` is `UNINITIALIZED` after roundtrip |
| `test_json_lambda_raises_serialization_error` | Graph with lambda → `write_json` raises `SerializationError` |
| `test_json_roundtrip_with_pandas_values` | Node with `pd.DataFrame` value survives roundtrip |
| `test_json_roundtrip_with_numpy_values` | Node with `np.ndarray` value survives roundtrip |
| `test_json_roundtrip_empty_graph` | `Computation()` with no nodes roundtrips cleanly |
| `test_json_roundtrip_file_path` | `write_json`/`read_json` accept string file paths (uses `tmp_path`) |
| `test_json_output_contains_version` | Written JSON has `"version"` key at top level |
| `test_json_output_is_valid_json_text` | Output is text (not binary), parseable by `json.loads` |
| `test_json_roundtrip_preserves_edges` | Edges with both kwd and arg parameters reconstruct correctly |
| `test_json_roundtrip_pinned_state` | PINNED node remains PINNED after roundtrip |
| `test_json_roundtrip_error_state` | ERROR node: state is ERROR, exception type and message preserved as strings |
| `test_json_roundtrip_with_nodekeys` | Hierarchical node names (`"foo/b"`) round-trip correctly |
| `test_json_roundtrip_with_block` | Block-based computation (using `add_block`) roundtrips correctly |

---

## Phase 2: Extend Transformer Framework

**Goal:** Add the four new transformer types needed for graph serialization.

**Files to modify:**
- `src/loman/serialization/transformer.py`
- `src/loman/serialization/default.py`
- `src/loman/serialization/__init__.py`

**Tasks:**

### 2.1 `EnumTransformer`

Serializes any registered `Enum` subclass by storing the class name and member name.
Maintains a registry of known enum classes for deserialization.

```python
class EnumTransformer(CustomTransformer):
    name = "enum"
    supported_subtypes = [Enum]

    def to_dict(self, transformer, o):
        return {"enum_class": type(o).__qualname__, "value": o.name}

    def from_dict(self, transformer, d):
        # Looks up registered enum class by qualname
        ...
```

Enum classes must be explicitly registered (e.g. `States`).

### 2.2 `FunctionRefTransformer`

Serializes importable callables by storing their `__module__` and `__qualname__`.
Raises `ValueError` for lambdas (qualname contains `<lambda>`) and closures
(qualname contains `<locals>`).

```python
class FunctionRefTransformer(CustomTransformer):
    name = "func_ref"
    supported_subtypes = [Callable]  # registered via supported_direct_types

    def to_dict(self, transformer, o):
        if "<lambda>" in o.__qualname__ or "<locals>" in o.__qualname__:
            raise ValueError(f"Cannot serialize non-importable function: {o!r}")
        return {"module": o.__module__, "qualname": o.__qualname__}

    def from_dict(self, transformer, d):
        module = importlib.import_module(d["module"])
        # Walk qualname parts to handle nested classes
        ...
```

### 2.3 `DataFrameTransformer` and `SeriesTransformer`

Serialize pandas objects. DataFrames use `orient="split"` for compact JSON representation.
Series stored with name, dtype, and data.

### 2.4 `NodeKeyTransformer`

Serializes `NodeKey` as its path string representation and reconstructs via `parse_nodekey`.

```python
class NodeKeyTransformer(CustomTransformer):
    name = "nodekey"
    supported_direct_types = [NodeKey]

    def to_dict(self, transformer, o):
        return {"path": str(o)}

    def from_dict(self, transformer, d):
        return parse_nodekey(d["path"])
```

**Acceptance tests that go green:** all 7 transformer tests from Phase 1.

---

## Phase 3: Create `ComputationSerializer`

**Goal:** Implement the class that converts a `Computation` to/from a JSON-serializable dict.

**File to create:** `src/loman/serialization/computation.py`

**Tasks:**

### 3.1 `ComputationSerializer` class

```python
class ComputationSerializer:
    VERSION = "1.0"

    def __init__(self, transformer: Transformer = None):
        self.transformer = transformer or _default_computation_transformer()

    def to_dict(self, computation: Computation) -> dict: ...
    def from_dict(self, d: dict) -> Computation: ...
    def write_json(self, computation: Computation, file_) -> None: ...
    def read_json(self, file_) -> Computation: ...
```

### 3.2 Node serialization

For each node in topological order:

- **State** always serialized as the enum name string
- **Value** serialized via transformer if state is `UPTODATE` or `PINNED`
- **Error info** serialized as strings if state is `ERROR` (type, message, traceback)
- **Func** serialized as `{module, qualname}` if present and not None — raises
  `SerializationError` if not importable
- **Args/kwds** serialized as lists/dicts of node name strings
- **Tags** serialized as list, excluding internal `SystemTags` from the list but preserving
  whether `__serialize__` was present (used to reconstruct `serialize=True/False` on add_node)

Nodes where `SystemTags.SERIALIZE` is not in tags are serialized as `UNINITIALIZED` with no
value (mirrors `__getstate__` behaviour).

### 3.3 Edge serialization

```python
def _serialize_edge(self, source, target, attr) -> dict:
    param_type, param_id = attr[EdgeAttributes.PARAM]
    if param_type == _ParameterType.ARG:
        return {"source": str(source), "target": str(target),
                "param_type": "arg", "param_index": param_id}
    else:
        return {"source": str(source), "target": str(target),
                "param_type": "kwd", "param_name": param_id}
```

### 3.4 Graph reconstruction

Use `add_node` with `inspect=False` (since kwds are already resolved from the serialized
arg/kwd maps, not inferred from function signatures). Restore state and value after adding
the node using `_set_state_and_literal_value`. Reconstruct edges manually using
`dag.add_edge`.

**Acceptance tests that go green:** all 15 computation tests from Phase 1.

---

## Phase 4: Add Public API

**Goal:** Expose `write_json` / `read_json` on the `Computation` class.

**File to modify:** `src/loman/computeengine.py`

```python
def write_json(self, file_: str | IO, *, transformer: Transformer = None) -> None:
    """Serialize computation to a JSON file.

    Only importable functions can be serialized. Lambdas and closures
    raise SerializationError.

    :param file_: File path (str) or writable text file-like object.
    :param transformer: Optional custom Transformer instance.
    :raises SerializationError: If any serializable node contains a
        non-importable function.
    """
    from loman.serialization.computation import ComputationSerializer
    ComputationSerializer(transformer).write_json(self, file_)

@staticmethod
def read_json(file_: str | IO, *, transformer: Transformer = None) -> "Computation":
    """Deserialize computation from a JSON file.

    :param file_: File path (str) or readable text file-like object.
    :param transformer: Optional custom Transformer instance.
    :returns: Restored Computation.
    """
    from loman.serialization.computation import ComputationSerializer
    return ComputationSerializer(transformer).read_json(file_)
```

Also update `src/loman/__init__.py` to export `ComputationSerializer`.

**Acceptance criteria:** All 22 Phase 1 tests are green.

---

## Phase 5: Deprecate Dill Methods

**Goal:** Signal to users that the dill API is going away.

**File to modify:** `src/loman/computeengine.py`

Add `DeprecationWarning` to `write_dill`, `read_dill`, and `write_dill_old` (which already
has a deprecation warning for a different reason):

```python
def write_dill(self, file_: str | BinaryIO) -> None:
    """Serialize computation using dill (deprecated).

    .. deprecated::
        Use :meth:`write_json` instead. dill serialization will be
        removed in the next major version.
    """
    warnings.warn(
        "write_dill is deprecated. Use write_json instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ...  # existing implementation unchanged
```

Update `docs/user/features/other/serializing_computations.md` to document `write_json` /
`read_json` as the primary API and note that `write_dill` / `read_dill` are deprecated.

**Acceptance criteria:** All existing dill tests still pass (with warnings); 22 JSON tests
still green.

---

## Phase 6: Migration Tooling

**Goal:** Help users convert existing `.dill` files to `.json`.

**File to create:** `src/loman/serialization/migration.py`

```python
def migrate_dill_to_json(dill_path: str, json_path: str, *, transformer=None) -> None:
    """Convert a dill-serialized computation file to JSON format.

    Requires the original dill file to be loadable (dill must be installed).
    Raises SerializationError if the computation contains non-importable functions.

    :param dill_path: Path to the existing .dill file.
    :param json_path: Path for the output .json file.
    :param transformer: Optional custom Transformer instance.
    """
    from loman.computeengine import Computation
    comp = Computation.read_dill(dill_path)
    comp.write_json(json_path, transformer=transformer)
```

**New tests** (`tests/test_serialization.py`):

- `test_migrate_dill_to_json` — creates a dill file, migrates, verifies JSON output
- `test_migrate_preserves_values` — values are correct after migration
- `test_migrate_non_importable_raises` — migration raises `SerializationError` for
  computations with non-importable functions

---

## Phase 7: Remove Dill Dependency

**Goal:** Complete removal of dill from the codebase.

**Files to modify:**

| File | Change |
|------|--------|
| `src/loman/computeengine.py` | Remove `import dill`; remove `__getstate__`, `__setstate__`, `write_dill_old`, `write_dill`, `read_dill` |
| `pyproject.toml` | Remove `dill >= 0.2.5` from dependencies |
| `tests/test_serialization.py` | Remove dill-specific test functions |
| `docs/user/features/other/serializing_computations.md` | Update to JSON API only |

**Checklist:**

- [ ] `import dill` removed from `computeengine.py`
- [ ] `__getstate__` removed
- [ ] `__setstate__` removed
- [ ] `write_dill_old` removed
- [ ] `write_dill` removed
- [ ] `read_dill` removed
- [ ] `dill >= 0.2.5` removed from `pyproject.toml`
- [ ] `dill` removed from `uv.lock` (run `uv lock`)
- [ ] All dill-related tests removed or converted
- [ ] Docs updated
- [ ] `bandit` skip comments for `B301`/`B403` removed if present

**Acceptance criteria:** `uv run pytest` passes with no dill references anywhere; `dill` is not
importable as a loman dependency.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Users rely on lambda serialization | Clear error message; document `serialize=False` pattern |
| Existing `.dill` files become inaccessible | Phase 6 migration tool; `read_dill` kept through Phase 5 |
| Custom types not handled | Extensible Transformer; document `CustomTransformer` pattern |
| Nested Computation as node value | Explicitly unsupported for now; raises `SerializationError` |
| JSON is slower than binary for large arrays | numpy arrays stored as lists; acceptable for operational use |

---

## Out of Scope

- Nested `Computation` as a node value (deferred — requires `ComputationTransformer`)
- Binary JSON formats (MessagePack, CBOR) for performance
- `_metadata` serialization
- Executor configuration serialization
- Converter function serialization
- Long-term stable storage format (format may change between versions)
