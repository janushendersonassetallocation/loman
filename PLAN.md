# Plan: Remove Dill and Pickle from Loman

## Executive Summary

This plan outlines the migration from dill-based serialization to a safe, JSON-based approach using the existing `Transformer` framework. The goal is to eliminate arbitrary code execution risks while maintaining practical serialization capabilities.

## Current State

### Dill Usage Locations

| File | Lines | Purpose |
|------|-------|---------|
| `src/loman/computeengine.py` | 17, 1512-1548 | Core serialization methods |
| `tests/test_dill_serialization.py` | 1-192 | Serialization tests |
| `tests/test_coverage_gaps.py` | 797-839, 1828-1835, 2994-3022 | Additional tests |

### Methods to Replace

| Method | Lines | Status |
|--------|-------|--------|
| `write_dill()` | 1519-1529 | Active API |
| `read_dill()` | 1531-1552 | Active API |
| `write_dill_old()` | 1488-1517 | Deprecated |
| `__getstate__()` | 1473-1480 | Pickle protocol |
| `__setstate__()` | 1482-1486 | Pickle protocol |

### Why Remove Dill

1. **Security Risk**: `dill.load()` can execute arbitrary code
2. **Portability**: Binary format is Python-version sensitive
3. **Transparency**: Binary format is not human-readable or auditable
4. **Dependency**: Reduces external dependencies

## Proposed Solution: Hybrid Approach

Use the existing `Transformer` framework for values and graph structure, with importable function references for callables.

### Output Format

```json
{
  "version": "1.0",
  "nodes": {
    "price": {
      "state": "UPTODATE",
      "value": {"type": "int", "value": 100},
      "tags": ["__serialize__"]
    },
    "quantity": {
      "state": "UPTODATE",
      "value": {"type": "int", "value": 5},
      "tags": ["__serialize__"]
    },
    "total": {
      "state": "UPTODATE",
      "value": {"type": "int", "value": 500},
      "func": {"module": "mymodule", "name": "calculate_total"},
      "args": [],
      "kwds": {"price": "price", "quantity": "quantity"},
      "tags": ["__serialize__"]
    }
  },
  "edges": [
    {"source": "price", "target": "total"},
    {"source": "quantity", "target": "total"}
  ]
}
```

## Implementation Phases

### Phase 1: Extend Transformer Framework

**Goal**: Add support for all node attribute types

**Files to modify**:
- `src/loman/serialization/transformer.py`

**Tasks**:

1. Add `EnumTransformer` for `States` enum
   ```python
   class EnumTransformer(CustomTransformer):
       @property
       def name(self):
           return "enum"

       def to_dict(self, transformer, o):
           return {"enum_class": type(o).__name__, "value": o.name}

       def from_dict(self, transformer, d):
           # Registry lookup for known enums
           return ENUM_REGISTRY[d["enum_class"]][d["value"]]
   ```

2. Add `FunctionRefTransformer` for callable references
   ```python
   class FunctionRefTransformer(CustomTransformer):
       @property
       def name(self):
           return "func_ref"

       def to_dict(self, transformer, o):
           if not callable(o):
               raise ValueError("Not callable")
           module = getattr(o, "__module__", None)
           name = getattr(o, "__qualname__", None)
           if module is None or name is None:
               raise ValueError(f"Cannot serialize {o}: not importable")
           return {"module": module, "name": name}

       def from_dict(self, transformer, d):
           module = importlib.import_module(d["module"])
           return getattr(module, d["name"])
   ```

3. Add `PandasTransformer` for DataFrame/Series
   ```python
   class DataFrameTransformer(CustomTransformer):
       @property
       def name(self):
           return "dataframe"

       def to_dict(self, transformer, o):
           return {
               "columns": list(o.columns),
               "data": transformer.to_dict(o.to_dict(orient="list")),
               "index": transformer.to_dict(o.index.tolist())
           }
   ```

4. Add `NodeKeyTransformer` for hierarchical node keys
   ```python
   class NodeKeyTransformer(CustomTransformer):
       @property
       def name(self):
           return "nodekey"

       def to_dict(self, transformer, o):
           return {"parts": list(o)}

       def from_dict(self, transformer, d):
           return NodeKey(*d["parts"])
   ```

**Estimated effort**: Medium

---

### Phase 2: Create Graph Serializer

**Goal**: Implement `ComputationSerializer` class

**Files to create**:
- `src/loman/serialization/computation.py`

**Tasks**:

1. Create `ComputationSerializer` class
   ```python
   class ComputationSerializer:
       VERSION = "1.0"

       def __init__(self, transformer: Transformer = None):
           self.transformer = transformer or default_transformer()

       def to_dict(self, computation: Computation) -> dict:
           """Convert Computation to serializable dict."""
           ...

       def from_dict(self, d: dict) -> Computation:
           """Reconstruct Computation from dict."""
           ...

       def write_json(self, computation: Computation, file_) -> None:
           """Write computation to JSON file."""
           ...

       def read_json(self, file_) -> Computation:
           """Read computation from JSON file."""
           ...
   ```

2. Implement node serialization logic
   ```python
   def _serialize_node(self, name, attrs) -> dict:
       node_dict = {
           "state": attrs.get(NodeAttributes.STATE, States.UNINITIALIZED).name
       }

       # Only serialize value if state is UPTODATE
       if attrs.get(NodeAttributes.STATE) == States.UPTODATE:
           node_dict["value"] = self.transformer.to_dict(
               attrs.get(NodeAttributes.VALUE)
           )

       # Serialize function reference if present
       if NodeAttributes.FUNC in attrs:
           func = attrs[NodeAttributes.FUNC]
           if func is not None:
               node_dict["func"] = self._serialize_func_ref(func)

       # Serialize args/kwds
       if NodeAttributes.ARGS in attrs:
           node_dict["args"] = list(attrs[NodeAttributes.ARGS])
       if NodeAttributes.KWDS in attrs:
           node_dict["kwds"] = dict(attrs[NodeAttributes.KWDS])

       # Serialize tags (excluding internal system tags)
       tags = attrs.get(NodeAttributes.TAG, set())
       if tags:
           node_dict["tags"] = list(tags)

       return node_dict
   ```

3. Implement graph reconstruction logic
   ```python
   def _deserialize_node(self, computation, name, node_dict):
       func = None
       if "func" in node_dict:
           func = self._deserialize_func_ref(node_dict["func"])

       args = node_dict.get("args", [])
       kwds = node_dict.get("kwds", {})
       tags = set(node_dict.get("tags", []))

       computation.add_node(
           name,
           func=func,
           args=args,
           kwds=kwds,
           tags=tags,
           serialize=SystemTags.SERIALIZE in tags
       )

       # Restore value if present
       if "value" in node_dict:
           value = self.transformer.from_dict(node_dict["value"])
           computation._set_value(name, value)
           computation._set_state(name, States[node_dict["state"]])
   ```

**Estimated effort**: Medium-High

---

### Phase 3: Add New Public API

**Goal**: Add `write_json()` and `read_json()` methods to `Computation`

**Files to modify**:
- `src/loman/computeengine.py`

**Tasks**:

1. Add new serialization methods
   ```python
   def write_json(self, file_, *, transformer=None):
       """Serialize computation to JSON file.

       Args:
           file_: File path (str) or file-like object
           transformer: Optional custom Transformer instance

       Note:
           Only importable functions can be serialized. Lambdas and
           closures will raise ValueError.
       """
       from loman.serialization.computation import ComputationSerializer
       serializer = ComputationSerializer(transformer)
       serializer.write_json(self, file_)

   @staticmethod
   def read_json(file_, *, transformer=None):
       """Deserialize computation from JSON file.

       Args:
           file_: File path (str) or file-like object
           transformer: Optional custom Transformer instance

       Returns:
           Computation: Restored computation graph
       """
       from loman.serialization.computation import ComputationSerializer
       serializer = ComputationSerializer(transformer)
       return serializer.read_json(file_)
   ```

2. Update `__init__.py` exports
   ```python
   __all__ = [
       ...
       "ComputationSerializer",  # New export
   ]
   ```

**Estimated effort**: Low

---

### Phase 4: Deprecate Dill Methods

**Goal**: Mark dill methods as deprecated with migration warnings

**Files to modify**:
- `src/loman/computeengine.py`

**Tasks**:

1. Add deprecation warnings to existing methods
   ```python
   def write_dill(self, file_):
       """Serialize computation using dill (DEPRECATED).

       .. deprecated:: 0.6.0
           Use :meth:`write_json` instead. Dill serialization will be
           removed in version 1.0.0.
       """
       warnings.warn(
           "write_dill is deprecated and will be removed in v1.0.0. "
           "Use write_json instead for safe serialization.",
           DeprecationWarning,
           stacklevel=2
       )
       # ... existing implementation

   @staticmethod
   def read_dill(file_):
       """Deserialize computation using dill (DEPRECATED).

       .. deprecated:: 0.6.0
           Use :meth:`read_json` instead.
       """
       warnings.warn(
           "read_dill is deprecated and will be removed in v1.0.0. "
           "Use read_json instead for safe serialization.",
           DeprecationWarning,
           stacklevel=2
       )
       # ... existing implementation
   ```

2. Update documentation to recommend `write_json`/`read_json`

**Estimated effort**: Low

---

### Phase 5: Add Migration Tooling

**Goal**: Help users migrate existing dill files to JSON

**Files to create**:
- `src/loman/serialization/migration.py`

**Tasks**:

1. Create migration utility
   ```python
   def migrate_dill_to_json(dill_path, json_path, *, transformer=None):
       """Convert dill-serialized computation to JSON format.

       Args:
           dill_path: Path to existing .dill file
           json_path: Path for output .json file
           transformer: Optional custom Transformer

       Raises:
           ValueError: If computation contains non-importable functions
       """
       comp = Computation.read_dill(dill_path)
       comp.write_json(json_path, transformer=transformer)
   ```

2. Add CLI command (optional)
   ```bash
   python -m loman.migrate input.dill output.json
   ```

**Estimated effort**: Low

---

### Phase 6: Remove Dill Dependency

**Goal**: Complete removal of dill from codebase

**Files to modify**:
- `src/loman/computeengine.py` - Remove dill import and methods
- `pyproject.toml` - Remove dill from dependencies
- `tests/test_dill_serialization.py` - Convert or remove tests

**Tasks**:

1. Remove `import dill` statement (line 17)

2. Remove methods:
   - `__getstate__()` (lines 1473-1480)
   - `__setstate__()` (lines 1482-1486)
   - `write_dill_old()` (lines 1488-1517)
   - `write_dill()` (lines 1519-1529)
   - `read_dill()` (lines 1531-1552)

3. Update `pyproject.toml`:
   ```diff
   dependencies = [
       "networkx >= 2.0",
   -   "dill >= 0.2.5",
       "pandas",
       ...
   ]
   ```

4. Convert tests to use JSON serialization

**Estimated effort**: Medium

---

## Testing Strategy

### New Tests Required

1. **Transformer tests** (`tests/test_transformer.py`)
   - `test_enum_transformer`
   - `test_func_ref_transformer_importable`
   - `test_func_ref_transformer_lambda_fails`
   - `test_pandas_dataframe_transformer`
   - `test_nodekey_transformer`

2. **ComputationSerializer tests** (`tests/test_json_serialization.py`)
   - `test_json_roundtrip_basic`
   - `test_json_roundtrip_with_values`
   - `test_json_serialize_skip_flag`
   - `test_json_nested_computation`
   - `test_json_lambda_raises_error`
   - `test_json_version_compatibility`

3. **Migration tests** (`tests/test_migration.py`)
   - `test_migrate_dill_to_json`
   - `test_migrate_preserves_values`
   - `test_migrate_non_importable_fails`

### Test Coverage Goals

- Maintain 100% coverage during migration
- All existing serialization behaviors must have JSON equivalents
- Edge cases: empty graphs, circular references, custom types

---

## Migration Path for Users

### Before (Current API)

```python
# Save computation
comp.write_dill("computation.dill")

# Load computation
comp = Computation.read_dill("computation.dill")
```

### After (New API)

```python
# Save computation
comp.write_json("computation.json")

# Load computation
comp = Computation.read_json("computation.json")
```

### Handling Lambdas

**Before** (works with dill):
```python
comp.add_node("b", lambda a: a + 1)
comp.write_dill("computation.dill")  # Works
```

**After** (requires importable function):
```python
# Define function in module
def increment(a):
    return a + 1

comp.add_node("b", increment)
comp.write_json("computation.json")  # Works
```

**Alternative** (values-only serialization):
```python
# For lambdas, serialize values only and rebuild graph
comp.add_node("b", lambda a: a + 1, serialize=False)
comp.write_json("computation.json")  # Skips 'b' function, keeps value if computed
```

---

## Timeline

| Phase | Description | Dependencies |
|-------|-------------|--------------|
| 1 | Extend Transformer Framework | None |
| 2 | Create Graph Serializer | Phase 1 |
| 3 | Add New Public API | Phase 2 |
| 4 | Deprecate Dill Methods | Phase 3 |
| 5 | Add Migration Tooling | Phase 3, 4 |
| 6 | Remove Dill Dependency | Phase 4, 5 + deprecation period |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Users rely on lambda serialization | Document alternatives; provide `serialize=False` pattern |
| Breaking change for existing .dill files | Provide migration tool; maintain read_dill for one version |
| Custom types not supported | Extensible Transformer allows registering custom types |
| Performance regression | JSON is text-based; benchmark and optimize if needed |

## Success Criteria

1. All existing tests pass with JSON serialization
2. No dill/pickle imports in production code
3. Migration tool successfully converts test .dill files
4. Documentation updated with new API
5. Deprecation warnings visible for one release cycle

## Open Questions

1. Should we support a binary JSON format (e.g., MessagePack) for performance?
2. How long should the deprecation period be before removing dill entirely?
3. Should we provide a "strict mode" that fails if any node has non-importable functions?
