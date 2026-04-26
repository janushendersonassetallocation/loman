# Migrating from write_dill to write_json

`write_dill` and `read_dill` are deprecated and will be removed in a future release. This page explains how to migrate to the replacement `write_json` / `read_json` API.

## Quick substitution

The call signatures are the same — both accept a file path or a file-like object:

```python
# Before (deprecated)
comp.write_dill('comp.dill')
comp2 = Computation.read_dill('comp.dill')

# After
comp.write_json('comp.json')
comp2 = Computation.read_json('comp.json')
```

## Key differences

| | `write_dill` / `read_dill` | `write_json` / `read_json` |
|---|---|---|
| Format | Binary (dill/pickle) | Text (JSON) |
| File mode | Binary (`'wb'` / `'rb'`) | Text (`'w'` / `'r'`) |
| Human-readable | No | Yes |
| Lambdas | Serialized | Raises `SerializationError` |
| Custom types | Any picklable type | Requires a custom `ComputationSerializer` |
| Long-term storage | Not recommended | Not recommended |

## Lambdas must be replaced (or opt in to dill)

`write_dill` serialized lambdas via pickle. `write_json` raises `SerializationError` if a node's function is a lambda. The cleanest fix is to replace lambdas with module-level functions:

```python
# Before — works with write_dill, fails with write_json
comp.add_node('b', lambda a: a + 1)

# After — works with write_json
def increment(a):
    return a + 1

comp.add_node('b', increment)
```

If refactoring is impractical, there are two escape hatches:

**Option 1 — serialize the value only** (function is lost, node cannot be re-run after load):

```python
comp.add_node('b', lambda a: a + 1, serialize=False)
```

**Option 2 — use `ComputationSerializer(use_dill_for_functions=True)`** (function is preserved as a dill blob, re-computation works after load):

```python
from loman import ComputationSerializer

s = ComputationSerializer(use_dill_for_functions=True)
comp.write_json('comp.json', serializer=s)
comp2 = Computation.read_json('comp.json', serializer=s)
```

The same serializer instance must be used for both write and read. The dill blob is not portable across Python versions.

## File-like objects must be text-mode

`write_dill` used binary mode. `write_json` uses text mode:

```python
import io

# Before
buf = io.BytesIO()
comp.write_dill(buf)

# After
buf = io.StringIO()
comp.write_json(buf)
```

## Custom types

If your computation holds values of types that are not handled by the default serializer (anything beyond Python scalars, lists, dicts, NumPy arrays, and Pandas DataFrames/Series), you need to register a custom transformer. See the [Serializing Computations](serializing_computations.md#custom-serialization-for-user-defined-types) page for an example.
