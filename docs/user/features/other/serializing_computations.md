# Serializing computations

Loman can serialize computations to a JSON file for later inspection or post-mortem debugging. This is useful when a scheduled job should capture its inputs, intermediates, and results so they can be examined if something goes wrong.

```pycon
>>> import math
>>> from loman import Computation
>>> comp = Computation()
>>> comp.add_node('x', value=4.0)
>>> def area(x):
...     return math.pi * x ** 2
>>> comp.add_node('area', area)
>>> comp.compute_all()
>>> comp.to_dict()
{'x': 4.0, 'area': 50.26548245743669}
```

To save and reload the computation:

```pycon
>>> comp.write_json('comp.json')
>>> comp2 = Computation.read_json('comp.json')
>>> comp2.v.area
50.26548245743669
```

The output is a plain JSON text file, so it is human-readable and can be inspected with any text editor.

## Excluding nodes from serialization

Sometimes a node holds a value that should not (or cannot) be saved — for example a database connection, a licensed dataset, or an object that does not support JSON serialization. Pass `serialize=False` when adding the node:

```pycon
>>> import sqlalchemy as sa
>>> comp = Computation()
>>> comp.add_node('engine', sa.create_engine('sqlite://'), serialize=False)
>>> comp.add_node('result', value=42)
>>> comp.write_json('comp.json')
>>> comp2 = Computation.read_json('comp.json')
>>> comp2.state('engine')
<States.UNINITIALIZED: 1>
>>> comp2.v.result
42
```

The excluded node is preserved in the file with `UNINITIALIZED` state and no value; all other nodes round-trip normally.

## Lambdas are not serializable by default

A lambda cannot be serialized because it has no importable module path. Use a module-level function instead:

```pycon
>>> from loman import Computation, ComputationSerializer, SerializationError
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: a + 1)
>>> comp.compute_all()
>>> import io
>>> try:
...     comp.write_json(io.StringIO())
... except SerializationError as e:
...     print(e)
Cannot serialize lambda function on node NodeKey(parts=('b',)). Use a module-level importable function, serialize=False, or ComputationSerializer(use_dill_for_functions=True).
```

Replace the lambda with a named function defined at module level:

```pycon
>>> def increment(a):
...     return a + 1
>>> comp.add_node('b', increment)
>>> comp.compute_all()
>>> comp.write_json('comp.json')       # now succeeds
```

### Using dill to serialize lambdas and closures

When refactoring to named functions is impractical, pass `use_dill_for_functions=True` to `ComputationSerializer`. This serializes any callable — including lambdas and closures that capture local variables — as a base64-encoded [dill](https://github.com/uqfoundation/dill) blob inside the JSON:

```pycon
>>> s = ComputationSerializer(use_dill_for_functions=True)
>>> comp = Computation()
>>> comp.add_node('a', value=3)
>>> comp.add_node('b', lambda a: a * 2)
>>> comp.compute_all()
>>> buf = io.StringIO()
>>> comp.write_json(buf, serializer=s)
>>> _ = buf.seek(0)
>>> comp2 = Computation.read_json(buf, serializer=s)
>>> comp2.v.b
6
>>> comp2.insert('a', 10)
>>> comp2.compute_all()
>>> comp2.v.b
20
```

The same serializer instance must be passed to both `write_json` and `read_json`.

!!! warning
    The dill blob embedded in the JSON is **not portable across Python versions** and shares the same stability caveats as the deprecated `write_dill`. Prefer named functions when long-term compatibility matters.

## File objects and strings

Both `write_json` and `read_json` accept either a file path (string) or any text-mode file-like object:

```pycon
>>> import io
>>> buf = io.StringIO()
>>> comp.write_json(buf)
>>> _ = buf.seek(0)
>>> comp3 = Computation.read_json(buf)
>>> comp3.v.b
2
```

## PINNED nodes

Pinned nodes round-trip correctly — their `PINNED` state and value are preserved:

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=10)
>>> comp.pin('a')
>>> comp.write_json('comp.json')
>>> comp2 = Computation.read_json('comp.json')
>>> comp2.state('a')
<States.PINNED: 5>
>>> comp2.v.a
10
```

## ERROR nodes

If a node is in `ERROR` state, its exception type, message, and traceback are preserved as strings so they can be read back for post-mortem inspection even without the original exception class:

```pycon
>>> def bad_func():
...     raise ValueError("something went wrong")
>>> comp = Computation()
>>> comp.add_node('result', bad_func)
>>> comp.compute_all()
>>> comp.state('result')
<States.ERROR: 4>
>>> comp.write_json('comp.json')
>>> comp2 = Computation.read_json('comp.json')
>>> comp2.state('result')
<States.ERROR: 4>
>>> comp2['result'].value.exception
Exception('something went wrong')
```

## Custom serialization for user-defined types

For types that are not handled by the default serializer, pass a custom `ComputationSerializer` instance with additional transformers registered:

```pycon
>>> from loman import Computation, ComputationSerializer
>>> from loman.serialization import CustomTransformer, Transformer
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
>>> point_transformer = CustomTransformer(
...     Point,
...     to_dict=lambda v: {'__point__': True, 'x': v.x, 'y': v.y},
...     from_dict=lambda d: Point(d['x'], d['y']),
... )
>>> s = ComputationSerializer()
>>> s._t.register(point_transformer)
>>> comp = Computation()
>>> comp.add_node('origin', value=Point(0, 0))
>>> buf = io.StringIO()
>>> comp.write_json(buf, serializer=s)
>>> _ = buf.seek(0)
>>> comp2 = Computation.read_json(buf, serializer=s)
>>> comp2.v.origin.x
0
```

## Pandas support

DataFrames and Series are serialized automatically:

```pycon
>>> import pandas as pd
>>> comp = Computation()
>>> comp.add_node('df', value=pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
>>> buf = io.StringIO()
>>> comp.write_json(buf)
>>> _ = buf.seek(0)
>>> comp2 = Computation.read_json(buf)
>>> comp2.v.df.shape
(2, 2)
```

## JSON format reference

The file is a single JSON object with three top-level keys:

```json
{
  "version": 1,
  "nodes": [ ... ],
  "edges": [ ... ]
}
```

### Node object

Each entry in `nodes` has:

| Field | Type | Description |
|---|---|---|
| `key` | string | Node name. Hierarchical keys use `/` as separator. |
| `state` | string \| null | `States` enum name: `"UPTODATE"`, `"STALE"`, `"UNINITIALIZED"`, `"ERROR"`, `"PINNED"`, … |
| `value` | any | Encoded value (see below), or `null` when absent. |
| `has_value` | bool | `true` when `value` should be restored; `false` when the node has no value. |
| `func` | object \| null | Encoded callable (see below), or `null`. |
| `serialize` | bool | Whether the node carries the `__serialize__` tag. |
| `tags` | list[string] | Non-system user tags. |

### Edge object

Each entry in `edges` has:

| Field | Type | Description |
|---|---|---|
| `src` | string | Source node key. |
| `dst` | string | Destination node key. |
| `param_type` | `"arg"` \| `"kwd"` \| null | How the value is passed to the function. |
| `param` | int \| string \| null | Positional index for `"arg"`, parameter name for `"kwd"`. |

### Value encoding

Plain Python scalars (`int`, `float`, `str`, `bool`, `None`) are stored as-is.
Compound types use a tagged object with a `"type"` discriminator:

**NumPy array**

```json
{
  "type": "ndarray",
  "shape": [3],
  "dtype": "<f8",
  "data": [1.0, 2.0, 3.0]
}
```

**Pandas DataFrame** (split orientation, column dtypes preserved)

```json
{
  "type": "dataframe",
  "columns": ["x", "y"],
  "index": [0, 1],
  "data": [[1.0, 3.0], [2.0, 4.0]],
  "dtypes": {"x": "int64", "y": "float64"}
}
```

**ERROR node value** (exception preserved as strings for post-mortem)

```json
{
  "__loman_error__": true,
  "exception_type": "ValueError",
  "exception_str": "something went wrong",
  "traceback": "Traceback (most recent call last):\n  ..."
}
```

### Function encoding

**Importable module-level function** (default)

```json
{
  "type": "func_ref",
  "module": "mypackage.calcs",
  "qualname": "compute_result"
}
```

**Lambda or closure** (only when `use_dill_for_functions=True`)

```json
{
  "type": "dill_func",
  "blob": "gASVyQAAAAAAAACMCmRpbGwuX2RpbGyU..."
}
```

The `blob` field is a base64-encoded [dill](https://github.com/uqfoundation/dill) byte string. It is not portable across Python versions.

!!! note
    The JSON serialization format is not intended for long-term storage. It is designed for short-term inspection and post-mortem debugging. The format may change between releases.

