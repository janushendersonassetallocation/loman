# Serializing computations

Loman can serialize computations to a JSON file for later inspection or post-mortem debugging. This is useful when a scheduled job should capture its inputs, intermediates, and results so they can be examined if something goes wrong.

```pycon
>>> import math
>>> from loman import Computation
>>> comp = Computation()
>>> comp.add_node("x", value=4.0)
>>> def area(x):
...     return math.pi * x**2
>>> comp.add_node("area", area)
>>> comp.compute_all()
>>> comp.to_dict()
{NodeKey('x'): 4.0, NodeKey('area'): 50.26548245743669}
```

To save and reload the computation:

```pycon
>>> comp.write_json("comp.json")
>>> comp2 = Computation.read_json("comp.json")
>>> comp2.v.area
50.26548245743669
```

The output is a plain JSON text file, so it is human-readable and can be inspected with any text editor.

## Excluding nodes from serialization

Sometimes a node holds a value that should not (or cannot) be saved — for example a database connection, a licensed dataset, or an object that does not support JSON serialization. Pass `serialize=False` when adding the node:

```pycon
>>> import threading
>>> comp = Computation()
>>> comp.add_node("lock", value=threading.Lock(), serialize=False)
>>> comp.add_node("result", value=42)
>>> comp.write_json("comp.json")
>>> comp2 = Computation.read_json("comp.json")
>>> comp2.state("lock")
<States.UNINITIALIZED: 1>
>>> comp2.v.result
42
```

A database engine or an open connection behaves the same way — anything whose
value is tied to the running process rather than to the data.

The excluded node is preserved in the file with `UNINITIALIZED` state and no value; all other nodes round-trip normally.

## Lambdas are not serializable by default

A lambda cannot be serialized because it has no importable module path. Use a module-level function instead:

```pycon
>>> from loman import Computation, ComputationSerializer, SerializationError
>>> comp = Computation()
>>> comp.add_node("a", value=1)
>>> comp.add_node("b", lambda a: a + 1)
>>> comp.compute_all()
>>> import io
>>> try:
...     comp.write_json(io.StringIO())
... except SerializationError as e:
...     print(e)
Cannot serialize lambda function on node NodeKey('b'). Use a module-level importable function, serialize=False, or ComputationSerializer(use_dill_for_functions=True).
```

Replace the lambda with a named function defined at module level:

```pycon
>>> def increment(a):
...     return a + 1
>>> comp.add_node("b", increment)
>>> comp.compute_all()
>>> comp.write_json("comp.json")  # now succeeds
```

### Using dill to serialize lambdas and closures

When refactoring to named functions is impractical, pass `use_dill_for_functions=True` to `ComputationSerializer`. This serializes any callable — including lambdas and closures that capture local variables — as a base64-encoded [dill](https://github.com/uqfoundation/dill) blob inside the JSON:

```pycon
>>> s = ComputationSerializer(use_dill_for_functions=True)
>>> comp = Computation()
>>> comp.add_node("a", value=3)
>>> comp.add_node("b", lambda a: a * 2)
>>> comp.compute_all()
>>> buf = io.StringIO()
>>> comp.write_json(buf, serializer=s)
>>> _ = buf.seek(0)
>>> comp2 = Computation.read_json(buf, serializer=s)
>>> comp2.v.b
6
>>> comp2.insert("a", 10)
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
>>> comp = Computation()
>>> comp.add_node("a", value=1)
>>> comp.add_node("b", increment)
>>> comp.compute_all()
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
>>> comp.add_node("a", value=10)
>>> comp.pin("a")
>>> comp.write_json("comp.json")
>>> comp2 = Computation.read_json("comp.json")
>>> comp2.state("a")
<States.PINNED: 6>
>>> comp2.v.a
10
```

## ERROR nodes

If a node is in `ERROR` state, its exception type, message, and traceback are preserved so they can be read back for post-mortem inspection. Builtin exception types are rebuilt as themselves, so `except ValueError` still matches after a round-trip:

```pycon
>>> def bad_func():
...     raise ValueError("something went wrong")
>>> comp = Computation()
>>> comp.add_node("result", bad_func)
>>> comp.compute_all()
>>> comp.state("result")
<States.ERROR: 5>
>>> comp.write_json("comp.json")
>>> comp2 = Computation.read_json("comp.json")
>>> comp2.state("result")
<States.ERROR: 5>
>>> comp2["result"].value.exception
ValueError('something went wrong')
```

A non-builtin exception class is *not* rebuilt: doing so would mean importing
whatever module the file names, which is running code the file chose. Those come
back as `loman.exception.DeserializedError`, carrying the original type name and
module as attributes so the post-mortem still tells you what was raised:

```pycon
>>> from loman.exception import DeserializedError
>>> class MyLibraryError(Exception):
...     pass
>>> def also_bad():
...     raise MyLibraryError("domain-specific failure")
>>> comp = Computation()
>>> comp.add_node("result", also_bad)
>>> comp.compute_all()
>>> comp.write_json("comp.json")
>>> exc = Computation.read_json("comp.json")["result"].value.exception
>>> isinstance(exc, DeserializedError)
True
>>> exc.exception_type
'MyLibraryError'
>>> str(exc)
'domain-specific failure'
```

## Custom serialization for user-defined types

For types that are not handled by the default serializer, pass a custom `ComputationSerializer` instance with additional transformers registered:

```pycon
>>> import io
>>> from loman import Computation, ComputationSerializer
>>> from loman.serialization import SimpleTransformer
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
>>> point_transformer = SimpleTransformer(
...     "point",
...     Point,
...     to_dict=lambda v: {"x": v.x, "y": v.y},
...     from_dict=lambda d: Point(d["x"], d["y"]),
... )
>>> s = ComputationSerializer()
>>> s.register(point_transformer)
>>> comp = Computation()
>>> comp.add_node("origin", value=Point(0, 0))
>>> buf = io.StringIO()
>>> comp.write_json(buf, serializer=s)
>>> _ = buf.seek(0)
>>> comp2 = Computation.read_json(buf, serializer=s)
>>> comp2.v.origin.x
0
```

The first argument is the discriminator written as the value's `"type"` field, so
it must be unique within a serializer and stable across releases — changing it
makes previously written files unreadable.

`SimpleTransformer` passes values straight to your callables without recursing.
When the encoding needs to nest other transformable values inside it, or to write
bytes out-of-line, subclass `CustomTransformer` instead:

```pycon
>>> from loman.serialization import CustomTransformer
>>> class PolygonTransformer(CustomTransformer):
...     @property
...     def name(self):
...         return "polygon"
...
...     def to_dict(self, transformer, o):
...         return {"points": transformer.to_dict(o.points)}
...
...     def from_dict(self, transformer, d):
...         return Polygon(transformer.from_dict(d["points"]))
...
...     @property
...     def supported_direct_types(self):
...         return [Polygon]
```

## Constant arguments that cannot be encoded

A constant argument given as `C(...)` is held on the node rather than on an edge,
and the node's function cannot be called without it. So where an unencodable
*function* is stored as `null` and the node merely loses the ability to
recalculate, an unencodable *constant* raises `SerializationError` naming the node
and the parameter, rather than writing a file that would raise `TypeError` from
the missing argument on the first recalculation:

```pycon
>>> from loman import Computation, C, ComputationSerializer
>>> comp = Computation()
>>> comp.add_node("number", value=1.2345)
>>> comp.add_node("rounded", round, kwds={"ndigits": C(object())})
>>> comp.write_json(io.StringIO())
Traceback (most recent call last):
    ...
loman.exception.SerializationError: Cannot serialize constant argument 'ndigits' on node NodeKey('rounded') ...
```

!!! note
    The check only reaches a node's constants when the node's *function* could
    itself be encoded. A function that is not importable — a lambda, a closure,
    or anything defined interactively — is stored as `null` first, and its
    constants are then never examined. This is why the example above uses a
    builtin: in a module-level script your own functions are importable and
    behave the same way, but pasted into a REPL they are not.

The remedy is usually to register a transformer for the type, as in
[Custom serialization for user-defined types](#custom-serialization-for-user-defined-types)
above, or to set `serialize=False` on the node.

Releases before this behaviour existed dropped such a constant silently. To keep
writing files while an existing codebase is fixed, ask for the old behaviour
explicitly — the constant is omitted and an `UnserializableConstantWarning` says
so, which is a step towards fixing it rather than a setting to leave in place:

```pycon
>>> serializer = ComputationSerializer(on_unserializable_constant="drop")
>>> comp.write_json(io.StringIO(), serializer=serializer)  # doctest: +SKIP
UnserializableConstantWarning: Cannot serialize constant argument 'ndigits' on node
NodeKey('rounded') ... Dropping it, because on_unserializable_constant='drop'. The
saved graph will raise TypeError from the missing argument when this node is
recalculated.
```

## Pandas support

DataFrames and Series are serialized automatically:

```pycon
>>> import pandas as pd
>>> comp = Computation()
>>> comp.add_node("df", value=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
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
  "version": 2,
  "nodes": [ ... ],
  "edges": [ ... ]
}
```

Version 2 added the node `args` and `kwds` maps described below. The two versions
are readable in either direction: a version 1 file loads here, its missing maps
read as "no constant arguments", and a version 2 file loads under releases that
predate them, which ignore the new fields. Neither direction repairs the other —
a graph saved as version 1 has no record of its constants, so it must be saved
again to gain them.

### Node object

Each entry in `nodes` has:

| Field | Type | Description |
|---|---|---|
| `key` | string | Node name. Hierarchical keys use `/` as separator. |
| `state` | string \| null | `States` enum name: `"UPTODATE"`, `"STALE"`, `"UNINITIALIZED"`, `"ERROR"`, `"PINNED"`, … |
| `value` | any | Encoded value (see below), or `null` when absent. |
| `has_value` | bool | `true` when `value` should be restored; `false` when the node has no value. |
| `func` | object \| null | Encoded callable (see below), or `null`. |
| `args` | object | Constant positional arguments, keyed by stringified index. Empty when the node has none. |
| `kwds` | object | Constant keyword arguments, keyed by parameter name. Empty when the node has none. |
| `serialize` | bool | Whether the node carries the `__serialize__` tag. |
| `tags` | list[string] | Non-system user tags. |

A node's function arguments come from two places, and a reader needs both to call
it. An argument taken from another node is recorded on an **edge**; an argument
given as a `ConstantValue` (`C(...)`) belongs to no node, so it is held on the
node itself in `args` and `kwds`. Dropping the
constants leaves the function short of arguments, which is why an unencodable one
raises rather than being skipped — see
[Constant arguments that cannot be encoded](#constant-arguments-that-cannot-be-encoded).

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

## How durable is a saved computation?

The blanket answer used to be "not for long-term storage". That is no longer
accurate, and the honest answer splits in two.

**Values are durable.** The format carries a version, changes to it are additive,
and files written by every earlier version are held as fixtures in the test suite
and asserted to still load *and still recompute*. Arrays and frames are stored as
`.npy` or parquet, both of which outlive this library. A file written today is
expected to keep loading.

**Functions are only as durable as your code.** A node's function is stored as a
module path and a qualified name, so it resolves only while that module still
exists and still exports that name — rename it, move it, or uninstall the package
and the value still loads but the node can no longer be recalculated. A function
stored via `use_dill_for_functions=True` is worse: dill blobs are not portable
across Python versions.

So a saved computation is a durable record of *what was computed*, and a
best-effort record of *how*. For an archive you expect to read years later,
prefer named functions in a stable module, and consider that being able to
recalculate may not survive a refactor even when the numbers do.

Two further caveats that are not about time:

- The file is **not safe to load from an untrusted source** — see
  [Loading untrusted files](saving_computations.md#loading-untrusted-files).
- pandas 2 and pandas 3 differ in default datetime resolution. Each value's
  resolution is recorded, so a file written under one loads correctly under the
  other; loman requires pandas 2.0 or later precisely because that recording
  depends on APIs introduced there.

