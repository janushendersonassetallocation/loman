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
<States.PINNED: 6>
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
<States.ERROR: 5>
>>> comp.write_json('comp.json')
>>> comp2 = Computation.read_json('comp.json')
>>> comp2.state('result')
<States.ERROR: 5>
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

Columns are encoded individually, so each keeps its own dtype rather than being
flattened through `object`. Datetimes, timezone-aware datetimes, timedeltas,
categoricals, nullable extension dtypes such as `Int64`, and MultiIndexes all
round-trip as themselves:

```pycon
>>> df = pd.DataFrame({
...     't': pd.date_range('2024-01-01', periods=3, freq='D'),
...     'c': pd.Categorical(['a', 'b', 'a']),
...     'n': pd.array([1, None, 3], dtype='Int64'),
... })
>>> comp = Computation()
>>> comp.add_node('df', value=df)
>>> buf = io.StringIO()
>>> comp.write_json(buf)
>>> _ = buf.seek(0)
>>> Computation.read_json(buf).v.df.dtypes.to_dict() == df.dtypes.to_dict()
True
```

Bare temporal values work too — `datetime`, `date`, `time`, `timedelta` and
their pandas counterparts — whether they are a node's value or nested inside a
list or dict.

## Archives: `.loman` and `.lm`

A JSON document holds every value inline. That is ideal for a small graph and
increasingly wasteful for a large one — numbers written as text cost around
2.7x their in-memory size, and the whole file has to be parsed before any of it
can be read.

An **archive** is a zip holding the same graph structure in a readable
`manifest.json`, plus one entry per large value in a format suited to its type:
parquet for DataFrames and Series, `.npy` for arrays.

```pycon
>>> comp.write_archive('run.loman')
>>> comp2 = Computation.read_archive('run.loman')
```

`.loman` and `.lm` are the same format; use whichever you prefer. `write` and
`read` pick the format from the extension, so a `.loman` or `.lm` path gets an
archive and anything else gets JSON:

```pycon
>>> comp.write('run.lm')            # archive
>>> comp.write('run.json')          # JSON document
>>> comp2 = Computation.read('run.lm')
```

On realistic data — repeated strings, bounded-precision floats — an archive is
several times smaller than the equivalent JSON and considerably faster to read.
Purely random float64 is the worst case for any encoding and still comes out
ahead, just less dramatically.

### Bulky values are found at any depth

A frame does not have to be a node's whole value to get its own payload. Any
DataFrame, Series or array above the size threshold is stored out of line
wherever it appears — inside a dict, a list, a tuple, a dataclass, an attrs
object, or several levels down:

```pycon
>>> import numpy as np
>>> prices = pd.DataFrame({'px': np.arange(50000, dtype='float64')})
>>> weights = np.arange(50000, dtype='float64')
>>> comp = Computation()
>>> comp.add_node('bundle', value={'prices': prices, 'weights': weights})
>>> comp.write_archive('run.loman')
```

That writes two payloads and a manifest holding two small references, rather
than inlining both values as JSON. Each frame in a collection gets its own
entry, so a list of ten frames becomes ten payloads and can be read back
selectively.

Values below the threshold stay inline regardless of depth — a two-row frame is
not worth a zip entry. Set `inline_threshold=0` to force everything out of line,
or a large value to keep everything in.

### Reading part of a computation

Because each large value is a separate zip entry, an archive can be read
partially. Name the nodes you want and the rest are never decompressed:

```pycon
>>> comp2 = Computation.read_archive('run.loman', nodes=['summary'])
>>> comp2.v.summary
...
```

Every node, edge and function is still restored, so the graph keeps its shape
and can simply be recomputed. Nodes whose values were skipped come back
`UNINITIALIZED`, or `COMPUTABLE` where their own inputs happen to have been
loaded.

`read_json` accepts `nodes=` too, but a JSON document must be parsed in full
regardless, so there the saving is only in decoding — not in I/O.

### Inspecting an archive

An archive is an ordinary zip, so `unzip -l` works. To attribute a large file to
a particular node from Python:

```pycon
>>> from loman import ArchiveSerializer
>>> with open('run.loman', 'rb') as f:
...     print(ArchiveSerializer().payload_summary(f).to_string())
                  name    size  compressed
0        manifest.json     596         226
1  payloads/p0.parquet  217283      217283
2      payloads/p1.npy  160128      153754
```

Parquet entries show equal `size` and `compressed` because parquet compresses
internally; deflating them again inside the zip would cost time and save
nothing.

### Parquet is optional

Parquet payloads need [pyarrow](https://arrow.apache.org/docs/python/), which is
an optional dependency:

```bash
pip install 'loman[archive]'
```

Without it, archives still work — DataFrames fall back to JSON payloads, which
are larger but correct. Arrays use `.npy` either way, which needs only numpy.
The manifest always records which encoding each payload used, so an archive
written with parquet and read without pyarrow says exactly that rather than
failing somewhere deep in a decoder.

## Format compatibility

Every serialized computation carries an integer `version`. Loman's commitment
is:

- A reader accepts **every format version from 1 up to the one it writes**.
  Files written by an older loman keep loading.
- The version increments whenever the schema changes.
- A file from a *newer* loman is refused with a clear message rather than
  parsed on a best-effort basis — the fields a future version adds are exactly
  the ones whose absence would corrupt the result silently.
- Dropping support for an old version is a breaking change, reserved for a
  major release.

This is enforced, not merely intended: `tests/data/formats/vN/` holds a corpus
of files captured when each version was current, and CI reads all of them on
every commit.

!!! warning
    One exception: `use_dill_for_functions=True` embeds dill blobs, and those
    are not portable across Python versions whatever loman promises. The
    guarantee above covers the document structure and all built-in value
    encodings; it cannot cover a pickled function object.

## JSON format reference

The file is a single JSON object with three top-level keys:

```json
{
  "version": 2,
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
Compound types use a tagged object with a `"type"` discriminator.

Arrays and frames are encoded **column-wise**, dispatching on dtype. Each column
carries a `"kind"` saying how to read it, which is why a decoder never needs to
know which format version produced a value.

| `kind` | Used for | Shape |
|---|---|---|
| `plain` | int, float, bool | `{"dtype": "<f8", "data": [1.0, 2.0]}` |
| `datetime` | `datetime64`, optionally with `tz` | `{"dtype": "datetime64[ns]", "data": [1704067200000000000, null]}` |
| `timedelta` | `timedelta64` | as `datetime` |
| `category` | pandas Categorical | `{"categories": {...}, "ordered": false, "codes": [0, 1]}` |
| `masked` | nullable `Int64`, `boolean`, `string` | `{"dtype": "Int64", "data": [1, null]}` |
| `object` | strings, custom types | `{"dtype": "|O", "data": [...]}` |

`NaT` is written as JSON `null` rather than numpy's internal sentinel, so the
file stays legible and survives a trip through other tools.

**NumPy array**

```json
{
  "type": "ndarray",
  "shape": [3],
  "dtype": "<f8",
  "values": {"kind": "plain", "dtype": "<f8", "data": [1.0, 2.0, 3.0]}
}
```

**Pandas DataFrame**

```json
{
  "type": "dataframe",
  "columns": {"kind": "index", "name": null, "data": {...}},
  "index":   {"kind": "index", "name": null, "data": {...}},
  "cols": [
    {"kind": "plain", "dtype": "<i8", "data": [1, 2]},
    {"kind": "datetime", "dtype": "datetime64[ns]", "data": [1704067200000000000, null]}
  ]
}
```

Indexes use `kind: "index"`, or `kind: "multiindex"` with `levels` and `codes`.
A regular index carries its `freq`, which pandas treats as part of the index's
identity.

**Archive payload reference** (archives only — the value lives in a zip entry)

```json
{
  "__loman_payload__": true,
  "id": "p0",
  "encoding": "parquet",
  "kind": "dataframe"
}
```

A reference can appear anywhere a value can, at any depth — a node's value, an
element of a list, a value in a dict, a field of a dataclass. A JSON document
has nowhere to put the payloads, so it always inlines; only archives use
references.

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
    The format may gain fields between releases, but older files keep loading —
    see [Format compatibility](#format-compatibility) above for the guarantee
    and its one exception.

