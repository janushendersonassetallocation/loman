# Repeated Blocks and Fan-In/Fan-Out

`loman.util` provides graph-building helpers for computations that repeat the
same block for a collection of keys. The helpers add ordinary Loman nodes and
dependencies: they do not read values or execute calculations while building
the graph.

## Building a repeated pipeline

Suppose one input dataframe contains data for several instruments and the same
calculation should run independently for each instrument:

```python
import pandas as pd

from loman import Computation, util


price_block = Computation()
price_block.add_node("data")
price_block.add_node("value", lambda data: data.assign(value=data["quantity"] * data["price"]))


def select_instrument(frame, instrument_id):
    return frame.loc[[instrument_id]]


def concat_values(values):
    return pd.concat(values, names=["instrument_id"])


comp = Computation()
comp.add_node("positions")

pipeline = util.add_repeated_pipeline(
    comp,
    price_block,
    ["AAPL", "MSFT"],
    base_path="instruments",
    source="positions",
    block_input="data",
    block_output="value",
    result="portfolio_values",
    transform=select_instrument,
    combine=concat_values,
)
```

This creates two block paths, `instruments/AAPL` and `instruments/MSFT`.
`select_instrument(value, key)` runs separately for each block input, and
`concat_values(values)` receives the ordered mapping:

```python
{
    "AAPL": comp.v["instruments/AAPL/value"],
    "MSFT": comp.v["instruments/MSFT/value"],
}
```

The callbacks run only when their nodes are computed. Existing Loman behavior
therefore remains intact: unneeded branches are lazy, changed inputs invalidate
their descendants, and independent blocks can use configured executors.

For process executors, `transform` and `combine` must be pickleable, just like
ordinary node functions. The generated adapter nodes use the computation's
default executor.

## Composing the helpers

The pipeline helper combines three independent utilities. They can also be used
separately for more complex graphs.

### Repeated blocks

```python
blocks = util.add_repeated_blocks(
    comp,
    price_block,
    ["AAPL", "MSFT"],
    base_path="instruments",
)
```

The return value maps each original key to its generated `NodeKey`. Keys become
real path parts, so non-string identifiers are supported in memory. JSON
serialization currently converts node path parts to strings, so use string keys
when serialized computations must preserve key types. Values from the block
template are not copied by default; pass `keep_values=True` when the repeated
instances should retain them.

### Fan-out

```python
util.add_fan_out(
    comp,
    source="positions",
    targets={key: path / "data" for key, path in blocks.items()},
    transform=select_instrument,
)
```

With no `transform`, the source value is broadcast unchanged. With a transform,
each target is calculated as `transform(source_value, key)`.

### Fan-in

```python
util.add_fan_in(
    comp,
    result="portfolio_values",
    sources={key: path / "value" for key, path in blocks.items()},
    combine=concat_values,
)
```

The combine function receives an insertion-ordered mapping from keys to source
values. This supports dataframe concatenation as well as scalar reductions:

```python
util.add_fan_in(
    comp,
    result="total_value",
    sources={key: path / "value" for key, path in blocks.items()},
    combine=lambda values: sum(values.values()),
)
```

If `combine` is omitted, the keyed mapping itself becomes the result value.

## Serialization

These utilities build on `Computation.add_block`, whose calculation functions
are not retained by the default JSON serializer. Constant callback arguments
used by fan-in and transformed fan-out nodes are also not retained currently.
Serialize computed values when a snapshot is sufficient; rebuild the utility
graph from its definition before recalculating it after a JSON roundtrip.
