# Repeated Blocks and Fan-In/Fan-Out

`loman.util` provides graph-building helpers for computations that repeat the
same block for a collection of keys. The helpers add ordinary Loman nodes and
dependencies: they do not read values or execute calculations while building
the graph.

## Defining repeated blocks

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

repeated = util.RepeatedBlocks(
    block=price_block,
    keys=("AAPL", "MSFT"),
    base_path="instruments",
    fan_out=(
        util.FanOut("positions", "data", transform=select_instrument),
    ),
    fan_in=(
        util.FanIn("value", "portfolio_values", combine=concat_values),
    ),
)
built = repeated.add_to(comp)
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

`built.blocks` maps each key to its generated block path, while
`built.results` maps each declared result name to its generated result node.

`RepeatedBlocks` accepts multiple `FanOut` and `FanIn` definitions, so a block
can consume several shared or keyed inputs and produce several aggregates. The
frozen dataclass can also be reused to add the same graph structure to multiple
computations.

## Repeated blocks in a computation factory

`repeated_blocks` declares the same structure inside a
[`@ComputationFactory`](creating_computation_factories.md) class, alongside
`input_node`, `calc_node` and `block`. The attribute name becomes the base path,
so the class below generates `instruments/AAPL` and `instruments/MSFT`:

```python
from loman import ComputationFactory, FanIn, FanOut, calc_node, input_node, repeated_blocks


@ComputationFactory
class InstrumentBlock:
    data = input_node()

    @calc_node
    def value(self, data):
        return data.assign(value=data["quantity"] * data["price"])


@ComputationFactory
class Portfolio:
    positions = input_node()

    def select_instrument(self, positions, instrument_id):
        return positions.loc[[instrument_id]]

    instruments = repeated_blocks(
        InstrumentBlock,
        keys=("AAPL", "MSFT"),
        fan_out=(FanOut("positions", "data", transform=select_instrument),),
        fan_in=(FanIn("value", "portfolio_values", combine=concat_values),),
    )

    @calc_node
    def total_value(self, portfolio_values):
        return portfolio_values["value"].sum()


comp = Portfolio()
```

The block may be a `Computation` or, as above, another computation factory,
matching `block`. `keep_values` is also accepted, and defaults to `False`.

`fan_out` sources, `fan_in` results and generated block nodes can be referred to
from anywhere in the class, regardless of the order in which class members are
declared: a node that is only referred to remains a placeholder until the member
that defines it is added. A name that another member *defines*, however, cannot
also be a fan-in result — declaring both `portfolio_values = input_node()` and a
`FanIn(..., "portfolio_values")` is an error.

Callbacks follow the same `self` convention as `calc_node`: `select_instrument`
above is declared with `self` as its first parameter and is bound to the
definition object, so it is called as `select_instrument(positions, key)` and can
use other methods of the class. Callbacks that do not take `self` — module-level
functions, lambdas, or `staticmethod`s — are used unchanged. Pass
`ignore_self=False` to `@ComputationFactory` to disable binding for the whole
class.

## Low-level helpers

The dataclass builder composes three independent utilities. They can also be
used directly for more dynamic graph construction.

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
