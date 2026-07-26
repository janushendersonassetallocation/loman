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

## Reading a different node per key

The example above slices one shared `positions` node. When each block should
instead read from a *different node that already exists*, pass a callable as the
`FanOut` source. It is applied to each key to resolve that block's source node:

```python
comp.add_node("data/AAPL", value=aapl_frame)
comp.add_node("data/MSFT", value=msft_frame)

util.RepeatedBlocks(
    block=price_block,
    keys=("AAPL", "MSFT"),
    base_path="instruments",
    fan_out=(util.FanOut(lambda key: f"data/{key}", "data"),),
).add_to(comp)
```

`instruments/AAPL/data` now depends only on `data/AAPL`, so inserting a new value
for one instrument invalidates only that block. A `transform` can be combined
with a per-key source, in which case it receives whatever that key resolved to.

Because a callable source is meaningful, a callable passed anywhere a plain node
name is expected — a fan-out target, a fan-in source or result — raises
`TypeError` rather than silently creating a node keyed by the function object.

## Giving each block its own key

`id_node` names a node created inside every block that holds that block's key, so
block functions can depend on their own key by name:

```python
block = Computation()
block.add_node("label")
block.add_node("data")
block.add_node("summary", lambda label, data: f"{label}: {data.sum()}")

built = util.RepeatedBlocks(
    block=block,
    keys=("AAPL", "MSFT"),
    base_path="instruments",
    fan_out=(util.FanOut("positions", "data", transform=select_instrument),),
    id_node="label",
).add_to(comp)
```

`instruments/AAPL/label` holds `"AAPL"`. These nodes have no predecessors — each
simply holds its key as a value — and `built.id_nodes` maps each key to its
generated identifier node. The template does not have to declare the node; if it
does, it must be an input node, and it cannot also be a fan-out target.

This is the natural way to let a block look data up by its own key, or branch on
it, without threading the key in from outside.

## Sharing a value across every block

Blocks are copied structure-first: `keep_values` defaults to `False`, so the
values currently held by the template are not carried into the generated copies.
This is the opposite of `Computation.add_block`, which defaults to `True`. The
defaults differ because the two calls do different jobs. `add_block` adds one
specific block, often a sub-model that has already been populated or calibrated
and would not compute without its values. The repeated-block utilities stamp out
many copies of one template, where whatever the template happened to hold when it
was last run is rarely what all of the copies should start from.

When every copy does need the same value, broadcast it with a `FanOut` that has
no `transform` rather than reaching for `keep_values=True`:

```python
comp.add_node("scale", value=100)

util.RepeatedBlocks(
    block=price_block,
    keys=("AAPL", "MSFT"),
    base_path="instruments",
    fan_out=(
        util.FanOut("positions", "data", transform=select_instrument),
        util.FanOut("scale", "scale"),
    ),
).add_to(comp)
```

Every block now reads `scale` from one outer node, so changing it is a single
`comp.insert("scale", 10)` rather than an insert into each generated copy. Use
`keep_values=True` when the copies really should start from a snapshot of the
template.

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
matching `block`. `id_node` and `keep_values` are accepted too, so a class can
declare per-key identifier nodes and per-key sources exactly as the dataclass
does:

```python
@ComputationFactory
class Book:
    prefix = "data"

    def price_source(self, label):
        return f"{self.prefix}/{label}"

    commodities = repeated_blocks(
        Commodity,
        keys=("CL", "GC"),
        fan_out=(FanOut(price_source, "price_series"),),
        fan_in=(FanIn("nav", "all_navs"),),
        id_node="label",
    )
```

`fan_out` sources, `fan_in` results and generated block nodes can be referred to
from anywhere in the class, regardless of the order in which class members are
declared: a node that is only referred to remains a placeholder until the member
that defines it is added. A name that another member *defines*, however, cannot
also be a fan-in result — declaring both `portfolio_values = input_node()` and a
`FanIn(..., "portfolio_values")` is an error.

Callbacks follow the same `self` convention as `calc_node`: `select_instrument`
and `price_source` above are declared with `self` as their first parameter and are
bound to the definition object, so they are called as
`select_instrument(positions, key)` and `price_source(key)` and can use other
methods and attributes of the class. Callbacks that do not take `self` —
module-level functions, lambdas, or `staticmethod`s — are used unchanged. Pass
`ignore_self=False` to `@ComputationFactory` to disable binding for the whole
class.

## Low-level helpers

The dataclass builder composes four independent utilities. They can also be
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
each target is calculated as `transform(source_value, key)`. Passing a callable
as `source` resolves a source node per key instead of broadcasting one:

```python
util.add_fan_out(
    comp,
    source=lambda key: f"data/{key}",
    targets={key: path / "data" for key, path in blocks.items()},
)
```

### Identifier nodes

```python
util.add_id_nodes(comp, blocks, "label")
```

Adds one node per block holding that block's key, and returns a mapping from each
key to the generated node.

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
