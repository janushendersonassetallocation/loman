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
    features=[
        util.FanOut("positions", "data", transform=select_instrument),
        util.FanIn("value", "portfolio_values", combine=concat_values),
    ],
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

`built.blocks` maps each key to its generated block path, `built.nodes` lists
every generated node in declaration order, and `built.named` maps the names
features chose to label — a `FanIn` labels its result — to the nodes created for
them.

## Features

`features` is an ordered list describing how data flows in and out of each copy.
Four are built in:

| feature | what it creates |
| --- | --- |
| `FanOut(source, target, transform=None)` | one node per block, fed from outside |
| `FanIn(source, result, combine=None)` | one outer node gathering a node from every block |
| `IdNode(name)` | one node per block holding that block's key |
| `InputValue(name, value)` | one shared outer node, linked into every block |

`Positional(fn)` wraps an aggregator that takes its values positionally, so a
`combine` written before the keyed mapping existed does not need a lambda at
every call site: `FanIn("value", "total", combine=Positional(df_hconcat))`.
It discards the keys. A keyed aggregator can use them instead — for dataframes,
`lambda m: pd.concat(m, axis=1)` turns them into column labels — but that is a
different result, not a drop-in replacement: it adds an outer level to the column
index where a flat positional concatenation does not.

Features are planned in the order given, and a later feature may read a node an
earlier one created. Nothing is applied until all of them have been planned and
checked together, so a definition that fails validation leaves the computation
completely untouched — no partially built blocks.

### Which names are relative to `base_path`

This differs by feature, and it is worth knowing before you go looking for a node
that is not where you expected:

| name | resolves to |
| --- | --- |
| `FanOut.target`, `FanIn.source`, `IdNode.name` | inside each block, `<base_path>/<key>/<name>` |
| `InputValue.name` | beside the blocks, `<base_path>/<name>` |
| `FanOut.source` | the outer computation, **verbatim** |
| `FanIn.result` | the outer computation, **verbatim** |

So with `base_path="instruments"`, `FanIn("value", "portfolio_values")` creates a
top-level `portfolio_values`, not `instruments/portfolio_values`. That is
deliberate — an aggregate usually belongs wherever the rest of the model expects
it, rather than being forced under the blocks it happens to be computed from — but
it is the opposite of `InputValue`, whose shared node is placed under `base_path`
so that two definitions cannot collide. `built.named` always reports the key that
was actually created.

### Feeding a node the template does not declare

`FanOut.target` must be a name the block template declares or refers to. A name
it never mentions is rejected, because that is usually a typo which would
otherwise add a dead node to every block. Where it is deliberate — injecting a
node the template itself has no use for — pass `create=True`:

```python
util.FanOut("leverage_source", "leverage", create=True)
```

The guard that matters still applies: a fan-out will not replace a calculation,
with or without `create`.

`InputValue` takes the same `create` flag, and defaults the same way. `IdNode`
defaults to `create=True` instead, because creating the node is its whole job — a
template that never mentions the name is the ordinary case there, not a mistake.
The cost is that a misspelled `IdNode("labl")` adds a node nothing reads and
leaves the real `label` unfilled; `validate()` reports that as an uninitialized
input, but only once the graph is built. Pass `create=False` where the template
does declare the node, to have the misspelling rejected at definition time.

Note that the low-level `add_fan_out` is more permissive, because it takes target
node names directly and has no template to check them against. If you are
comparing the two, that is the difference.

### Templates built by a computation factory

A `@ComputationFactory` class makes a perfectly good template — build it, then
pass the computation:

```python
util.RepeatedBlocks(PositionsComputation(), keys, "positions", features=[...])
```

One thing to watch: a factory commonly gives inputs a default with
`input_node(value=2)`. Because `keep_values` is `False`, those defaults are **not**
carried into the copies, so the blocks stay uninitialized until the input is
supplied. Pass `keep_values=True` to keep them, or feed the input with an
`InputValue` or a `FanOut`.

## Reading a different node per key

The first example slices one shared `positions` node. When each block should
instead read from a *different node that already exists*, pass a callable as the
`FanOut` source. It is applied to each key to resolve that block's source node:

```python
comp.add_node("data/AAPL", value=aapl_frame)
comp.add_node("data/MSFT", value=msft_frame)

util.RepeatedBlocks(
    block=price_block,
    keys=("AAPL", "MSFT"),
    base_path="instruments",
    features=[util.FanOut(lambda key: f"data/{key}", "data")],
).add_to(comp)
```

`instruments/AAPL/data` now depends only on `data/AAPL`, so inserting a new value
for one instrument invalidates only that block.

Because a callable source is meaningful, a callable passed anywhere a plain node
name is expected raises `TypeError` rather than silently creating a node keyed by
the function object.

## Giving each block its own key

`IdNode` creates a node inside every block holding that block's key, so block
functions can depend on their own key by name:

```python
block = Computation()
block.add_node("label")
block.add_node("data")
block.add_node("summary", lambda label, data: f"{label}: {data.sum()}")

util.RepeatedBlocks(
    block=block,
    keys=("AAPL", "MSFT"),
    base_path="instruments",
    features=[
        util.IdNode("label"),
        util.FanOut("positions", "data", transform=select_instrument),
    ],
).add_to(comp)
```

`instruments/AAPL/label` holds `"AAPL"`. These nodes have no predecessors — each
simply holds its key as a value. The template does not have to declare the node;
if it does, it must be an input node.

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

When every copy does need the same value, use `InputValue`, which creates one
node beside the blocks and links it into each of them:

```python
util.RepeatedBlocks(
    block=price_block,
    keys=("AAPL", "MSFT"),
    base_path="instruments",
    features=[util.InputValue("scale", 100)],
).add_to(comp)
```

Every block now reads `scale` from `instruments/scale`, so changing it is a
single `comp.insert("instruments/scale", 10)` rather than an insert into each
generated copy. Use a `FanOut` with no `transform` to broadcast a node that
already exists elsewhere, and `keep_values=True` only when the copies really
should start from a snapshot of the template.

## Writing your own feature

A feature is any object with a `plan` method. It never changes the computation:
it describes the nodes it wants as `PlannedNode` values, and the builder
validates every feature's plan together before applying any of them.

```python
from loman import PlannedNode


class Doubled:
    """Add <block>/doubled = 2 * <block>/<source> to every block."""

    def __init__(self, source, name="doubled"):
        self.source = source
        self.name = name

    def plan(self, ctx):
        source = ctx.require_block_node(self.source, "Doubled source")
        for block_path in ctx.blocks.values():
            yield PlannedNode.calc(
                block_path.join(self.name),
                lambda value: value * 2,
                (block_path.join(source),),
            )
```

`ctx` is a `BlockContext`. It carries `blocks`, mapping each key to its block
path, along with the template and the destination computation, and offers
`require_block_node` and `require_block_input` so port checks raise the same
errors the built-in features do.

`PlannedNode` has three constructors: `input_node(key, value)` for a node holding
a value, `link(key, source)` for a node that copies another unchanged, and
`calc(key, func, args)` for a calculation, where each argument is either a node
key to depend on or a `C(...)` constant. Pass `label=` to have the node appear in
`built.named`.

## Low-level helpers

The features are built on independent utilities, which can also be used directly
for more dynamic graph construction.

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
as `source` resolves a source node per key instead of broadcasting one.

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

If `combine` is omitted, the keyed mapping itself becomes the result value. Note
that it receives the mapping, not the values, so `combine=sum` would add up the
keys — use `lambda values: sum(values.values())`.

## Repeated blocks in a computation factory

`repeated_blocks` declares the same structure inside a
[`@ComputationFactory`](creating_computation_factories.md) class, alongside
`input_node`, `calc_node` and `block`. The attribute name becomes the base path,
so the class below generates `instruments/AAPL` and `instruments/MSFT`:

```python
from loman import ComputationFactory, FanIn, FanOut, IdNode, calc_node, input_node, repeated_blocks


@ComputationFactory
class InstrumentBlock:
    label = input_node()
    data = input_node()

    @calc_node
    def value(self, label, data):
        return data.assign(instrument=label, value=data["quantity"] * data["price"])


@ComputationFactory
class Portfolio:
    positions = input_node()

    def select_instrument(self, positions, instrument_id):
        return positions.loc[[instrument_id]]

    instruments = repeated_blocks(
        InstrumentBlock,
        keys=("AAPL", "MSFT"),
        features=[
            IdNode("label"),
            FanOut("positions", "data", transform=select_instrument),
            FanIn("value", "portfolio_values", combine=concat_values),
        ],
    )

    @calc_node
    def total_value(self, portfolio_values):
        return portfolio_values["value"].sum()


comp = Portfolio()
```

The block may be a `Computation` or, as above, another computation factory,
matching `block`. `keep_values` is accepted too.

Nodes that features refer to can be declared anywhere in the class, regardless of
order: a node that is only referred to remains a placeholder until the member
that defines it is added. A name that another member *defines*, however, cannot
also be a fan-in result — declaring both `portfolio_values = input_node()` and a
`FanIn(..., "portfolio_values")` is an error.

Callbacks follow the same `self` convention as `calc_node`. `select_instrument`
above is declared with `self` as its first parameter and is bound to the
definition object, so it is called as `select_instrument(positions, key)` and can
use other methods and attributes of the class. This applies to a `FanOut` source
resolver and a `FanIn` combine function too. Callbacks that do not take `self` —
module-level functions, lambdas, or `staticmethod`s — are used unchanged. Pass
`ignore_self=False` to `@ComputationFactory` to disable binding for the whole
class.

## Serialization

The generated wiring nodes roundtrip through JSON. A fan-out transform and a
fan-in combine function are passed as constant arguments, and constants are
recorded in the serialized node, so a fan-out or fan-in node can be recalculated
after a reload. As with any node function, the callback must be importable —
a module-level function, or any callable when
`ComputationSerializer(use_dill_for_functions=True)` is used. A constant that
cannot be encoded raises `SerializationError` naming the node, rather than being
dropped and failing later.

What still does not survive is the block template's *own* calculations.
`Computation.add_block` deliberately sets `serialize=False` on the nodes it
copies, so their functions are not retained:

```python
block.add_node("doubled", double)  # calculated inside every generated block
```

After a roundtrip, `instruments/AAPL/doubled` keeps its stored value but has no
function, so it cannot be recalculated and anything downstream of it stays stale.
If your per-key work happens in the template — the normal case — serialize
computed values when a snapshot is enough, and rebuild the graph from its Python
definition when you need to recalculate.

Keys become real path parts, and JSON converts path parts to strings, so use
string keys when a serialized computation must preserve key types.
