# Converters and Validation

A node can be given a `converter`: a callable that Loman applies to a value on its way
into the node. The node stores what the converter returns, not what was supplied.

```pycon
>>> comp = Computation()
>>> comp.add_node('a', converter=float)
>>> comp.insert('a', '3.5')
>>> comp.v.a
3.5
```

The string never reaches the node. `comp.v.a` is a `float`, so every downstream
calculation can rely on that without re-checking.

## Where the converter runs

The converter runs every time a value is set on the node, whichever way it arrives:

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1, converter=float)      # value passed to add_node
>>> comp.v.a
1.0

>>> comp.insert('a', 2, force=True)                   # insert
>>> comp.v.a
2.0
```

It also runs on values a node calculates for itself, which is easy to miss. Here the
lambda returns an `int`, but the node holds a `float`:

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: a + 1, converter=float)
>>> comp.compute_all()
>>> comp.v.b
2.0
```

So a converter is a property of the *node*, not of the input path. It normalizes
supplied values and computed values alike. `insert_many` applies converters too.

## Using a converter as a validator

Loman has no separate validator hook. A converter that checks its argument and raises
gives you the same thing, because a converter that raises puts the node into `ERROR`
state and never stores the value:

```pycon
>>> def positive(x):
...     if x <= 0:
...         raise ValueError(f"must be positive, got {x}")
...     return x

>>> comp = Computation()
>>> comp.add_node('size', converter=positive)
>>> comp.insert('size', -5)
Traceback (most recent call last):
    ...
ValueError: must be positive, got -5
>>> comp.s.size
<States.ERROR: 5>
```

Return the value unchanged when it passes. A validator that forgets to return
silently replaces the node's value with `None`.

Because bad values are rejected at the boundary rather than surfacing later, this is
worth doing on the inputs of a long computation, where a nonsensical value would
otherwise be discovered several nodes downstream.

The two jobs combine, which is often what you want in practice — coerce first, then
assert the result is usable:

```pycon
>>> def positive_float(x):
...     value = float(x)
...     if value <= 0:
...         raise ValueError(f"must be positive, got {value}")
...     return value

>>> comp = Computation()
>>> comp.add_node('notional', converter=positive_float)
>>> comp.insert('notional', '1000')
>>> comp.v.notional
1000.0
```

## How failures are reported

The node ends in `ERROR` state either way, with the exception available as
`comp.v.<node>.exception`. Whether the exception also propagates to your code depends
on how the value arrived:

| Value arrives via | Node state | Exception raised to caller |
| --- | --- | --- |
| `add_node(value=...)` | `ERROR` | Yes |
| `insert` / `insert_many` | `ERROR` | Yes |
| A calculation (`compute`, `compute_all`) | `ERROR` | No |

Insertion raises, because supplying a value is something your code just did and can
handle immediately. A conversion failure during computation is treated like any other
node failure: the node is marked `ERROR`, its descendants are left unable to compute,
and the run continues so that unrelated branches still make progress.

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: a - 10, converter=positive)
>>> comp.compute_all()          # does not raise
>>> comp.s.b
<States.ERROR: 5>
>>> comp.v.b.exception
ValueError('must be positive, got -9')
```

`comp.validate()` and `comp.plan()` will report a node left in `ERROR` by a failed
conversion, the same as any other failed node — see
[Validation and Planning](../../../notebooks/validation_and_planning.html).

## Things to know

- **Redefining a node drops its converter.** `add_node` sets the whole node
  definition, so calling it again without `converter=` leaves the node with no
  converter. Pass the converter each time you redefine the node.
- **Converters are not saved by `write_json`.** A converter is a live Python callable,
  and JSON serialization does not attempt to encode it: a computation loaded with
  `read_json` has no converters, and values inserted into it are stored unchanged.
  Re-apply converters after loading, or set them up in a
  [computation factory](creating_computation_factories.md) so that the graph is always
  constructed the same way. (The deprecated `write_dill`/`read_dill` pair does preserve
  them, since it pickles the callable.)
- **`add_block` preserves converters.** A node's converter comes along when its
  computation is added as a block, so a validated block template stays validated
  wherever it is used.
- **Keep converters cheap and free of side effects.** A converter runs on every set,
  including repeat inserts of the same node, and its return value is what everything
  downstream sees.
