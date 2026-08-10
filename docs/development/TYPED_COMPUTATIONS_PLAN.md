# Design plan: typed and abstract computations

Status: proposed, not implemented. This document is the design for review; no code
has been written on this branch.

## Recommendation

Add `loman.ComputationBase`, a `Computation` subclass that populates itself from
its own class body, so a computation is **declared by inheritance rather than by
decoration**:

```python
class Signal(ComputationBase):
    prices = input_node()

    @calc_node
    def df_signals(self, prices):
        return prices * 2
```

`@ComputationFactory` keeps working unchanged; this is additive.

The reason is narrow and measurable. A class decorator that returns something
other than the class **cannot be typed**, in any current checker, by design. The
consequence today is that every legitimate use of a factory-built computation is
a false positive:

```text
error[unresolved-attribute]: Object of type `Signal` has no attribute `compute_all`
error[unresolved-attribute]: Object of type `Signal` has no attribute `insert`
error[unresolved-attribute]: Object of type `Signal` has no attribute `v`
```

Inheritance produces none of those, while still catching real mistakes, because
the declared class genuinely *is* the type of the object you get back.

As a second-order effect it also gives abstract computations essentially for
free, which is the other thing being asked for.

## Evidence

Five shapes were measured under `ty`, each with a deliberate nonsense call to
confirm the checker was really looking.

| shape | inferred type | legitimate calls | nonsense caught |
| --- | --- | --- | --- |
| class decorator returning a factory function (today) | the class | **all error** | yes, for the wrong reason |
| the same decorator applied by assignment | `Computation` | pass | yes |
| class decorator returning `type[Computation]` | the class | error | — |
| **inheritance** | the subclass | pass | yes |
| typed facade over a computation | the facade | pass | yes, and node values are typed |

Two findings matter more than the table.

**Class-decorator return types are ignored regardless of what they are.** The
third row is the control: annotating the decorator as returning
`type[Computation]` changes nothing. This is long-standing, deliberate behaviour,
so no annotation, overload or stub can fix the current shape. An earlier
suggestion to add `@overload` to `computation_factory` was tested and does not
work.

**The assignment form does type correctly.** `Signal = computation_factory(Spec)`
infers `Computation`, because it is an ordinary call rather than a class
decorator. It is worth documenting as an immediate workaround for anyone blocked
today, but it discards the class identity, so it is a stopgap and not the
destination.

## Prototype results

`ComputationBase` was prototyped against real loman and compared with the
existing decorator on the same model.

- **Runtime is identical.** Same generated nodes, same computed result.
- **Typing is correct.** `compute_all`, `insert` and `v` all resolve; a nonsense
  attribute is still an error.
- **`isinstance(s, Computation)` is true**, which the decorator cannot offer.
- **Declared nodes are reachable as class attributes.** `Signal.prices` is an
  `InputNode`. Consumers currently hand-roll this by walking the class and
  calling `setattr` on the returned function.
- **Name and docstring are preserved natively**, with no `functools.wraps`. That
  also removes an existing bug, described below.
- **`self`-binding in calc nodes still works**, so helper methods are unaffected.

### Abstract computations fall out of it

This ran as-is in the prototype:

```python
class AbstractSignal(ComputationBase):
    prices = input_node()

class ConcreteSignal(AbstractSignal):
    @calc_node
    def df_signals(self, prices):
        return prices * 3
```

Both nodes are present on the instance, `isinstance(c, AbstractSignal)` is true,
and `def run(sig: AbstractSignal)` type-checks at the call site. Subclassing *is*
the declaration, and it is enforced at construction because construction is what
populates the graph.

What inheritance does **not** cover, and still needs machinery:

- **Structural checking of a computation built some other way** — imperatively,
  or by repeated blocks. An interface should be checkable against any
  `Computation`, not only against one that inherited from it.
- **Dependency-shape assertions** — "this node must depend on that one" — which
  no amount of class structure expresses.
- **Typed node values.** `sig.df_signals` is still `Any`, because values are
  reached through `comp.v`, which is an `AttributeView`. Typing the *value* needs
  the facade shape from the last row of the table, layered on top.

## Proposed API

### Layer 1: `ComputationBase`

```python
class ComputationBase(Computation):
    """A Computation whose nodes are declared as class members."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        populate_computation_from_class(self, type(self), self, self._ignore_self)
```

`_ignore_self` stays a class attribute so the per-class opt-out that
`@ComputationFactory(ignore_self=False)` provides has an equivalent.

### Layer 2: structural contracts

For computations not built by inheritance:

```python
Signal.check(comp)          # -> report, in the idiom of validate() and to_df()
Signal.requires(depends_on={"df_signals": ["prices"]})   # dependency shape, opt-in
```

Two independent checks — node presence and kind, and dependency shape — usable
separately or together, because an interface that pins wiring is sometimes what
you want and usually is not.

### Layer 3: typed node access

```python
sig = Signal.view(comp)     # facade; sig.df_signals is statically a DataFrame
```

Only the facade can type node values, since attribute access on a real class is
checkable where string-keyed access is not. Worth designing now, but shippable
last.

## What this makes redundant downstream

Consumers currently patch the decorator's output to restore what it loses:
copying `__name__`, `__qualname__`, `__module__` and `__wrapped__`, setting a
return annotation, and walking the class to re-attach node members. Under
inheritance all of that is native and the shim can be deleted.

There is also a real bug to fix while this is open, independent of the direction
chosen. `functools.wraps(cls, updated=())` copies the **class body's**
annotations onto the factory function, so a class declaring `prices: str = input_node()`
produces a factory whose `__annotations__` claim it takes a `prices: str`
parameter. It does not. Under inheritance the question disappears; if the
decorator is kept as-is, the annotations should be set rather than inherited.

## Caveats

- **`copy()` loses the subclass.** It returns a plain `Computation`, so a typed
  `Signal` becomes untyped after copying. Values survive. Fixable, and it should
  be, since silently downgrading the type is worse than not having it.
- **Deserialization loses the subclass.** Unavoidable: the deserializer has no
  way to know which class produced the graph. Must be documented, and it argues
  for the structural `check()` in layer 2 as the way to re-establish an interface
  after a roundtrip.
- **Two ways to define a computation.** The decorator and the base class would
  coexist indefinitely. That is a genuine cost in documentation and in answering
  "which should I use". The docs should recommend one — the base class — and
  explain that the decorator remains supported.

## Delivery phases

Each phase leaves the branch green and is independently reviewable.

- **Phase 1 — `ComputationBase`.** The class, tests covering equivalence with the
  decorator, `self`-binding, and subclass composition. Docs page and a changelog
  entry. No change to existing behaviour.
- **Phase 2 — `copy()` preserves the subclass**, plus a test. Small and separable.
- **Phase 3 — structural contracts.** `check()` returning a report in the
  existing `ValidationReport` idiom, with node presence and kind. Dependency
  shape as an opt-in second check.
- **Phase 4 — typed node access.** The facade, and whatever generation or
  declaration it needs to stay in step with the interface.
- **Phase 5 — documentation.** Recommend the base class, document the decorator
  as supported, and state the copy and serialization caveats plainly.

## Testing approach

Equivalence is the main assertion: for the same model expressed both ways, the
generated node set, the edges and the computed values must match. That is cheap
to write and is what protects the decorator from regressing while the base class
is added.

Typing itself needs a different kind of test, since the suite cannot assert what
a checker infers. The practical option is a small fixture module of intentionally
correct and intentionally wrong usages, checked in CI by the existing `ty` step,
with the wrong ones expected to produce diagnostics. Without that, the typing
guarantee is untested and will rot.

## Risks and open questions

- **Does the base class interact with `add_block` and repeated blocks?** Adding a
  subclass instance as a block worked in the prototype. Using one as a repeated
  block template was not tested, because those utilities are on an unmerged
  branch. Needs checking before phase 1 lands.
- **Metaclass conflicts.** Anyone whose computation class already has a metaclass
  would now hit `Computation`'s. Unlikely, but it is a hard error when it
  happens.
- **`__init__` signature.** `ComputationBase.__init__` must pass executor
  arguments through to `Computation.__init__` while also populating. Subclasses
  that define their own `__init__` need a documented rule about calling
  `super().__init__()`.
- **Is one recommended way worth the churn?** Inheritance is better typed and
  strictly more capable, but the decorator is what every existing example and
  document uses. Migrating the documentation is most of the work and none of the
  risk, and it should be a deliberate decision rather than a side effect.
