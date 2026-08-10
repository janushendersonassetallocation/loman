# Design plan: typed and abstract computations

Status: proposed, not implemented. This document is the design for review; no code
has been written on this branch.

## Recommendation

Two independent things are being asked for, and separating them is the main
decision in this document.

**The interface — "this computation is a Signal" — should be defined over
`Computation`, not over how the computation was built.** A contract is a
statement about a graph: these nodes exist, this one is an input, that one is a
calculation. Nothing about that depends on whether the graph came from a
decorator, from inheritance, or from `add_node` calls in a loop. Defining it that
way means the existing `@ComputationFactory` gets interfaces on day one, with no
migration:

```python
class Signal(ComputationInterface):
    prices = required_input()
    df_signals = required_calc()

Signal.check(comp)      # -> report, over any Computation
sig = Signal.view(comp) # -> typed facade; sig.df_signals is statically typed
```

This was prototyped against real loman and the same contract accepted
decorator-built, inheritance-built and imperatively-built computations
identically, rejecting a non-conforming one with reasons
(`"missing node 'df_signals'"`, `"'prices' should be an input"`).

**Separately**, `loman.ComputationBase` lets a computation be declared by
inheriting rather than decorating, which is the only way to give the computation
*object itself* a usable static type:

```python
class Signal(ComputationBase):
    prices = input_node()

    @calc_node
    def df_signals(self, prices):
        return prices * 2
```

`@ComputationFactory` keeps working unchanged; both are additive.

The reason the second one is needed at all is narrow and measurable. A class
decorator that returns something other than the class **cannot be typed**, in any
current checker, by design. The consequence today is that every legitimate use of
a factory-built computation is a false positive:

```text
error[unresolved-attribute]: Object of type `Signal` has no attribute `compute_all`
error[unresolved-attribute]: Object of type `Signal` has no attribute `insert`
error[unresolved-attribute]: Object of type `Signal` has no attribute `v`
```

Inheritance produces none of those, while still catching real mistakes, because
the declared class genuinely *is* the type of the object you get back.

The order matters for delivery: the interface layer is worth more, is not
blocked on anything, and helps every existing user. `ComputationBase` is worth
having, but only decorator users who also want the computation object typed have
to adopt it.

## Evidence

Six shapes were measured under `ty`, each with a deliberate nonsense call to
confirm the checker was really looking.

| shape | inferred type | legitimate calls | nonsense caught |
| --- | --- | --- | --- |
| class decorator returning a factory function (today) | the class | **all error** | yes, for the wrong reason |
| the same decorator applied by assignment | `Computation` | pass | yes |
| class decorator returning `type[Computation]` | the class | error | — |
| a class declaring `__new__` returning `Computation` | the class | error | — |
| **inheritance** | the subclass | pass | yes |
| **typed facade over a computation** | the facade | pass | yes, and node values are typed |

Three findings matter more than the table.

**Class-decorator return types are ignored regardless of what they are.** The
third row is the control: annotating the decorator as returning
`type[Computation]` changes nothing. This is long-standing, deliberate behaviour,
so no annotation, overload or stub can fix the current shape. An earlier
suggestion to add `@overload` to `computation_factory` was tested and does not
work.

**Nor does `__new__`.** The fourth row is the last mechanism that could have
rescued the decorator form: a class whose `__new__` is declared to return
`Computation`. `ty` ignores that too. There is no way to make the decorated
symbol itself type as a computation, so anything that needs the decorator to keep
working must go through the facade rather than through the decorated name.

**The facade works over anything.** It is an ordinary class, so its attributes
type normally, and it takes a `Computation` rather than caring where the
computation came from. That is what lets the interface layer serve decorator
users without asking them to change how they build computations.

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

The first two layers work over any `Computation`, whatever built it. Only the
third asks a user to change how they declare one.

### Layer 1: structural contracts

```python
class Signal(ComputationInterface):
    prices = required_input()
    df_signals = required_calc()

Signal.check(comp)      # -> report, in the idiom of validate() and to_df()
```

Two independent kinds of assertion — node presence and kind, and dependency shape
— usable separately or together, because an interface that pins wiring is
sometimes what you want and usually is not:

```python
class Signal(ComputationInterface):
    df_signals = required_calc(depends_on=["prices"])   # opt-in, not the default
```

The prototype's report is a list of reasons rather than a bare boolean, which is
what makes it useful at a boundary: `"missing node 'df_signals'"`,
`"'prices' should be an input"`.

### Layer 2: typed node access

```python
sig = Signal.view(comp)     # facade; sig.df_signals is statically a DataFrame
```

Only the facade can type node values, since attribute access on a real class is
checkable where string-keyed access is not. `view` checks conformance first and
refuses a computation that does not satisfy the contract, so a typed handle
cannot be obtained for a graph that would not support it.

Together these two layers give a decorator user everything except a static type
for the computation object:

```python
@ComputationFactory
class MySignal:
    prices = input_node()

    @calc_node
    def df_signals(self, prices): ...


sig = Signal.view(MySignal())   # typed, conformance-checked, no migration
```

### Layer 3: `ComputationBase`

```python
class ComputationBase(Computation):
    """A Computation whose nodes are declared as class members."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        populate_computation_from_class(self, type(self), self, self._ignore_self)
```

`_ignore_self` stays a class attribute so the per-class opt-out that
`@ComputationFactory(ignore_self=False)` provides has an equivalent.

This is the layer that types the computation object itself, and the one that
requires adopting a new declaration style. Worth having, and worth being last.

### Declaring conformance at construction

A factory can name the interfaces it claims, so a broken implementation fails
where it is defined rather than at the call site that needed it:

```python
@ComputationFactory(implements=Signal)
class MySignal: ...


class MySignal(ComputationBase, implements=Signal): ...
```

Both are runtime checks — the static type of the decorator form is unaffected,
for the reasons in the evidence section. The value is that the error arrives at
construction with the contract's reasons attached.

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
  for the structural `check()` in layer 1 as the way to re-establish an interface
  after a roundtrip — which works precisely because the contract is defined over
  `Computation` rather than over the class that built it.
- **Two ways to define a computation.** The decorator and the base class would
  coexist indefinitely. That is a genuine cost in documentation and in answering
  "which should I use". Splitting the interface layer out of the base class
  softens it considerably: the answer becomes "either, and interfaces work with
  both", rather than "switch". The docs should still say which is recommended for
  new code.

## Delivery phases

Each phase leaves the branch green and is independently reviewable.

- **Phase 1 — structural contracts.** `ComputationInterface` and `check()`,
  returning a report in the existing `ValidationReport` idiom, with node presence
  and kind. Dependency shape as an opt-in second assertion. Works over any
  `Computation`, so it lands value for existing `@ComputationFactory` users
  without asking them to change anything, and it does not depend on any of the
  phases below.
- **Phase 2 — the typed facade.** `view()`, checking conformance before handing
  back a typed handle. This is what gives decorator users typed node access.
- **Phase 3 — `ComputationBase`.** The class, tests covering equivalence with the
  decorator, `self`-binding, and subclass composition. Docs page and a changelog
  entry. No change to existing behaviour.
- **Phase 4 — `copy()` preserves the subclass**, plus a test. Small and separable,
  and only meaningful once phase 3 exists.
- **Phase 5 — conformance at construction.** `implements=` on both the decorator
  and the base class, as a runtime check reporting the contract's reasons.
- **Phase 6 — documentation.** Show interfaces working with both declaration
  styles, say which is recommended for new code, and state the copy and
  serialization caveats plainly.

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
