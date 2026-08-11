# Design plan: typed and abstract computations

Status: proposed, not implemented. No code has been written on this branch.

Two things are being solved, and keeping them apart is the main decision here:

1. **Interfaces** — declaring that a computation *is a* Signal, checking it, and
   reading its nodes with real types. Defined over `Computation`, so it works with
   every existing computation and needs no migration.
2. **Typing the computation object** — `Signal()` should be a `Computation` to a
   type checker. Only inheritance can do this, so it is a separate, later layer
   that users opt into.

## Scope

**In scope**

- A declared contract: which nodes must exist, and whether each is an input or a
  calculation. Optionally, which nodes a calculation must depend on.
- Checking any `Computation` against a contract, with actionable reasons.
- Statically typed access to a conforming computation's node values.
- Declaring conformance at construction, for both declaration styles.
- `ComputationBase`, so a computation can be declared by inheriting.

**Out of scope**

- Runtime checking of node *value* types. The contract is about graph shape.
  Declared types drive the static facade; nothing asserts `isinstance` at compute
  time. Revisit only if asked for.
- Changing or deprecating `@ComputationFactory`. It keeps working, unchanged,
  and gains interfaces for free.
- Typing `comp.v` / `comp.s` and the other attribute views. String-keyed access
  is untypeable; the facade exists precisely to sidestep it.
- Generating interfaces from an existing computation.

## Decided, with evidence

These were measured, not reasoned about. They should not need relitigating.

**A class decorator cannot be typed.** `@ComputationFactory class Signal` gives a
symbol that checkers treat as the class, so every legitimate use of the result is
a false positive:

```text
error[unresolved-attribute]: Object of type `Signal` has no attribute `compute_all`
error[unresolved-attribute]: Object of type `Signal` has no attribute `insert`
error[unresolved-attribute]: Object of type `Signal` has no attribute `v`
```

Three separate escapes were tried and all fail: annotating the decorator as
returning `type[Computation]`, adding `@overload` to `computation_factory`, and
declaring `__new__` as returning `Computation`. Class-decorator return types are
ignored by design. **Do not spend more time here** — anything that must keep the
decorator working goes through the facade, not through the decorated name.

**The assignment form does type.** `Signal = computation_factory(Spec)` infers
`Computation`, because it is an ordinary call. Worth documenting as a workaround;
it discards the class identity, so it is not the destination.

**Contracts are construction-agnostic.** A prototype contract accepted
decorator-built, inheritance-built and imperatively-built computations
identically, and rejected a non-conforming one with reasons
(`"missing node 'df_signals'"`, `"'prices' should be an input"`).

**Inheritance works and is faithful.** `ComputationBase` prototyped against real
loman produced identical nodes and identical results to the decorator, with
`isinstance` working, declared nodes reachable as class attributes, name and
docstring preserved without `functools.wraps`, and `self`-binding in calc nodes
unaffected. Subclassing composes: a subclass inherits declared nodes, so an
abstract computation is just a base class.

### Two mechanism constraints

Both were found by testing and will silently degrade everything if missed.

- **`view()` must return `Self`, never `Any`.** With `Any`, the gradual type
  flows through the annotated variable and every node access degrades to `Any`
  with no error anywhere. Verified: with `Self`, `sig.df_signals` types as the
  declared type; with `Any`, it types as `Any`.
- **`required_input()` and `required_calc()` must be annotated `-> Any`,** the way
  `dataclasses.field()` is. That lets `df_signals: DataFrame = required_calc()`
  assign without complaint while the annotation governs attribute access.

## API

```python
class Signal(ComputationInterface):
    """Any computation that behaves as a signal."""

    prices: pd.DataFrame = required_input()
    df_signals: pd.DataFrame = required_calc(depends_on=["prices"])
```

One declaration serves three jobs — specification, checking, and typed access.

```python
Signal.check(comp)      # -> report; reasons it does not conform, empty if it does
Signal.view(comp)       # -> Signal; refuses a non-conforming computation
```

Verified behaviour of the facade under `ty`:

| expression | result |
| --- | --- |
| `sig = Signal.view(comp)` | `Signal` |
| `sig.df_signals` | `pd.DataFrame` |
| `sig.df_signals.not_a_frame_method()` | error |
| `sig.not_declared` | error |
| `Signal.check(comp)` | `list[str]` |

This is what a decorator user gets, with no migration:

```python
@ComputationFactory
class MySignal:
    prices = input_node()

    @calc_node
    def df_signals(self, prices): ...


sig = Signal.view(MySignal())
```

### Declaring conformance at construction

```python
@ComputationFactory(implements=Signal)
class MySignal: ...


class MySignal(ComputationBase, implements=Signal): ...
```

Both are runtime checks that fail where the computation is defined rather than at
the call site that needed it, reporting the contract's reasons. Neither changes
the decorator's static type; that is settled above.

### `ComputationBase`

```python
class ComputationBase(Computation):
    """A Computation whose nodes are declared as class members."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        populate_computation_from_class(self, type(self), self, self._ignore_self)
```

`_ignore_self` is a class attribute, mirroring
`@ComputationFactory(ignore_self=False)`.

## Phases

Each is independently reviewable and leaves the branch green.

**Phase 1 — contracts.** `ComputationInterface`, `required_input`,
`required_calc`, and `check()`.
*Acceptance:* the same contract checks a decorator-built, inheritance-built and
imperatively-built computation identically; a non-conforming computation reports
every reason, not just the first; the report follows the `ValidationReport` idiom
including `to_df()`. *Depends on:* nothing. *Size:* medium.

**Phase 2 — the typed facade.** `view()`, checking conformance before returning.
*Acceptance:* `sig.df_signals` types as the declared type under `ty`; an
undeclared attribute is an error; `view` on a non-conforming computation raises
with the reasons. Includes the typing fixture described below. *Depends on:*
phase 1. *Size:* small.

**Phase 3 — `ComputationBase`.** *Acceptance:* for one model expressed both ways,
generated nodes, edges and computed values match exactly; `self`-binding works;
a subclass inherits declared nodes; `isinstance` holds. *Depends on:* nothing.
*Size:* medium.

**Phase 4 — `copy()` preserves the subclass.** *Acceptance:* `Signal().copy()` is
a `Signal`, values intact. *Depends on:* phase 3. *Size:* small.

**Phase 5 — `implements=`.** Both declaration styles.
*Acceptance:* a non-conforming implementation fails at construction with the
contract's reasons. *Depends on:* phases 1 and 3. *Size:* small.

**Phase 6 — documentation.** Interfaces with both declaration styles, which is
recommended for new code, and the caveats below stated plainly. *Depends on:* the
rest. *Size:* small.

Phases 1 and 2 deliver most of the value and are not blocked by anything. If the
work is cut short, stopping after phase 2 leaves something coherent and useful.

## Testing approach

Two kinds, because the suite cannot assert what a checker infers.

**Behaviour**, in the normal suite: contract checking across all three
construction styles; every failure mode reporting all its reasons; equivalence
between the decorator and `ComputationBase` on the same model.

**Typing**, as a fixture module of intentionally correct and intentionally wrong
usages, checked in CI by the existing `ty` step, with the wrong ones expected to
produce diagnostics. Without this the typing guarantee is untested and will rot —
and given the whole point of this work is static types, an untested typing
guarantee is the failure mode to avoid.

## Caveats

- **`copy()` loses the subclass** until phase 4: it returns a plain
  `Computation`, so a typed `Signal` silently becomes untyped. Values survive.
- **Deserialization loses the subclass**, unavoidably — the deserializer cannot
  know which class built the graph. `check()` is the way to re-establish an
  interface after a roundtrip, which works because contracts are defined over
  `Computation`.
- **Two ways to declare a computation** will coexist. Splitting interfaces out
  softens this a lot: the answer to "which should I use" becomes "either, and
  interfaces work with both". The docs should still name one for new code.

## Fix while this is open

Independent of the direction chosen: `functools.wraps(cls, updated=())` copies
the **class body's** annotations onto the factory function, so a class declaring
`prices: str = input_node()` produces a factory whose `__annotations__` claim it
takes a `prices: str` parameter. It does not. Under inheritance the question
disappears; if the decorator is kept as-is, annotations should be set rather than
inherited.

## Open questions

- **`check()`'s return type.** A `list[str]` is easy to consume; a report object
  matching `ValidationReport` is consistent with the rest of loman and gets
  `to_df()` for free. Leaning to the report; the phase-1 acceptance criteria
  assume it.
- **Does `view()` snapshot or stay live?** A facade holding the computation reads
  current values on each access, which is probably what people expect, but it
  means a `Signal` can stop conforming after it was handed out.
- **Interaction with repeated blocks.** Adding a `ComputationBase` subclass as a
  block works; using one as a *repeated-block template* is untested, because
  those utilities were on an unmerged branch when this was written. Check before
  phase 3.
- **Metaclass conflicts.** A computation class that already has a metaclass would
  now meet `Computation`'s. Unlikely, but a hard error when it happens.
- **Should interfaces compose?** `class Tradeable(Signal, Priced)` is natural to
  reach for and needs a rule for merging contracts.

## Evidence appendix

Six shapes measured under `ty`, each with a deliberate nonsense call to confirm
the checker was really looking.

| shape | inferred type | legitimate calls | nonsense caught |
| --- | --- | --- | --- |
| class decorator returning a factory function (today) | the class | **all error** | yes, for the wrong reason |
| the same decorator applied by assignment | `Computation` | pass | yes |
| class decorator returning `type[Computation]` | the class | error | — |
| a class declaring `__new__` returning `Computation` | the class | error | — |
| inheritance | the subclass | pass | yes |
| typed facade over a computation | the facade | pass | yes, and node values are typed |
