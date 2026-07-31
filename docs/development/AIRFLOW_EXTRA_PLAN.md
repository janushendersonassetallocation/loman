# Design plan: `loman[airflow]`

Status: proposed, not implemented. This document is the design for review; no code
has been written on this branch.

## Recommendation

Build `loman.airflow`: a rule-based partitioner that cuts a loman `Computation`
into a small number of coarse stages, plus a thin emitter that renders those
stages as Airflow 3 tasks and task groups.

The decision that makes this tractable is that **a loman node is not an Airflow
task**. The unit of Airflow scheduling is a *stage*: a rule-selected set of nodes
that runs in one worker process using loman's normal in-process engine. Values
inside a stage never leave the process. Only values on edges that *cross* a stage
boundary go through a pluggable value channel.

The mental model to give users: **the Airflow DAG is the collapsed view of the
loman graph** — the same thing `comp.draw(collapse_all=True)` already renders.
That framing keeps the serialization problem to a handful of deliberately chosen
boundary values rather than every intermediate DataFrame, and it lets most of the
new code be pure Python that never imports Airflow.

## Concept mapping

| loman | Airflow 3 | Notes |
| --- | --- | --- |
| `Computation` via a zero-arg factory | `DAG` | Factory, not instance — see below |
| stage (rule-selected node set) | one `@task` | The scheduling unit; a new concept |
| loman node | nothing — a node inside a stage | Node-per-task is opt-in via `per="node"` |
| block, a `NodeKey` path prefix | `TaskGroup` when a block splits into several stages | `a/b/c` becomes `a.b.c` |
| edge crossing a stage boundary | task dependency plus one channel handle | The only values serialized |
| edge within a stage | nothing | Stays a Python object in memory |
| node-held constants (`C`/`ConstantValue`) | baked into the stage at build time | Never need a channel |
| `NodeAttributes.EXECUTOR` and `executor_map` | `queue` and `pool`, via an explicit rule | Not automatic: `executor_map` holds live `Executor` objects |
| `default_executor` | intra-task parallelism only | Preserved inside a task |
| tags | stage-selection predicate | A grammar selector, not an Airflow concept |
| `validate()` / `ValidationReport` | build-time gate | Refuse to emit a DAG with cycles or placeholders |
| `States` | does not map | Airflow task states are per-run; loman states are per-graph |
| partial recalculation | does not map | The deepest mismatch — see risks |

## Proposed API

Two layers. The lower one needs no Airflow at all.

```python
from datetime import timedelta

from loman.airflow import AirflowSpec, PathChannel, Queue, Stage, StageDefaults, build_plan

SPEC = AirflowSpec(
    stages=(
        Stage("market_data/**"),
        Stage("risk/*", per="block"),
        Stage.tagged("slow", per="node"),
        Stage("report"),
    ),
    queues=(Queue(executor="gpu", queue="gpu-queue", pool="gpu_pool"),),
    channel=PathChannel("/mnt/shared/loman/{dag_id}/{run_id}"),
    defaults=StageDefaults(retries=2, execution_timeout=timedelta(minutes=30)),
    on_unassigned="single_stage",
)

plan = build_plan(build_risk_computation(), SPEC)
plan.to_df()
```

Selectors reuse the glob language loman already has in `src/loman/nodekey.py`
(`match_pattern`, `is_pattern`), where `*` matches one path part and `**` matches
zero or more. The grammar is therefore not a new vocabulary. Every type is a
frozen dataclass with a `to_df()`, matching `ValidationReport` and
`ExecutionPlan` in `src/loman/planning.py`.

Only one module imports Airflow:

```python
from loman.airflow import to_dag

from my_pkg.pipelines import build_risk_computation

dag = to_dag(build_risk_computation, spec=SPEC, dag_id="risk", schedule="@daily")
```

Inside a worker, each task calls one entry point, which rebuilds the computation
from the factory, reads inbound handles from the channel and inserts them, calls
`comp.compute(targets)`, writes outbound boundary values, and returns the small
handle dict as its XCom.

## Design decisions

### Granularity

Task-per-node gives maximum Airflow UI fidelity but requires serializing every
intermediate; for loman's typical payloads (DataFrames, curve and portfolio
objects) that is a non-starter. Whole-computation-in-one-task is trivially
correct but adds no Airflow value.

Recommended: rule-selected stages, defaulting to one task per top-level block,
with `per="node"` available per rule. **This needs confirmation** — it is the
choice that most shapes the feature.

### How values cross a boundary

Airflow 3 removed `enable_xcom_pickling`; XCom is JSON-only. So the plan defines
a `ValueChannel` protocol with two implementations:

- `XComChannel` runs the value through loman's existing `Transformer` stack,
  which already handles ndarray, DataFrame and Series, with a size guard that
  raises rather than bloating the metadata database.
- `PathChannel` writes a document under `{dag_id}/{run_id}/{task_id}/{node}` on
  shared storage and puts only the URI in XCom.

`PathChannel` is the recommended default for real use. The protocol is public so
users can back it with object storage. The docs must be honest that loman's
"arbitrary Python object" guarantee stops at a stage boundary.

### Output form

`to_dag` takes a **zero-arg factory, not a `Computation` instance**. Airflow
re-parses DAG files roughly every 30 seconds, and the same graph must be rebuilt
deterministically in the scheduler, the DAG processor and every worker. Building
a loman graph is construction only, with no compute, so it is cheap — provided
the factory does not hit a database.

Validate at build time that the factory is importable, reusing
`FunctionRefTransformer.to_dict()` in `src/loman/serialization/transformer.py`,
which already does an import round-trip check and rejects lambdas and closures.
Users then get a clear error at parse time rather than a mystery worker failure.

Source generation is plausible later but v1 should not own a code generator.

### Which loman API is the bridge

`create_execution_plan` in `src/loman/planning.py` prunes anything already
`UPTODATE` or `PINNED` and reports uninitialized inputs as blocked, so its output
is state-dependent and wrong as a source of DAG shape: a freshly built graph
would report every input as blocked.

Instead, at build time call `validate_graph` to reject cycles, placeholders and
missing executors, compute the partition and the quotient graph over stages from
`comp.dag` directly, and topologically sort with `graph_utils.topological_sort`.
A rule set producing a cyclic quotient must be a build-time error naming the
crossing edges — this is the main correctness obligation of the module.

At run time, inside a task, `comp.compute(targets)` already does the right
partial recalculation, and `comp.plan(targets).to_df()` is good task-log material.

### Task identifier stability

Airflow validates identifiers against `^[A-Za-z0-9_\-.]+$` with a 250 character
limit, and task groups prefix with `.`. Two things must be errors rather than
silent mangling: node keys with non-string parts, whose `str()` is not a stable
identity; and two distinct keys sanitizing to the same identifier. Identifiers
must be stable across parses or Airflow loses task history.

### Dependency choice

Recommended: depend on `apache-airflow-task-sdk` rather than full
`apache-airflow`. The Task SDK is a separate distribution intended for DAG
authoring without installing Airflow core, and exports everything needed. This
makes the extra small enough to install in CI.

Unverified and worth a short spike: whether the Python TaskFlow `@task`
decorator resolves with the SDK alone or also needs
`apache-airflow-providers-standard`, since `PythonOperator` lives in that
provider.

## Layout and packaging

```text
src/loman/airflow/__init__.py     exports; lazy to_dag via module __getattr__
src/loman/airflow/spec.py         AirflowSpec, Stage, Group, Queue, StageDefaults
src/loman/airflow/partition.py    selector matching, stage assignment, quotient graph
src/loman/airflow/plan.py         DagPlan, StagePlan, Crossing, to_df()
src/loman/airflow/naming.py       NodeKey to task_id, legality and collision checks
src/loman/airflow/channels.py     ValueChannel protocol, XComChannel, PathChannel
src/loman/airflow/runtime.py      execute_stage()
src/loman/airflow/_sdk.py         the only module that imports airflow
src/loman/airflow/dag.py          to_dag()
```

`dag.py` must reference the SDK as attributes (`_sdk.task(...)`) rather than
importing the names directly, so tests can monkeypatch `loman.airflow._sdk`
without touching `sys.modules`. This works whether or not Airflow is installed,
which is the whole coverage strategy.

Nothing outside `src/loman/airflow/` imports it, and `src/loman/__init__.py` is
untouched. No `Computation.to_airflow()` method in v1.

The `[project.optional-dependencies]` convention and the shared optional-import
helper are **owned by the `loman[ui]` branch**; this branch adds only its own
entry once that has landed:

```toml
[project.optional-dependencies]
airflow = ["apache-airflow-task-sdk>=1.0,<2"]

[tool.deptry.package_module_name_map]
apache-airflow-task-sdk = "airflow"
```

The deptry entry is required because the distribution and module names differ.

## Delivery phases

| Phase | Content | Ships alone |
| --- | --- | --- |
| 0 | Spike: does the Task SDK alone give a working `@task`? Does it install on Windows and Python 3.14? | n/a |
| 1 | `spec.py`, `partition.py`, `naming.py`, `plan.py`. Pure Python, no new dependency | yes |
| 2 | `channels.py`, `runtime.py`. Round-trips a multi-stage computation with no Airflow present | yes |
| 3 | `_sdk.py`, `dag.py`, `to_dag()`, pyproject entry, CI job | yes |
| 4 | Docs page, mkdocs nav entry, changelog, install note | yes |
| 5 | Later: dynamic task mapping, asset-based scheduling, channel short-circuit as a partial-recalc analogue | — |

Phases 1 and 2 are valuable even if phase 3 never merges, which makes this low
risk to start.

## Testing approach

`make test` runs coverage in every matrix job, so each of ubuntu, macos and
windows across Python 3.11 to 3.14 must independently hit the bar. Airflow does
not install on Windows and its Python 3.14 support is unconfirmed, so **coverage
cannot depend on Airflow being present**. Three tiers:

1. Pure tests for the partitioner, naming, plan and channels — ordinary Python
   over a `networkx` graph, testable the way `tests/test_planning.py` already is,
   reusing the fixtures in `tests/conftest.py`.
2. Stubbed emitter tests: a fake SDK recorder monkeypatched onto
   `loman.airflow._sdk`, asserting the emitted structure matches the plan. This
   gives real line coverage of `dag.py` on every platform. Only the small
   try/except in `_sdk.py` carries a coverage pragma.
3. One real integration test on ubuntu with a single Python version, guarded by
   `pytest.importorskip`, asserting against Airflow's own API. This is the guard
   against the fake drifting from reality.

Note that `ty` runs on `src/` without Airflow installed, so the import in
`_sdk.py` will be unresolved. Expect to need a suppression or a `TYPE_CHECKING`
shim; check this early in phase 3.

## Risks and open questions

Fundamental mismatches, which should be stated plainly in the docs:

- Partial recalculation has no Airflow equivalent. loman recomputes only what is
  stale from live in-memory state; Airflow reruns a DAG on a schedule. The
  nearest analogue is short-circuiting when a channel handle already exists,
  which is a weaker guarantee and a later feature.
- loman states and Airflow task states are different things and should never be
  conflated.
- XCom is JSON-only in Airflow 3, so arbitrary objects crossing a boundary need
  `PathChannel` or a custom backend.

Needing a decision:

- Default granularity: one task per top-level block, per-node, or no default at
  all.
- Whether v1 adds a convenience method on `Computation` or keeps everything in
  `loman.airflow`.
- `apache-airflow-task-sdk` versus full `apache-airflow`.
- Which channel is the default.

To verify before phase 3:

- Whether `@task` needs `apache-airflow-providers-standard`.
- Whether the Task SDK installs on Windows and Python 3.14.
- Whether Airflow 3 has a per-task `executor` parameter, or whether queue-based
  routing is the only per-task lever.
- The exact SDK version floor.

Operational risks:

- A factory that touches a database at parse time will be re-executed every 30
  seconds by the DAG processor. This is the most likely way a user gets hurt and
  needs a loud documentation warning.
- Any change to the grammar or to block names silently renames tasks and orphans
  their Airflow history. Consider emitting a stable identifier manifest that can
  be diffed in review.
- Adding the extra will produce a large `uv.lock` diff; worth a separate commit.
