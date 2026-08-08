"""Tests for graph validation and non-mutating execution planning."""

from concurrent.futures import Executor

import pandas as pd
import pytest

from loman import Computation, ExecutionPlan, NonExistentNodeError, States, ValidationReport
from loman.consts import NodeAttributes
from loman.nodekey import NodeKey, to_nodekey


class FailOnSubmitExecutor(Executor):
    """Executor that fails if planning accidentally submits work."""

    def submit(self, fn, /, *args, **kwargs):
        """Reject submitted work."""
        raise AssertionError

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Shut down without resources to release."""


def test_validate_empty_graph():
    """An empty graph is valid and ready."""
    report = Computation().validate()

    assert isinstance(report, ValidationReport)
    assert report.is_valid
    assert report.is_ready
    assert report.cycles == ()
    assert report.placeholders == ()
    assert report.uninitialized_inputs == ()
    assert report.error_nodes == ()
    assert report.missing_executors == ()


def test_validate_reports_graph_problems_in_insertion_order():
    """Validation reports structural and readiness findings without raising."""
    comp = Computation()
    comp.add_node("missing_input")
    comp.add_node("uses_placeholder", lambda placeholder: placeholder)
    comp.add_node("bad_executor", lambda: 1, executor="missing")
    comp.add_node("error", lambda: 1 / 0)
    comp.compute("error")
    comp.add_node("cycle_a", lambda cycle_b: cycle_b)
    comp.add_node("cycle_b", lambda cycle_a: cycle_a)

    report = comp.validate()

    assert not report.is_valid
    assert not report.is_ready
    assert report.placeholders == (to_nodekey("placeholder"),)
    assert report.uninitialized_inputs == (to_nodekey("missing_input"),)
    assert report.error_nodes == (to_nodekey("error"),)
    assert report.missing_executors == ((to_nodekey("bad_executor"), "missing"),)
    assert report.cycles == ((to_nodekey("cycle_a"), to_nodekey("cycle_b")),)


def test_validate_distinguishes_valid_from_ready():
    """Missing supplied inputs affect readiness but not structural validity."""
    comp = Computation()
    comp.add_node("input")
    comp.add_node("output", lambda input: input)

    report = comp.validate()

    assert report.is_valid
    assert not report.is_ready
    assert report.uninitialized_inputs == (to_nodekey("input"),)


def test_validation_to_df_has_one_row_per_issue():
    """Validation findings are available as a compact issue table."""
    comp = Computation()
    comp.add_node("input")
    comp.add_node("uses_placeholder", lambda absent: absent)
    comp.add_node("bad_executor", lambda: 1, executor="gpu")
    comp.add_node("cycle_a", lambda cycle_b: cycle_b)
    comp.add_node("cycle_b", lambda cycle_a: cycle_a)

    result = comp.validate().to_df()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["issue", "node", "detail"]
    assert result["issue"].tolist() == [
        "cycle",
        "placeholder",
        "uninitialized_input",
        "missing_executor",
    ]
    assert result.loc[result["issue"] == "cycle", "node"].item() == "cycle_a, cycle_b"
    assert result.loc[result["issue"] == "missing_executor", "detail"].item() == "Executor 'gpu' is not configured"


def test_empty_validation_to_df_has_stable_columns():
    """A valid graph returns an empty DataFrame with a useful schema."""
    result = Computation().validate().to_df()

    assert result.empty
    assert list(result.columns) == ["issue", "node", "detail"]


def test_plan_orders_pending_nodes_and_records_executors():
    """A target plan lists required calculations in dependency order."""
    named_executor = FailOnSubmitExecutor()
    comp = Computation(default_executor=FailOnSubmitExecutor(), executor_map={"io": named_executor})
    comp.add_node("input", value=2)
    comp.add_node("double", lambda input: input * 2, executor="io")
    comp.add_node("total", lambda double: double + 1)

    plan = comp.plan("total")

    assert isinstance(plan, ExecutionPlan)
    assert plan.is_feasible
    assert plan.targets == (to_nodekey("total"),)
    assert plan.execution_order == (to_nodekey("double"), to_nodekey("total"))
    assert plan.current_nodes == (to_nodekey("input"),)
    assert plan.blocked_nodes == ()
    assert plan.executor_assignments == ((to_nodekey("double"), "io"), (to_nodekey("total"), "default"))


def test_plan_multiple_targets_is_deterministic_and_deduplicated():
    """Multiple targets share work and preserve deterministic graph order."""
    comp = Computation()
    comp.add_node("input", value=2)
    comp.add_node("left", lambda input: input + 1)
    comp.add_node("right", lambda input: input * 2)

    plan = comp.plan(["right", "left", "right"])

    assert plan.targets == (to_nodekey("right"), to_nodekey("left"))
    assert plan.execution_order == (to_nodekey("left"), to_nodekey("right"))
    assert plan.current_nodes == (to_nodekey("input"),)


def test_plan_empty_targets_has_no_work():
    """An empty target list produces an empty feasible plan."""
    comp = Computation()
    comp.add_node("unused")

    plan = comp.plan([])

    assert plan.is_feasible
    assert plan.targets == ()
    assert plan.execution_order == ()
    assert plan.current_nodes == ()
    assert plan.blocked_nodes == ()


def test_plan_none_inspects_all_branches_and_keeps_runnable_work():
    """A whole-graph plan reports blocked work while retaining independent work."""
    comp = Computation()
    comp.add_node("ready", lambda: 1)
    comp.add_node("missing")
    comp.add_node("blocked", lambda missing: missing + 1)

    plan = comp.plan()

    assert not plan.is_feasible
    assert plan.targets is None
    assert plan.execution_order == (to_nodekey("ready"),)
    assert plan.blocked_nodes == (to_nodekey("missing"), to_nodekey("blocked"))
    assert plan.validation.uninitialized_inputs == (to_nodekey("missing"),)


def test_target_plan_ignores_unrelated_cycle_and_placeholder():
    """Problems outside a target's dependency slice do not block its plan."""
    comp = Computation()
    comp.add_node("input", value=1)
    comp.add_node("output", lambda input: input + 1)
    comp.add_node("cycle_a", lambda cycle_b: cycle_b)
    comp.add_node("cycle_b", lambda cycle_a: cycle_a)
    comp.add_node("unrelated", lambda absent: absent)

    plan = comp.plan("output")

    assert plan.is_feasible
    assert plan.execution_order == (to_nodekey("output"),)
    assert plan.validation.cycles == ()
    assert plan.validation.placeholders == ()

    comp.compute("output")
    assert comp.value("output") == 2


def test_plan_propagates_blockers_to_required_descendants():
    """Missing inputs block every required downstream calculation."""
    comp = Computation()
    comp.add_node("input")
    comp.add_node("middle", lambda input: input + 1)
    comp.add_node("output", lambda middle: middle + 1)

    plan = comp.plan("output")

    assert not plan.is_feasible
    assert plan.execution_order == ()
    assert plan.blocked_nodes == (
        to_nodekey("input"),
        to_nodekey("middle"),
        to_nodekey("output"),
    )
    assert dict(plan.blocked_by) == {
        to_nodekey("input"): (to_nodekey("input"),),
        to_nodekey("middle"): (to_nodekey("input"),),
        to_nodekey("output"): (to_nodekey("input"),),
    }


def test_plan_to_df_explains_status_order_and_root_blockers():
    """The plan table combines runnable work with downstream root causes."""
    comp = Computation(executor_map={"io": FailOnSubmitExecutor()})
    comp.add_node("current", value=1)
    comp.add_node("runnable", lambda current: current + 1, executor="io")
    comp.add_node("missing")
    comp.add_node("blocked", lambda missing: missing + 1)
    comp.add_node("target", lambda runnable, blocked: runnable + blocked)

    result = comp.plan("target").to_df()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["state", "plan_status", "order", "executor", "blocked_by", "reason"]
    assert result.loc["current", "state"] == States.UPTODATE
    assert result.loc["current", "plan_status"] == "current"
    assert result.loc["runnable", "state"] == States.COMPUTABLE
    assert result.loc["runnable", "plan_status"] == "pending"
    assert result.loc["runnable", "order"] == 1
    assert result.loc["runnable", "executor"] == "io"
    assert result.loc["missing", "state"] == States.UNINITIALIZED
    assert result.loc["missing", "plan_status"] == "blocked"
    assert result.loc["target", "blocked_by"] == "missing"
    assert result.loc["target", "reason"] == "Input value has not been supplied"


def test_plan_tracks_multiple_root_blockers():
    """A downstream node reports every independent root cause that reaches it."""
    comp = Computation()
    comp.add_node("missing_a")
    comp.add_node("missing_b")
    comp.add_node("left", lambda missing_a: missing_a)
    comp.add_node("right", lambda missing_b: missing_b)
    comp.add_node("output", lambda left, right: left + right)

    plan = comp.plan("output")

    assert dict(plan.blocked_by)[to_nodekey("output")] == (
        to_nodekey("missing_a"),
        to_nodekey("missing_b"),
    )
    assert plan.to_df().loc["output", "blocked_by"] == "missing_a, missing_b"


def test_plan_reports_missing_executor_only_when_work_is_pending():
    """A missing named executor blocks pending work but not a current value."""
    comp = Computation()
    comp.add_node("output", lambda: 1, executor="missing")

    pending_plan = comp.plan("output")
    assert not pending_plan.is_feasible
    assert pending_plan.blocked_nodes == (to_nodekey("output"),)
    assert pending_plan.validation.missing_executors == ((to_nodekey("output"), "missing"),)

    comp.insert("output", 1)
    current_plan = comp.plan("output")
    assert current_plan.is_feasible
    assert current_plan.validation.missing_executors == ()


def test_plan_stops_at_current_and_pinned_nodes():
    """Current values form boundaries even when their ancestors are unavailable."""
    comp = Computation()
    comp.add_node("unavailable")
    comp.add_node("cached", lambda unavailable: unavailable)
    comp.insert("cached", 10)
    comp.pin("cached")
    comp.add_node("output", lambda cached: cached + 1)

    plan = comp.plan("output")

    assert plan.is_feasible
    assert plan.current_nodes == (to_nodekey("cached"),)
    assert plan.execution_order == (to_nodekey("output"),)
    assert to_nodekey("unavailable") not in plan.validation.uninitialized_inputs


def test_whole_graph_plan_does_not_propagate_blocker_through_current_node():
    """A current node prevents its unavailable ancestor from blocking descendants."""
    comp = Computation()
    comp.add_node("unavailable")
    comp.add_node("cached", lambda unavailable: unavailable)
    comp.insert("cached", 10)
    comp.add_node("output", lambda cached: cached + 1)

    plan = comp.plan()

    assert not plan.is_feasible
    assert plan.execution_order == (to_nodekey("output"),)
    assert plan.blocked_nodes == (to_nodekey("unavailable"),)


def test_pinned_predecessor_can_be_computed_downstream():
    """Pinned values satisfy downstream dependencies during real execution."""
    comp = Computation()
    comp.add_node("input", value=1)
    comp.pin("input")
    comp.add_node("output", lambda input: input + 1)

    comp.compute("output")

    assert comp.state("output") == States.UPTODATE
    assert comp.value("output") == 2


def test_plan_supports_non_string_scalar_names():
    """A hashable tuple remains a scalar node name rather than a target list."""
    name = ("tuple", 1)
    comp = Computation()
    comp.add_node(name, lambda: 1)

    plan = comp.plan(name)

    assert plan.targets == (NodeKey((name,)),)
    assert plan.execution_order == (NodeKey((name,)),)


def test_plan_unknown_target_raises():
    """Planning a missing target uses the public missing-node exception."""
    with pytest.raises(NonExistentNodeError, match="Node absent does not exist"):
        Computation().plan("absent")


def test_plan_does_not_execute_convert_or_mutate():
    """Planning is observational even when nodes have executable behavior."""
    calls = []

    def calculate():
        calls.append("calculate")
        return 1

    def convert(value):
        calls.append("convert")
        return value

    comp = Computation(default_executor=FailOnSubmitExecutor())
    comp.add_node("output", calculate, converter=convert)
    state_before = comp.state("output")
    value_before = comp.value("output")
    timing_before = comp.get_timing("output")
    state_map_before = {state: nodes.copy() for state, nodes in comp._state_map.items()}

    comp.validate()
    plan = comp.plan("output")

    assert plan.is_feasible
    assert calls == []
    assert comp.state("output") == state_before
    assert comp.value("output") == value_before
    assert comp.get_timing("output") == timing_before
    assert comp._state_map == state_map_before
    assert comp.dag.nodes[to_nodekey("output")][NodeAttributes.CONVERTER] is convert


def test_plan_error_node_blocks_target():
    """An existing error is reported as a blocker until recalculated or replaced."""
    comp = Computation()
    comp.add_node("error", lambda: 1 / 0)
    comp.add_node("output", lambda error: error)
    comp.compute("output")

    plan = comp.plan("output")

    assert not plan.is_feasible
    assert plan.validation.error_nodes == (to_nodekey("error"),)
    assert plan.blocked_nodes == (to_nodekey("error"), to_nodekey("output"))


def test_plan_self_loop_is_blocked():
    """Self-loops are reported as cycles rather than topologically sorted."""
    comp = Computation()
    comp.add_node("loop", lambda loop: loop)

    plan = comp.plan("loop")

    assert not plan.is_feasible
    assert plan.validation.cycles == ((to_nodekey("loop"),),)
    assert plan.blocked_nodes == (to_nodekey("loop"),)
    assert plan.execution_order == ()
