"""Non-mutating validation and execution planning for computation graphs."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from .consts import NodeAttributes, States
from .nodekey import NodeKey

_CURRENT_STATES = {States.UPTODATE, States.PINNED}


@dataclass(frozen=True)
class ValidationReport:
    """Structural and readiness findings for a computation graph."""

    cycles: tuple[tuple[NodeKey, ...], ...]
    placeholders: tuple[NodeKey, ...]
    uninitialized_inputs: tuple[NodeKey, ...]
    error_nodes: tuple[NodeKey, ...]
    missing_executors: tuple[tuple[NodeKey, str], ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the graph is structurally valid."""
        return not self.cycles and not self.placeholders and not self.missing_executors

    @property
    def is_ready(self) -> bool:
        """Return whether the graph is valid and has all required inputs."""
        return self.is_valid and not self.uninitialized_inputs and not self.error_nodes

    def to_df(self) -> pd.DataFrame:
        """Return validation findings as one row per issue."""
        rows: list[dict[str, str]] = []
        for component in self.cycles:
            nodes = ", ".join(str(node) for node in component)
            rows.append({"issue": "cycle", "node": nodes, "detail": f"Dependency cycle: {nodes}"})
        rows.extend(
            {"issue": "placeholder", "node": str(node), "detail": "Node has no definition"}
            for node in self.placeholders
        )
        rows.extend(
            {"issue": "uninitialized_input", "node": str(node), "detail": "Input value has not been supplied"}
            for node in self.uninitialized_inputs
        )
        rows.extend(
            {"issue": "error", "node": str(node), "detail": "Previous calculation failed"} for node in self.error_nodes
        )
        rows.extend(
            {
                "issue": "missing_executor",
                "node": str(node),
                "detail": f"Executor {executor_name!r} is not configured",
            }
            for node, executor_name in self.missing_executors
        )
        return pd.DataFrame(rows, columns=["issue", "node", "detail"])


@dataclass(frozen=True)
class ExecutionPlan:
    """A non-mutating description of work required by a computation."""

    targets: tuple[NodeKey, ...] | None
    execution_order: tuple[NodeKey, ...]
    current_nodes: tuple[NodeKey, ...]
    blocked_nodes: tuple[NodeKey, ...]
    node_states: tuple[tuple[NodeKey, States], ...]
    blocked_by: tuple[tuple[NodeKey, tuple[NodeKey, ...]], ...]
    blocker_reasons: tuple[tuple[NodeKey, str], ...]
    executor_assignments: tuple[tuple[NodeKey, str], ...]
    validation: ValidationReport

    @property
    def is_feasible(self) -> bool:
        """Return whether all requested targets can be brought up to date."""
        return self.validation.is_ready and not self.blocked_nodes

    def to_df(self) -> pd.DataFrame:
        """Return current, runnable, and blocked nodes as an execution table."""
        order = {node: index + 1 for index, node in enumerate(self.execution_order)}
        executors = dict(self.executor_assignments)
        states = dict(self.node_states)
        blocked_by = dict(self.blocked_by)
        reasons = dict(self.blocker_reasons)
        rows = []
        for node in (*self.current_nodes, *self.execution_order, *self.blocked_nodes):
            blockers = blocked_by.get(node, ())
            blocker_details = tuple(dict.fromkeys(reasons[blocker] for blocker in blockers))
            plan_status = "blocked" if blockers else "current" if node in self.current_nodes else "pending"
            rows.append(
                {
                    "node": str(node),
                    "state": states[node],
                    "plan_status": plan_status,
                    "order": order.get(node),
                    "executor": executors.get(node),
                    "blocked_by": ", ".join(str(blocker) for blocker in blockers) or None,
                    "reason": "; ".join(blocker_details) or None,
                }
            )
        return pd.DataFrame(
            rows,
            columns=["node", "state", "plan_status", "order", "executor", "blocked_by", "reason"],
        ).set_index("node")


def _ordered_nodes(dag: nx.DiGraph, nodes: Collection[NodeKey]) -> tuple[NodeKey, ...]:
    """Return *nodes* in graph insertion order."""
    return tuple(node for node in dag.nodes if node in nodes)


def _find_cycles(dag: nx.DiGraph, nodes: Collection[NodeKey]) -> tuple[tuple[NodeKey, ...], ...]:
    """Return cyclic strongly connected components in graph insertion order."""
    node_order = {node: index for index, node in enumerate(dag.nodes)}
    graph = dag.subgraph(nodes)
    components = []
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1 or any(graph.has_edge(node, node) for node in component):
            components.append(tuple(sorted(component, key=node_order.__getitem__)))
    components.sort(key=lambda component: node_order[component[0]])
    return tuple(components)


def validate_graph(
    dag: nx.DiGraph,
    executor_map: Mapping[str, object],
    *,
    nodes: Collection[NodeKey] | None = None,
    executor_nodes: Collection[NodeKey] | None = None,
) -> ValidationReport:
    """Inspect a graph, or a subset of it, without changing it."""
    selected = set(dag.nodes if nodes is None else nodes)
    checked_executors = selected if executor_nodes is None else set(executor_nodes)

    placeholders: set[NodeKey] = set()
    uninitialized_inputs: set[NodeKey] = set()
    error_nodes: set[NodeKey] = set()
    missing_executors: list[tuple[NodeKey, str]] = []

    for node_key in dag.nodes:
        if node_key not in selected:
            continue
        node = dag.nodes[node_key]
        state = node.get(NodeAttributes.STATE)
        if state == States.PLACEHOLDER:
            placeholders.add(node_key)
        elif state != States.ERROR and state not in _CURRENT_STATES and node.get(NodeAttributes.FUNC) is None:
            uninitialized_inputs.add(node_key)
        if state == States.ERROR:
            error_nodes.add(node_key)

        executor_name = node.get(NodeAttributes.EXECUTOR)
        if (
            node_key in checked_executors
            and node.get(NodeAttributes.FUNC) is not None
            and executor_name is not None
            and executor_name not in executor_map
        ):
            missing_executors.append((node_key, executor_name))

    return ValidationReport(
        cycles=_find_cycles(dag, selected),
        placeholders=_ordered_nodes(dag, placeholders),
        uninitialized_inputs=_ordered_nodes(dag, uninitialized_inputs),
        error_nodes=_ordered_nodes(dag, error_nodes),
        missing_executors=tuple(missing_executors),
    )


def _select_required_nodes(dag: nx.DiGraph, targets: Sequence[NodeKey]) -> tuple[set[NodeKey], set[NodeKey]]:
    """Select target dependencies, stopping traversal at nodes with current values."""
    selected: set[NodeKey] = set()
    current: set[NodeKey] = set()
    to_visit = list(reversed(targets))

    while to_visit:
        node_key = to_visit.pop()
        if node_key in selected:
            continue
        selected.add(node_key)
        if dag.nodes[node_key].get(NodeAttributes.STATE) in _CURRENT_STATES:
            current.add(node_key)
            continue
        to_visit.extend(reversed(list(dag.predecessors(node_key))))

    return selected, current


def create_execution_plan(
    dag: nx.DiGraph,
    executor_map: Mapping[str, object],
    targets: Sequence[NodeKey] | None,
) -> ExecutionPlan:
    """Build a deterministic execution plan without running or mutating nodes."""
    plan_targets = tuple(dag.nodes) if targets is None else tuple(dict.fromkeys(targets))
    selected, current = _select_required_nodes(dag, plan_targets)
    pending = selected - current
    validation = validate_graph(dag, executor_map, nodes=selected, executor_nodes=pending)

    blocker_reasons: dict[NodeKey, str] = {}
    blocker_reasons.update((node, "Node has no definition") for node in validation.placeholders)
    blocker_reasons.update((node, "Input value has not been supplied") for node in validation.uninitialized_inputs)
    blocker_reasons.update((node, "Previous calculation failed") for node in validation.error_nodes)
    for component in validation.cycles:
        detail = f"Dependency cycle: {', '.join(str(node) for node in component)}"
        blocker_reasons.update((node, detail) for node in component)
    blocker_reasons.update(
        (node, f"Executor {executor_name!r} is not configured") for node, executor_name in validation.missing_executors
    )

    pending_graph = dag.subgraph(pending)
    blocked_by: dict[NodeKey, set[NodeKey]] = {node: {node} for node in blocker_reasons if node in pending}
    for blocker in blocker_reasons:
        if blocker in pending:
            for descendent in nx.descendants(pending_graph, blocker):
                blocked_by.setdefault(descendent, set()).add(blocker)

    blocked = set(blocked_by)
    node_order = {node: index for index, node in enumerate(dag.nodes)}

    runnable = pending - blocked
    runnable_graph = dag.subgraph(runnable)
    execution_order = tuple(nx.topological_sort(runnable_graph))
    assignments = tuple(
        (
            node_key,
            "default"
            if dag.nodes[node_key].get(NodeAttributes.EXECUTOR) is None
            else dag.nodes[node_key][NodeAttributes.EXECUTOR],
        )
        for node_key in execution_order
    )

    return ExecutionPlan(
        targets=None if targets is None else tuple(dict.fromkeys(targets)),
        execution_order=execution_order,
        current_nodes=_ordered_nodes(dag, current),
        blocked_nodes=_ordered_nodes(dag, blocked),
        node_states=tuple((node, dag.nodes[node][NodeAttributes.STATE]) for node in dag.nodes if node in selected),
        blocked_by=tuple(
            (node, tuple(sorted(blocked_by[node], key=node_order.__getitem__)))
            for node in dag.nodes
            if node in blocked_by
        ),
        blocker_reasons=tuple((node, blocker_reasons[node]) for node in dag.nodes if node in blocker_reasons),
        executor_assignments=assignments,
        validation=validation,
    )
