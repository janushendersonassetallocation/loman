# Loman Project Analysis Report

## Executive Summary

**Loman** is a Python library for managing complex computation flows through DAG-based (Directed Acyclic Graph) dependency tracking. It enables intelligent, selective recalculation—computing only the nodes whose dependencies have changed. NetworkX serves as the architectural backbone, providing the core graph data structure and algorithms.

---

## 1. Project Overview

### Purpose

Loman is designed to solve the problem of efficiently managing computational dependencies in complex workflows. Instead of recomputing entire pipelines when inputs change, Loman tracks dependencies and only recomputes affected nodes.

### Key Features

- **Smart Recalculation**: Only computes nodes that have changed
- **Automatic Dependency Tracking**: Graph-based dependency management
- **Selective Updates**: Control which outputs to compute and when
- **State Persistence**: Serialization support for checkpointing
- **Visual Debugging**: GraphViz integration for visualization
- **Real-time System Support**: Handles inputs that tick at different rates
- **Framework Agnostic**: Works with NumPy, Pandas, and other libraries

### Use Cases

| Use Case | Description |
|----------|-------------|
| Real-time Systems | Efficient update propagation when inputs change at different rates |
| Batch Processing | Pipeline checkpointing and resumption |
| Research Workflows | Iterative development with cached intermediate results |
| Analytics Pipelines | Complex data transformations with dependency tracking |

---

## 2. Project Structure

```
src/loman/
├── __init__.py              # Main module exports
├── computeengine.py         # Core computation graph engine (~1600 lines)
├── graph_utils.py           # NetworkX graph utilities
├── visualization.py         # GraphViz visualization (~500 lines)
├── nodekey.py               # Node naming/pathing system
├── consts.py                # Constants and enums
├── exception.py             # Custom exceptions
├── util.py                  # Utility functions
├── compat.py                # Compatibility functions
└── serialization/           # Serialization module
    ├── __init__.py
    ├── default.py           # Default transformer
    └── transformer.py       # Generic serialization framework
```

### Key Components

| File | Responsibility |
|------|----------------|
| `computeengine.py` | Core `Computation` class managing the DAG, node states, and execution |
| `graph_utils.py` | Graph operations (topological sort, cycle detection) |
| `visualization.py` | `GraphView` class for GraphViz rendering |
| `nodekey.py` | Hierarchical node naming system (`NodeKey` class) |
| `consts.py` | Enums for `States`, `NodeAttributes`, `EdgeAttributes`, `SystemTags` |
| `serialization/` | Custom serialization framework (alternative to pickle/dill) |

---

## 3. NetworkX Integration

### 3.1 Dependency Declaration

NetworkX is declared as a required dependency in `pyproject.toml`:

```toml
dependencies = [
    "networkx >= 2.0",
    ...
]
```

### 3.2 Core Data Structure

The entire computation engine is built on `nx.DiGraph()`. This is initialized in `computeengine.py:310`:

```python
self.dag = nx.DiGraph()
```

The `dag` (Directed Acyclic Graph) stores:

- **Nodes**: Computational units or data values
- **Edges**: Data dependencies between nodes
- **Node Attributes**: State, values, functions, tags, timing info
- **Edge Attributes**: Parameter mapping information

### 3.3 NetworkX Functions Used

| Function | Location | Purpose |
|----------|----------|---------|
| `nx.DiGraph()` | `computeengine.py:310` | Initialize the computation graph |
| `nx.topological_sort()` | `graph_utils.py:54` | Determine correct execution order |
| `nx.find_cycle()` | `graph_utils.py:59` | Detect circular dependencies |
| `nx.ancestors()` | `computeengine.py:1009, 1015, 1363` | Find upstream dependencies |
| `nx.descendants()` | `computeengine.py:746, 824, 1415, 1432` | Find downstream dependents |
| `nx.relabel_nodes()` | `computeengine.py:641` | Rename nodes in the graph |
| `nx.get_node_attributes()` | `computeengine.py:1303, 1322, 1475, 1502` | Extract node metadata |

### 3.4 Node Attributes

Each node in the graph stores the following attributes (defined in `consts.py`):

| Attribute | Description |
|-----------|-------------|
| `STATE` | Current computation state |
| `VALUE` | The computed result |
| `FUNC` | Function to compute this node |
| `ARGS` | Positional parameter sources |
| `KWDS` | Keyword parameter sources |
| `TAG` | User-assigned tags |
| `TIMING` | Execution timing information |

### 3.5 Node States

Nodes transition through the following states:

```
UNINITIALIZED → COMPUTABLE → UPTODATE
                    ↓
                  ERROR

UPTODATE → STALE → COMPUTABLE → UPTODATE
```

| State | Description |
|-------|-------------|
| `UNINITIALIZED` | Node has no value and cannot be computed |
| `COMPUTABLE` | Node can be computed (all dependencies ready) |
| `UPTODATE` | Node has been computed and is current |
| `STALE` | Node's dependencies have changed; needs recomputation |
| `ERROR` | Computation failed with an exception |
| `PLACEHOLDER` | Node is a placeholder for future definition |

### 3.6 Graph Algorithms

#### Topological Sort (`graph_utils.py:24-67`)

Used to determine the correct computation order, ensuring dependencies are computed before dependents:

```python
def _topological_sort(dag, sources):
    """
    Perform topological sort on a subset of the DAG.
    Raises LoopDetectedError if cycles are detected.
    """
    try:
        return list(nx.topological_sort(dag.subgraph(sources)))
    except nx.NetworkXUnfeasible:
        cycle = nx.find_cycle(dag)
        raise LoopDetectedError(cycle)
```

#### Ancestor/Descendant Traversal

- **Ancestors** (`nx.ancestors()`): Find all upstream dependencies
- **Descendants** (`nx.descendants()`): Find all downstream dependents

These are critical for:
1. Determining what needs to be computed for a given target
2. Cascading state changes (marking nodes as `STALE` when inputs change)

---

## 4. Visualization System

The `visualization.py` module provides GraphViz integration through the `GraphView` class:

```python
class GraphView:
    def __init__(self, comp: Computation):
        self.comp = comp
        self.struct_dag: nx.DiGraph | None = None
        self.viz_dag: nx.DiGraph | None = None
```

### Features

- Node transformation (COLLAPSE, CONTRACT, EXPAND)
- State-based coloring
- Export to multiple formats (PNG, SVG, PDF)
- Interactive notebook display

---

## 5. Serialization

### Current Implementation

The project currently uses `dill` for serialization (`computeengine.py:1519-1552`):

```python
def write_dill(self, filename):
    """Serialize computation to file using dill."""
    with open(filename, "wb") as f:
        dill.dump(self, f)

@staticmethod
def read_dill(filename):
    """Deserialize computation from file."""
    with open(filename, "rb") as f:
        return dill.load(f)
```

### State Handling

The `__getstate__` method prepares computations for serialization by filtering nodes:

```python
def __getstate__(self):
    node_serialize = nx.get_node_attributes(self.dag, NodeAttributes.TAG)
    obj = self.copy()
    for name, tags in node_serialize.items():
        if SystemTags.SERIALIZE not in tags:
            obj._set_uninitialized(name)
    return {"dag": obj.dag}
```

### Current Branch Work

The branch `7-avoid-serialization-with-pickle-and-dill` introduces a custom serialization framework in `serialization/transformer.py` to avoid pickle/dill security concerns.

---

## 6. Design Patterns

### 6.1 Lazy Evaluation

Nodes remain unevaluated until explicitly requested. Dependencies are tracked but not computed until needed.

### 6.2 Memoization

Computed values are stored in node attributes, avoiding recomputation when inputs haven't changed.

### 6.3 State Machine

Each node follows a state machine pattern, with transitions triggering cascade updates to dependent nodes.

### 6.4 Implicit Dependency Inference

Function parameter names automatically map to node names:

```python
@comp.node
def total(price, quantity):  # Automatically depends on 'price' and 'quantity' nodes
    return price * quantity
```

### 6.5 Observer Pattern

When a node's state changes, all descendants are notified and marked as `STALE`.

---

## 7. Strengths of NetworkX Integration

| Strength | Benefit |
|----------|---------|
| **Mature Library** | Well-tested, production-ready graph operations |
| **Built-in Algorithms** | Topological sort, cycle detection, traversal out of the box |
| **Attribute Storage** | Native support for node/edge metadata |
| **Extensibility** | Easy to add custom graph algorithms |
| **Performance** | Efficient for DAG operations at typical scales |
| **Serialization** | NetworkX graphs can be pickled (relevant to current work) |

---

## 8. Conclusion

Loman demonstrates an effective use of NetworkX as a foundation for building domain-specific computation engines. The integration is deep and well-architected:

- NetworkX provides the core data structure (`DiGraph`)
- Built-in algorithms handle graph operations (sort, traversal, cycle detection)
- Node attributes store computational metadata
- The abstraction layer hides NetworkX complexity from end users

The current branch work on serialization shows continued evolution of the library, addressing security concerns while maintaining the NetworkX-based architecture.

---

*Report generated: January 2026*
