# Loman Serialization Analysis

## Current State

Loman currently uses **dill** for serialization (`computeengine.py:1519-1552`). This choice was driven by the need to serialize Python functions stored in node attributes.

### Why Dill?

NetworkX provides several built-in serialization formats, but none can handle Loman's requirements:

| Format | Can Serialize Functions? | Notes |
|--------|--------------------------|-------|
| JSON (`node_link_data`) | No | Text-based, no callable support |
| GraphML | No | XML-based, no callable support |
| GML | No | Text-based, no callable support |
| Pickle (`write_gpickle`) | Limited | Cannot serialize lambdas, closures |
| **Dill** | Yes | Extends pickle for complex callables |

The core issue is that Loman stores **Python functions** in node attributes:

```python
class NodeAttributes:
    VALUE = "value"       # Arbitrary Python objects
    FUNC = "func"         # Python callables ← Problem!
    EXECUTOR = "executor" # Executor objects
    CONVERTER = "converter"
```

Standard pickle cannot serialize lambdas, closures, or interactively-defined functions. Dill extends pickle to handle these cases.

### Security Concerns

From `computeengine.py:1535-1539`:

```python
.. warning::
    This method uses dill.load() which can execute arbitrary code.
    Only load files from trusted sources. Never load data from
    untrusted or unauthenticated sources as it may lead to arbitrary
    code execution.
```

---

## The Transformer Approach

The `serialization/` module introduces a new framework that converts objects to JSON-serializable dictionaries without executing arbitrary code.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Transformer                               │
│  Main orchestrator - routes objects to appropriate handlers      │
├─────────────────────────────────────────────────────────────────┤
│  Built-in support:                                               │
│  • Primitives (str, int, float, bool, None)                     │
│  • Collections (list, dict, tuple)                              │
│  • dataclasses                                                   │
│  • attrs classes                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Extensible via:                                                 │
│  • CustomTransformer - for external types (e.g., numpy)         │
│  • Transformable - for self-serializing classes                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. `Transformer` Class

The main serializer that orchestrates conversion:

```python
from loman.serialization import default_transformer

transformer = default_transformer(strict=True)

# Serialize to dict
data = transformer.to_dict(my_object)

# Deserialize from dict
obj = transformer.from_dict(data)
```

#### 2. `CustomTransformer` ABC

For adding support for external types:

```python
class NdArrayTransformer(CustomTransformer):
    @property
    def name(self):
        return "ndarray"

    def to_dict(self, transformer, o):
        return {
            "shape": list(o.shape),
            "dtype": o.dtype.str,
            "data": transformer.to_dict(o.ravel().tolist())
        }

    def from_dict(self, transformer, d):
        return np.array(d["data"], d["dtype"]).reshape(d["shape"])

    @property
    def supported_direct_types(self):
        return [np.ndarray]
```

#### 3. `Transformable` ABC

For classes that can serialize themselves:

```python
class MyClass(Transformable):
    def to_dict(self, transformer):
        return {"field": self.field}

    @classmethod
    def from_dict(cls, transformer, d):
        return cls(field=d["field"])
```

### Output Format

Objects are serialized with type markers for reconstruction:

```python
# Tuple
{"type": "tuple", "values": [1, 2, 3]}

# Numpy array
{"type": "ndarray", "shape": [2, 3], "dtype": "<f8", "data": [...]}

# Dataclass
{"type": "dataclass", "class": "MyData", "data": {"x": 1, "y": 2}}

# attrs class
{"type": "attrs", "class": "MyAttrs", "data": {"name": "foo"}}
```

### Type Registration

The transformer uses a registration system:

```python
transformer = Transformer(strict=True)

# Register custom transformer for external types
transformer.register(NdArrayTransformer())

# Register transformable class
transformer.register(MyTransformableClass)

# Register dataclass (auto-detected)
transformer.register(MyDataClass)

# Register attrs class (auto-detected)
transformer.register(MyAttrsClass)
```

### Strict vs Non-Strict Mode

- **Strict mode** (`strict=True`): Raises exceptions for unrecognized types
- **Non-strict mode** (`strict=False`): Returns `MissingObject()` sentinel for unrecognized types

---

## Comparison

| Aspect | Dill | Transformer |
|--------|------|-------------|
| **Output format** | Binary (opaque) | JSON (human-readable) |
| **Security** | Arbitrary code execution | No code execution |
| **Portability** | Python-version sensitive | Language-agnostic |
| **Debugging** | Difficult to inspect | Easy to inspect |
| **Function support** | Full | None (by design) |
| **Extensibility** | Limited | Plugin-based |
| **Selective** | All-or-nothing | Register only needed types |

---

## Current Limitations

The transformer framework handles **data** well but **not functions**. For Loman to fully adopt this approach, the following strategies could be considered:

### Option 1: Store Function References

Instead of storing function objects, store importable references:

```python
# Instead of storing the function object
{"func": <function total at 0x...>}

# Store a reference
{"func_module": "mymodule", "func_name": "total"}
```

Reconstruction requires the function to be importable at deserialization time.

### Option 2: Serialize Values Only

Only serialize computed values, not the computation graph structure:

```python
# Serialize
values = {name: node.value for name in comp.nodes()}

# Deserialize requires rebuilding the graph from code
comp = build_computation()  # From source code
comp.insert_from(values)    # Restore values
```

### Option 3: Hybrid Approach

Use the transformer for values, with a separate mechanism for graph structure:

```python
{
    "nodes": {
        "price": {"value": 100, "state": "UPTODATE"},
        "quantity": {"value": 5, "state": "UPTODATE"},
        "total": {"value": 500, "state": "UPTODATE"}
    },
    "edges": [
        ["price", "total", {"param": ["kwd", "price"]}],
        ["quantity", "total", {"param": ["kwd", "quantity"]}]
    ],
    "functions": {
        "total": {"module": "mymodule", "name": "calculate_total"}
    }
}
```

---

## Recommendations

1. **For checkpointing computed values**: The transformer approach is ideal - serialize node values to JSON, reload into a fresh computation graph built from code.

2. **For full graph persistence**: Consider a hybrid approach where:
   - Values use the transformer (safe, inspectable)
   - Functions use importable references (requires code availability)

3. **For untrusted environments**: Never use dill - require computation graphs to be rebuilt from code, only restoring values.

---

## Files

| File | Purpose |
|------|---------|
| `serialization/__init__.py` | Public API exports |
| `serialization/transformer.py` | Core framework (~340 lines) |
| `serialization/default.py` | Default transformer with numpy support |
