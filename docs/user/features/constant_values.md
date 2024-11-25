# Constant Values

When you are using a pre-existing function for a node, and one or more of the parameters takes a constant value, one way is to define a lambda, which fixes the parameter value. For example, below we use a lambda to fix the second parameter passed to the add function:

```pycon
>>> def add(x, y):
...    return x + y

>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: add(a, 1))
>>> comp.compute_all()
>>> comp.v.b
2
```

However providing `ConstantValue` objects to the `args` or `kwds` parameters of `add_node`, make this simpler. `C` is an alias for `ConstantValue`, and in the example below, we use that to tell node `b` to calculate by taking parameter `x` from node `a`, and `y` as a constant, `1`:

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', add, kwds={"x": "a", "y": C(1)})
>>> comp.compute_all()
>>> comp.v.b
2
```

