# Non-string node names

In the previous example, the nodes have all been given strings as keys. This is not a requirement, and in fact any object that could be used as a key in a dictionary can be a key for a node. As function parameters can only be strings, we have to rely on the `kwds` argument to `add_node` to specify which nodes should be used as inputs for calculation nodes' functions. For a simple but frivolous example, we can represent a finite part of the Fibonacci sequence using tuples of the form `('fib', [int])` as keys:

```pycon
>>> comp = Computation()
>>> comp.add_node(('fib', 1), value=1)
>>> comp.add_node(('fib', 2), value=1)
>>> for i in range(3,7):
...    comp.add_node(('fib', i), lambda x, y: x + y, kwds={'x': ('fib', i - 1), 'y': ('fib', i - 2)})
...
>>> comp.draw()
```

```{graphviz}
    digraph {
        n0 [label="('fib', 1)" fillcolor="#15b01a" style=filled]
        n1 [label="('fib', 2)" fillcolor="#15b01a" style=filled]
        n2 [label="('fib', 3)" fillcolor="#9dff00" style=filled]
        n3 [label="('fib', 4)" fillcolor="#0343df" style=filled]
        n4 [label="('fib', 5)" fillcolor="#0343df" style=filled]
        n5 [label="('fib', 6)" fillcolor="#0343df" style=filled]
            n0 -> n2
            n1 -> n2
            n1 -> n3
            n2 -> n3
            n2 -> n4
            n3 -> n4
            n3 -> n5
            n4 -> n5
    }
```

```pycon
>>> comp.compute_all()
>>> comp.value(('fib', 6))
8
```
