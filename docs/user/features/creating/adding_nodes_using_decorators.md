# Adding Nodes Using the `node` Decorator

Loman provide a decorator `@node`, which allows functions to be added to computations. The first parameter is the Computation object to add a node to. By default, it will take the node name from the function, and the names of input nodes from the names of the parameter of the function, but any parameters provided are passed through to `add_node`, including name:

```pycon
>>> from loman import *
>>> comp = Computation()
>>> comp.add_node('a', value=1)

>>> @node(comp)
... def b(a):
...    return a + 1

>>> @node(comp, 'c', args=['a'])
... def foo(x):
...    return 2 * x

>>> @node(comp, kwds={'x': 'a', 'y': 'b'})
... def d(x, y):
...    return x + y

>>> comp.draw()
```

```{graphviz}
    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#9dff00" style=filled]
        n2 [label=c fillcolor="#9dff00" style=filled]
        n3 [label=d fillcolor="#0343df" style=filled]
            n0 -> n1
            n0 -> n2
            n1 -> n3
            n2 -> n3
    }
```

