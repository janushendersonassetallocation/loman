# Automatically expanding named tuples

Often, a calculation will return more than one result. For example, a numerical solver may return the best solution it found, along with a status indicating whether the solver converged. Python introduced namedtuples in version 2.6. A namedtuple is a tuple-like object where each element can be accessed by name, as well as by position. If a node will always contain a given type of namedtuple, Loman has a convenience method `add_named_tuple_expansion` which will create new nodes for each element of a namedtuple, using the naming convention **parent_node.tuple_element_name**. This can be useful for clarity when different downstream nodes depend on different parts of computation result:

```pycon
>>> Coordinate = namedtuple('Coordinate', ['x', 'y'])
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: Coordinate(a+1, a+2))
>>> comp.add_named_tuple_expansion('b', Coordinate)
>>> comp.add_node('c', lambda *args: sum(args), args=['b.x', 'b.y'])
>>> comp.compute_all()
>>> comp.get_value_dict()
{'a': 1, 'b': Coordinate(x=2, y=3), 'b.x': 2, 'b.y': 3, 'c': 5}
>>> comp.draw()
```

```{graphviz}
    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#9dff00" style=filled]
        n2 [label="b.x" fillcolor="#0343df" style=filled]
        n3 [label="b.y" fillcolor="#0343df" style=filled]
        n4 [label=c fillcolor="#0343df" style=filled]
            n0 -> n1
            n1 -> n2
            n1 -> n3
            n2 -> n4
            n3 -> n4
    }
```
