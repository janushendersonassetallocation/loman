# Serializing computations

Loman can serialize computations to disk using the dill package. This can be useful to have a system store the inputs, intermediates and results of a scheduled calculation for later inspection if required:

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: a + 1)
>>> comp.compute_all()
>>> comp.draw()
```

```{graphviz}
    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#15b01a" style=filled]
            n0 -> n1
    }
```

```pycon
>>> comp.to_dict()
{'a': 1, 'b': 2}
>>> comp.write_dill('foo.dill')
>>> comp2 = Computation.read_dill('foo.dill')
>>> comp2.draw()
```

```{graphviz}
    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#15b01a" style=filled]
            n0 -> n1
    }
```

    >>> comp.get_value_dict()
    {'a': 1, 'b': 2}

It is also possible to request that a particular node not be serialized, in which case it will have no value, and uninitialized state when it is deserialized. This can be useful where an object is not serializable, or where data is not licensed to be distributed:

```pycon
>>> comp.add_node('a', value=1, serialize=False)
>>> comp.compute_all()
>>> comp.write_dill('foo.dill')
>>> comp2 = Computation.read_dill('foo.dill')
>>> comp2.draw()
```

```{graphviz}
    digraph {
        n0 [label=a fillcolor="#0343df" style=filled]
        n1 [label=b fillcolor="#15b01a" style=filled]
            n0 -> n1
    }
```

:::{note}
The serialization format is not currently stabilized. While it is convenient to be able to inspect the results of previous calculations, this method should *not* be relied on for long-term storage.
:::
