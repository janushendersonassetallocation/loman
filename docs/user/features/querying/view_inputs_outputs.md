# Viewing node inputs and outputs

Loman computations contain methods to see what nodes are inputs to any node, and what nodes a given node is itself an input to. First, let's define a simple Computation:

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: a + 1)
>>> comp.add_node('c', lambda a: 2 * a)
>>> comp.add_node('d', lambda b, c: b + c)
>>> comp.compute_all()
>>> comp
```

```{graphviz}
    digraph G {
        n0       [fillcolor="#15b01a", label=a, style=filled];
        n1       [fillcolor="#15b01a", label=b, style=filled];
        n2       [fillcolor="#15b01a", label=c, style=filled];
        n3       [fillcolor="#15b01a", label=d, style=filled];
        n0 -> n1
        n0 -> n2
        n1 -> n3
        n2 -> n3
    }
```

We can find the inputs of a node using the `get_inputs` method, or the `i` attribute (which works similarly to the `v` and `s` attributes to access value and state):

```pycon
>>> comp.get_inputs('b')
['a']
>>> comp.get_inputs('d')
['c', 'b']
>>> comp.i.d # Attribute-style access
['c', 'b']
>>> comp.i['d'] # Dictionary-style access
['c', 'b']
>>> comp.i[['b', 'd']] # Multiple dictionary-style accesses:
[['a'], ['c', 'b]]
```

We can also use `get_original_inputs` to find the inputs of the entire Computation (or a subset of it):

```pycon
>>> comp.get_original_inputs()
['a']
>>> comp.get_original_inputs(['b']) # Just the inputs used to compute b
['a']
```

To find what a node feeds into, there are `get_outputs`, the `o` attribute and `get_final_outputs` (although as intermediate nodes are often useful, this latter is less useful):

```pycon
>>> comp.get_outputs('a')
['b', 'c']
>>> comp.o.a
['b', 'c']
>>> comp.o[['b', 'c']]
[['d'], ['d']]
>>> comp.get_final_outputs()
['d']
```

Finally, these can be used with the `v` accessor to quickly see all the input values to a given node:

```pycon
>>> {n: comp.v[n] for n in comp.i.d}
{'c': 2, 'b': 2}
```
