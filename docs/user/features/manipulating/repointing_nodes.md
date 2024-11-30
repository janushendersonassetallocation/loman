# Repointing Nodes

It is possible to repoint existing nodes to a new node. This can be useful when it is desired to make a small change in one node, without having to recreate all descendant nodes. As an example:

```pycon
>>> from loman import *
>>> comp = Computation()
>>> comp.add_node('a', value = 2)
>>> comp.add_node('b', lambda a: a + 1)
>>> comp.add_node('c', lambda a: 10*a)
>>> comp.compute_all()
>>> comp.v.b
3
>>> comp.v.c
20
>>> comp.add_node('modified_a', lambda a: a*a)
>>> comp.compute_all()
>>> comp.v.a
2
>>> comp.v.modified_a
4
>>> comp.v.b
3
>>> comp.v.c
20
>>> comp.repoint('a', 'modified_a')
>>> comp.compute_all()
>>> comp.v.b
5
>>> comp.v.c
40
```
