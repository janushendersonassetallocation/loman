# Viewing Computation Graphs

Loman supports viewing computations as graphs using the open source Graphviz graph visualization software.

By default, in a Jupyter notebook, when a graph is evaluated, it will show a graphical representation of its state:

```pycon
>>> from loman import *

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
n0 [fillcolor="#15b01a", label=a, style=filled];
n1 [fillcolor="#15b01a", label=b, style=filled];
n2 [fillcolor="#15b01a", label=c, style=filled];
n3 [fillcolor="#15b01a", label=d, style=filled];
n0 -> n1;
n0 -> n2;
n1 -> n3;
n2 -> n3;
}
```

This can also be achieved using the `.draw()` method, which allows some customization of drawing:
```pycon
>>> comp.draw()
```

```{graphviz}
digraph G {
n0 [fillcolor="#15b01a", label=a, style=filled];
n1 [fillcolor="#15b01a", label=b, style=filled];
n2 [fillcolor="#15b01a", label=c, style=filled];
n3 [fillcolor="#15b01a", label=d, style=filled];
n0 -> n1;
n0 -> n2;
n1 -> n3;
n2 -> n3;
}
```

And the `.view()` method will produce a graphical representation as a pdf,and display it in the default external pdf viewer - this can be useful to keep a view of the computation graph while making other changes in a notebook.

## State information

The following example has been contrived to show every state currently supported by Loman:
```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('e', lambda a: 1)
>>> comp.add_node('b', lambda a, doesnotexist: 1)
>>> comp.add_node('c', lambda a: a / 0.)
>>> comp.add_node('d', lambda b, c: b + c)
>>> comp.compute('c')
```

```{graphviz}
digraph G {
n0 [fillcolor="#15b01a", label=a, style=filled];
n1 [fillcolor="#0343df", label=b, style=filled];
n2 [fillcolor="#e50000", label=c, style=filled];
n3 [fillcolor="#ffff14", label=d, style=filled];
n4 [fillcolor="#9dff00", label=e, style=filled];
n5 [fillcolor="#f97306", label=doesnotexist, style=filled];
n0 -> n4;
n0 -> n1;
n0 -> n2;
n1 -> n3;
n2 -> n3;
n5 -> n1;
}
```

| Color        | State         | Meaning                                                                                                                                         |
|:-------------|:--------------|:------------------------------------------------------------------------------------------------------------------------------------------------| 
| Blue         | UNINITIALIZED | Node has been created, but does not contain a value                                                                                             |
| Green        | UPTODATE      | Node has been set/calculated and is up-to-date                                                                                                  |
| Yellow       | STALE         | Upstream nodes have been recalculated, so value is stale                                                                                        |
| Green-Yellow | COMPUTABLE    | Value is stale/does not exist, but all immediate parent nodes are up-to-date, so node can be recalculated immediately                           |
| Red          | ERROR         | An error was encountered while calculating this node. Value will contain the Exception object and a Traceback object.                           |
| Orange       | PLACEHOLDER   | This node was referenced by another node, but has not yet been defined, so a placeholder has been created, pending the definition of this node. |

## Showing timing information

Loman can also draw graph nodes to show timing information. In the following example, we deliberately create some nodes that are slow to calculate. We can the use `Computation.draw(colors='timing')` to show which nodes are slow to compute (green is fastest, red is slowest). This can be useful for diagnosing bottlenecks in complex computations.

```python
from loman import *
import time

@ComputationFactory
class ExampleComputation:
    a = input_node()

    @calc_node
    def b(self, a):
        time.sleep(1)
        return a + 1

    @calc_node
    def c(self, a):
        time.sleep(2)
        return 2 * a

    @calc_node
    def d(self, b, c):
        time.sleep(3)
        return b + c
```

```pycon
>>> comp = ExampleComputation()
>>> comp.insert('a', 3)
>>> comp.compute_all()
>>> comp.draw(colors='timing')
```

```{graphviz}
digraph G {
n0 [fillcolor="#FFFFFF", label=a, style=filled];
n1 [fillcolor="#15b01a", label=b, style=filled];
n2 [fillcolor="#fffe14", label=c, style=filled];
n3 [fillcolor="#e50000", label=d, style=filled];
n0 -> n1;
n0 -> n2;
n1 -> n3;
n2 -> n3;
}
```

## Styling

You can control how nodes are rendered using the `style` keyword of the `add_node` method or `calc_node` decorator. Two styles are currently supported: 'small' and 'dot'. This can be useful to de-emphasize nodes in complex computations:

```pycon
>>> comp = Computation()
>>> comp.add_node('a', value=1, style='small')
>>> comp.add_node('b', lambda a: a + 1, style='dot')
>>> comp.add_node('c', lambda a: 2 * a, style='dot')
>>> comp.add_node('d', lambda b, c: b + c)
>>> comp.compute_all()
>>> comp
```

```{graphviz}
digraph G {
n0 [fillcolor="#15b01a", fontsize=8, height=0.2, label=a, style=filled, width=0.3];
n1 [fillcolor="#15b01a", label=b, peripheries=1, shape=point, style=filled, width=0.1];
n2 [fillcolor="#15b01a", label=c, peripheries=1, shape=point, style=filled, width=0.1];
n3 [fillcolor="#15b01a", label=d, style=filled];
n0 -> n1;
n0 -> n2;
n1 -> n3;
n2 -> n3;
}
```

## Showing type information

You can ask loman to graphically show type information for nodes by calling `draw` with the keyword argument `shape='type'`. This example shows how different types are shown:

```pycon
>>> import numpy as np, pandas as pd
>>> comp = Computation()
>>> comp.add_node('scalar', value=1)
>>> comp.add_node('array', lambda scalar: np.array([scalar]))
>>> comp.add_node('dataframe', lambda scalar: pd.DataFrame([[1]], columns=['A']))
>>> comp.add_node('list', lambda scalar: [scalar])
>>> comp.add_node('tuple', lambda scalar: (scalar,))
>>> comp.add_node('dict', lambda scalar: {'a': scalar})
>>> comp.add_node('computation', lambda scalar: Computation())
>>> comp.add_node('other', lambda scalar: object())
>>> comp.compute_all()
>>> comp.draw(shapes='type')
```

```{graphviz}
digraph G {
n0 [fillcolor="#15b01a", label=scalar, shape=ellipse, style=filled];
n1 [fillcolor="#15b01a", label=array, shape=rect, style=filled];
n2 [fillcolor="#15b01a", label=dataframe, shape=box3d, style=filled];
n3 [fillcolor="#15b01a", label=list, peripheries=2, shape=ellipse, style=filled];
n4 [fillcolor="#15b01a", label=tuple, peripheries=2, shape=ellipse, style=filled];
n5 [fillcolor="#15b01a", label=dict, peripheries=2, shape=house, style=filled];
n6 [fillcolor="#15b01a", label=computation, shape=hexagon, style=filled];
n7 [fillcolor="#15b01a", label=other, shape=diamond, style=filled];
n0 -> n1;
n0 -> n2;
n0 -> n3;
n0 -> n4;
n0 -> n5;
n0 -> n6;
n0 -> n7;
}
```