# `@ComputationFactory`: Define computations with classes  

Loman provides a decorator `@ComputationFactory`, which allows complete computations to be specified as a single class. This is often a more convenient way to define classes, providing better encapsulation, and making them easier to read later.

Within a computation factory class, we can use `input_node` to declare input nodes, and `calc_node` to define functions used to calculate nodes

```pycon
>>> @ComputationFactory
... class ExampleComputation:
...     a = input_node()
... 
...     @calc_node
...     def b(a):
...         return a + 1
... 
...     @calc_node
...     def c(a):
...         return 2 * a
... 
...     @calc_node
...     def d(b, c):
...         return b + c
```

Once the computation factory is defined, we can use is to instantiate computations, which we can then use just like regular loman Computations. 

```
>>> comp = ExampleComputation()
>>> comp.insert('a', 3)
>>> comp.compute_all()
>>> comp.v.d
10
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

## Using `self`

In the example above, we used functions defined exactly as we have in previous articles. Since many IDEs expect the first parameter of a function defined in a class to be 'self', loman supports this. If the first parameter of a calc_node function defined within a ComputationFactory is 'self', then it will not refer to a 'self' node of the computation, but instead will allow access to non-calc_node methods defined within the class. For example, this class acts exactly the same as the previous example, with the `d` node using the `custom_add` method to perform addition:

```python
@ComputationFactory
class ExampleComputation2:
    a = input_node()

    @calc_node
    def b(self, a):
        return a + 1

    @calc_node
    def c(self, a):
        return 2 * a

    def custom_add(self, x, y):
        return x + y
    
    @calc_node
    def d(self, b, c):
        return self.custom_add(x, y)
```

If this behavior for 'self' is not required, it can be disabled at the class level or for individual nodes by providing the kwarg `ignore_self=False`.

## Providing optional arguments through `@calc_node`

Arguments provided to the `@calc_node` are passed through to `add_node`, and can be used to control node creation, argument mapping, styling, etc.

```pycon
@ComputationFactory
class ExampleComputation3:
    a = input_node()

    @calc_node(style='dot', tags=['tag'])
    def b(self, a):
        return a + 1

    @calc_node(kwds={'x': 'a'}, style='dot')
    def c(self, x):
        return 2 * x

    @calc_node(serialize=False)
    def d(self, b, c):
        return b + c

>>> comp = ExampleComputation3()
>>> comp.insert('a', 3)
>>> comp.compute_all()
>>> comp
```

```{graphviz}
digraph G {
n0 [fillcolor="#15b01a", label=a, style=filled];
n1 [fillcolor="#15b01a", label=b, peripheries=1, shape=point, style=filled, width=0.1];
n2 [fillcolor="#15b01a", label=c, peripheries=1, shape=point, style=filled, width=0.1];
n3 [fillcolor="#15b01a", label=d, style=filled];
n0 -> n1;
n0 -> n2;
n1 -> n3;
n2 -> n3;
}
```