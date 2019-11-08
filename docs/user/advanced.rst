Advanced Features
=================

Viewing node inputs and outputs
-------------------------------

Loman computations contain methods to see what nodes are inputs to any node, and what nodes a given node is itself an input to. First, let's define a simple Computation:

    >>> comp = Computation()
    >>> comp.add_node('a', value=1)
    >>> comp.add_node('b', lambda a: a + 1)
    >>> comp.add_node('c', lambda a: 2 * a)
    >>> comp.add_node('d', lambda b, c: b + c)
    >>> comp.compute_all()
    >>> comp

.. graphviz::

    digraph G {
        n0	 [fillcolor="#15b01a", label=a, style=filled];
        n1	 [fillcolor="#15b01a", label=b, style=filled];
        n2	 [fillcolor="#15b01a", label=c, style=filled];
        n3	 [fillcolor="#15b01a", label=d, style=filled];
        n0 -> n1
        n0 -> n2
        n1 -> n3
        n2 -> n3
    }

We can find the inputs of a node using the ``get_inputs`` method, or the ``i`` attribute (which works similarly to the ``v`` and ``s`` attributes to access value and state)::

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

We can also use ``get_original_inputs`` to find the inputs of the entire Computation (or a subset of it)::

    >>> comp.get_original_inputs()
    ['a']
    >>> comp.get_original_inputs(['b']) # Just the inputs used to compute b
    ['a']

To find what a node feeds into, there are ``get_outputs``, the ``o`` attribute and ``get_final_outputs`` (although as intermediate nodes are often useful, this latter is less useful)::

    >>> comp.get_outputs('a')
    ['b', 'c']
    >>> comp.o.a
    ['b', 'c']
    >>> comp.o[['b', 'c']]
    [['d'], ['d']]
    >>> comp.get_final_outputs()
    ['d']

Finally, these can be used with the ``v`` accessor to quickly see all the input values to a given node::

    >>> {n: comp.v[n] for n in comp.i.d}
    {'c': 2, 'b': 2}



Constant Values
---------------

When you are using a pre-existing function for a node, and one or more of the parameters takes a constant value, one way is to define a lambda, which fixes the parameter value. For example, below we use a lambda to fix the second parameter passed to the add function::

    >>> def add(x, y):
    ...    return x + y

    >>> comp = Computation()
    >>> comp.add_node('a', value=1)
    >>> comp.add_node('b', lambda a: add(a, 1))
    >>> comp.compute_all()
    >>> comp.v.b
    2

However providing ``ConstantValue`` objects to the ``args`` or ``kwds`` parameters of ``add_node``, make this simpler. ``C`` is an alias for ``ConstantValue``, and in the example below, we use that to tell node ``b`` to calculate by taking parameter ``x`` from node ``a``, and ``y`` as a constant, ``1``::

    >>> comp = Computation()
    >>> comp.add_node('a', value=1)
    >>> comp.add_node('b', add, kwds={"x": "a", "y": C(1)})
    >>> comp.compute_all()
    >>> comp.v.b
    2

Interactive Debugging
---------------------

As shown in the quickstart section "Error-handling", loman makes it easy to see a traceback for any exceptions that are shown while calculating nodes, and also makes it easy to update calculation functions in-place to fix errors. However, it is often desirable to use Python's interactive debugger at the exact time that an error occurs. To support this, the ``calculate`` method takes a parameter ``raise_exceptions``. When it is ``False`` (the default), nodes are set to state ERROR when exceptions occur during their calculation. When it is set to ``True`` any exceptions are not caught, allowing the user to invoke the interactive debugger

.. code:: python

    comp = Computation()
    comp.add_node('numerator', value=1)
    comp.add_node('divisor', value=0)
    comp.add_node('result', lambda numerator, divisor: numerator / divisor)
    comp.compute('result', raise_exceptions=True)


::


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-38-4243c7243fc5> in <module>()
    ----> 1 comp.compute('result', raise_exceptions=True)


    [... skipped ...]


    <ipython-input-36-c0efbf5b74f7> in <lambda>(numerator, divisor)
          3 comp.add_node('numerator', value=1)
          4 comp.add_node('divisor', value=0)
    ----> 5 comp.add_node('result', lambda numerator, divisor: numerator / divisor)


    ZeroDivisionError: division by zero


.. code:: python

    %debug


.. parsed-literal::

    > <ipython-input-36-c0efbf5b74f7>(5)<lambda>()
          1 from loman import *
          2 comp = Computation()
          3 comp.add_node('numerator', value=1)
          4 comp.add_node('divisor', value=0)
    ----> 5 comp.add_node('result', lambda numerator, divisor: numerator / divisor)

    ipdb> p numerator
    1
    ipdb> p divisor
    0

Creating Nodes Using a Decorator
--------------------------------

Loman provide a decorator ``@node``, which allows functions to be added to computations. The first parameter is the Computation object to add a node to. By default, it will take the node name from the function, and the names of input nodes from the names of the parameter of the function, but any parameters provided are passed through to ``add_node``, including name::

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

.. graphviz::

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

Tagging Nodes
-------------

Nodes can be tagged with string tags, either when the node is added, using the ``tags`` parameter of ``add_node``, or later, using the ``set_tag`` or ``set_tags`` methods, which can take a single node or a list of nodes::

    >>> from loman import *
    >>> comp = Computation()
    >>> comp.add_node('a', value=1, tags=['foo'])
    >>> comp.add_node('b', lambda a: a + 1)
    >>> comp.set_tag(['a', 'b'], 'bar')

.. note:: Tags beginning and ending with double-underscores ("__[tag]__") are reserved for internal use by Loman.

The tags associated with a node can be inspected using the ``tags`` method, or the ``t`` attribute-style accessor::

    >>> comp.tags('a')
    {'__serialize__', 'bar', 'foo'}
    >>> comp.t.b
    {'__serialize__', 'bar'}

Tags can also be cleared with the ``clear_tag`` and ``clear_tags`` methods::

    >>> comp.clear_tag(['a', 'b'], 'foo')
    >>> comp.t.a
    {'__serialize__', 'bar'}

By design, no error is thrown if a tag is added to a node that already has that tag, nor if a tag is cleared from a node that does not have that tag.

In future, it is intended it will be possible to control graph drawing and calculation using tags (for example, by requesting that only nodes with or without certain tags are rendered or calculated).

Automatically expanding named tuples
------------------------------------

Often, a calculation will return more than one result. For example, a numerical solver may return the best solution it found, along with a status indicating whether the solver converged. Python introduced namedtuples in version 2.6. A namedtuple is a tuple-like object where each element can be accessed by name, as well as by position. If a node will always contain a given type of namedtuple, Loman has a convenience method ``add_named_tuple_expansion`` which will create new nodes for each element of a namedtuple, using the naming convention **parent_node.tuple_element_name**. This can be useful for clarity when different downstream nodes depend on different parts of computation result::

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

.. graphviz::

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

Serializing computations
------------------------

Loman can serialize computations to disk using the dill package. This can be useful to have a system store the inputs, intermediates and results of a scheduled calculation for later inspection if required::

    >>> comp = Computation()
    >>> comp.add_node('a', value=1)
    >>> comp.add_node('b', lambda a: a + 1)
    >>> comp.compute_all()
    >>> comp.draw()

.. graphviz::

    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#15b01a" style=filled]
            n0 -> n1
    }

::

    >>> comp.to_dict()
    {'a': 1, 'b': 2}
    >>> comp.write_dill('foo.dill')
    >>> comp2 = Computation.read_dill('foo.dill')
    >>> comp2.draw()

.. graphviz::

    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#15b01a" style=filled]
            n0 -> n1
    }

::

    >>> comp.get_value_dict()
    {'a': 1, 'b': 2}

It is also possible to request that a particular node not be serialized, in which case it will have no value, and uninitialized state when it is deserialized. This can be useful where an object is not serializable, or where data is not licensed to be distributed::

    >>> comp.add_node('a', value=1, serialize=False)
    >>> comp.compute_all()
    >>> comp.write_dill('foo.dill')
    >>> comp2 = Computation.read_dill('foo.dill')
    >>> comp2.draw()

.. graphviz::

    digraph {
        n0 [label=a fillcolor="#0343df" style=filled]
        n1 [label=b fillcolor="#15b01a" style=filled]
            n0 -> n1
    }

.. note:: The serialization format is not currently stabilized. While it is convenient to be able to inspect the results of previous calculations, this method should *not* be relied on for long-term storage.

Non-string node names
---------------------

In the previous example, the nodes have all been given strings as keys. This is not a requirement, and in fact any object that could be used as a key in a dictionary can be a key for a node. As function parameters can only be strings, we have to rely on the ``kwds`` argument to ``add_node`` to specify which nodes should be used as inputs for calculation nodes' functions. For a simple but frivolous example, we can represent a finite part of the Fibonacci sequence using tuples of the form ``('fib', [int])`` as keys::

    >>> comp = Computation()
    >>> comp.add_node(('fib', 1), value=1)
    >>> comp.add_node(('fib', 2), value=1)
    >>> for i in range(3,7):
    ...    comp.add_node(('fib', i), lambda x, y: x + y, kwds={'x': ('fib', i - 1), 'y': ('fib', i - 2)})
    ...
    >>> comp.draw()

.. graphviz::

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

::

    >>> comp.compute_all()
    >>> comp.value(('fib', 6))
    8
    
Repointing Nodes
----------------

It is possible to repoint existing nodes to a new node. This can be useful when it is desired to make a small change in one node, without having to recreate all descendant nodes. As an example: 

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
