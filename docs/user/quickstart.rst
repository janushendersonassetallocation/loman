Quick Start
===========

In Loman, a computation is represented as a set of nodes. Each node can be either an input node, which must be provided, or a calculation node which can be calculated from input nodes or other calculation nodes.

Creating and Running a Computation
----------------------------------

Let's start by creating a computation object and adding a couple of nodes to it::

    >>> comp = Computation()
    >>> comp.add_node('a')
    >>> comp.add_node('b', lambda a: a + 1)
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=a fillcolor="#0343df" style=filled]
        n1 [label=b fillcolor="#0343df" style=filled]
            n0 -> n1
    }

Loman gives us a quick and easy way to visualize our computations. The graph above shows us that node **b** depends on node **a**. Both are colored blue as neither has a value. Let's insert a value into node **a**::

    >>> comp.insert('a', 1)
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#9dff00" style=filled]
            n0 -> n1
    }

Now we see that node **a** is colored dark green, indicating that it is up-to-date, since we just inserted a value. Node **b** is colored light green, indicating it is computable - that is to say that it is not up-to-date, but it can immediately be calculated. Let's do that::

    >>> comp.compute_all()

.. graphviz::

    digraph {
        n0 [label=a fillcolor="#15b01a" style=filled]
        n1 [label=b fillcolor="#15b01a" style=filled]
            n0 -> n1
    }

Now **b** is up-to-date, and is also colored dark green.

Inspecting Nodes
----------------

Loman gives us several ways of inspecting nodes. We can use the ``value`` and ``state`` methods::

    >>> comp.value('b')
    2
    >>> comp.state('b')
    <States.UPTODATE: 4>

The []-operator provides both the state and value::

    >>> comp['b']
    NodeData(state=<States.UPTODATE: 4>, value=2)

There are also methods ``get_value_dict()`` and ``get_df()`` which get the values of all the nodes::

    >>> comp.get_value_dict()
    {'a': 1, 'b': 2}
    >>> comp.get_df()
                 state  value  is_expansion
    a  States.UPTODATE      1           NaN
    b  States.UPTODATE      2           NaN

More Ways to Define Nodes
-------------------------

In our first example, we used a lambda expression to provide a function to calculate **b**. We can also provide a named function. The name of the function is unimportant. However, the names of the function parameters will be used to determine which nodes should supply inputs to the function::

    >>> comp = Computation()
    >>> comp.add_node('input_node')
    >>> def foo(input_node):
    ...   return input_node + 1
    ...
    >>> comp.add_node('result_node', foo)
    >>> comp.insert('input_node', 1)
    >>> comp.compute_all()
    >>> comp.value('result_node')
    2

We can explicitly specify the mapping from parameter names to node names if we require, using the ``kwds`` parameter. And a node can depend on more than one input node. Here we have  a function of two parameters. The argument to ``kwds`` can be read as saying "Parameter **a** comes from node **x**, parameter **b** comes from node **y**"::

    >>> comp = Computation()
    >>> comp.add_node('x')
    >>> comp.add_node('y')
    >>> def add(a, b):
    ...   return a + b
    ...
    >>> comp.add_node('result', add, kwds={'a': 'x', 'b': 'y'})
    >>> comp.insert('x', 20)
    >>> comp.insert('y', 22)
    >>> comp.compute_all()
    >>> comp.value('result')
    42

For input nodes, the ``add_node`` method can optionally take a value, rather than having to separately call the insert method::

    >>> comp = Computation()
    >>> comp.add_node('a', value=1)
    >>> comp.add_node('b', lambda a: a + 1)
    >>> comp.compute_all()
    >>> comp.value('result')
    2

Finally, the function supplied to **add_node** can have ``\*args`` or ``\*\*kwargs`` arguments. When this is done, the ``args`` and ``kwds`` provided to **add_node** control what will be placed in ``\*args`` or ``\*\*kwargs``::

    >>> comp = Computation()
    >>> comp.add_node('x', value=1)
    >>> comp.add_node('y', value=2)
    >>> comp.add_node('z', value=3)
    >>> comp.add_node('args', lambda *args: args, args=['x', 'y', 'z'])
    >>> comp.add_node('kwargs', lambda **kwargs: kwargs, kwds={'a': 'x', 'b': 'y', 'c': 'z'})
    >>> comp.compute_all()
    >>> comp.value('args')
    (1, 2, 3)
    >>> comp.value('kwargs')
    {'a': 1, 'b': 2, 'c': 3}

Controlling Computation
-----------------------

For these examples, we define a more complex Computation::

    >>> comp = Computation()
    >>> comp.add_node('input1')
    >>> comp.add_node('input2')
    >>> comp.add_node('intermediate1', lambda input1: 2 * input1)
    >>> comp.add_node('intermediate2', lambda input1, input2: input1 + input2)
    >>> comp.add_node('intermediate3', lambda input2: 3 * input2)
    >>> comp.add_node('result1', lambda intermediate1, intermediate2: intermediate1 + intermediate2)
    >>> comp.add_node('result2', lambda intermediate2, intermediate3: intermediate2 + intermediate3)
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#0343df" style=filled]
        n1 [label=input2 fillcolor="#0343df" style=filled]
        n2 [label=intermediate1 fillcolor="#0343df" style=filled]
        n3 [label=intermediate2 fillcolor="#0343df" style=filled]
        n4 [label=intermediate3 fillcolor="#0343df" style=filled]
        n5 [label=result1 fillcolor="#0343df" style=filled]
        n6 [label=result2 fillcolor="#0343df" style=filled]
            n0 -> n2
            n0 -> n3
            n1 -> n3
            n1 -> n4
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
    }

We insert values into **input1** and **input2**::

    >>> comp.insert('input1, 1)
    >>> comp.insert('input2', 2)
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#9dff00" style=filled]
        n3 [label=intermediate2 fillcolor="#9dff00" style=filled]
        n4 [label=intermediate3 fillcolor="#9dff00" style=filled]
        n5 [label=result1 fillcolor="#ffff14" style=filled]
        n6 [label=result2 fillcolor="#ffff14" style=filled]
            n0 -> n2
            n0 -> n3
            n1 -> n3
            n1 -> n4
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
    }

As before, we see that the nodes we have just inserted data for are colored dark green, indicating they are up-to-date. The intermediate nodes are all colored light green, to indicate that they are computable - that is that their immediate upstream nodes are all up-to-date, and so any one of them can be immediately calculated. The result nodes are colored yellow. This means that they are stale - they are not up-to-date, and they cannot be immediately calculated without first calculating some nodes that they depend on.

We saw before that we can use the ``compute_all`` method to calculate nodes. We can also specify exactly which nodes we would like calculated using the ``compute`` method. This method will calculate any upstream dependencies that are not up-to-date, but it will not calculate nodes that do not need to be calculated. For example, if we request the **result1** be calculated, **intermediate1** and **intermedate2** will be calculated first, but **intermediate3** and **result2** will not be calculated::

    >>> comp.compute('result1')
    >>> comp.value('result1')
    5
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#15b01a" style=filled]
        n3 [label=intermediate2 fillcolor="#15b01a" style=filled]
        n4 [label=intermediate3 fillcolor="#9dff00" style=filled]
        n5 [label=result1 fillcolor="#15b01a" style=filled]
        n6 [label=result2 fillcolor="#ffff14" style=filled]
            n0 -> n2
            n0 -> n3
            n1 -> n3
            n1 -> n4
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
    }

Inserting new data
------------------

Often, in real-time systems, updates will come periodically for one or more of the inputs to a computation. We can insert this updated data into a computation and Loman will corresponding mark any downstream nodes as stale or computable i.e. no longer up-to-date. Continuing from the previous example, we insert a new value into **input1**::

    >>> comp.insert('input1', 2)
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#9dff00" style=filled]
        n3 [label=intermediate2 fillcolor="#9dff00" style=filled]
        n4 [label=intermediate3 fillcolor="#9dff00" style=filled]
        n5 [label=result1 fillcolor="#ffff14" style=filled]
        n6 [label=result2 fillcolor="#ffff14" style=filled]
            n0 -> n2
            n0 -> n3
            n1 -> n3
            n1 -> n4
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
    }

And again we can ask Loman to calculate nodes in the computation, and give us results. Here we calculate all nodes::

    >>> comp.compute_all()
    >>> comp.value('result1')
    8

Overriding calculation nodes
----------------------------

In fact, we are not restricted to inserting data into input nodes. It is perfectly possible to use the ``insert`` method to override the value of a calculated node also. The overridden value will remain in place until the node is recalculated (which will happen after one of its upstreams is updated causing it to be marked stale, or when it is explicitly marked as stale, and then recalculated). Here we override **intermediate2** and calculate **result2** (note that **result1** is not recalculated, because we didn't ask anything that required it to be)::

    >>> comp.insert('intermediate2', 100)
    >>> comp.compute('result2')
    106
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#15b01a" style=filled]
        n3 [label=intermediate2 fillcolor="#15b01a" style=filled]
        n4 [label=intermediate3 fillcolor="#15b01a" style=filled]
        n5 [label=result1 fillcolor="#9dff00" style=filled]
        n6 [label=result2 fillcolor="#15b01a" style=filled]
            n0 -> n2
            n0 -> n3
            n1 -> n3
            n1 -> n4
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
    }

Changing calculations
---------------------

As well as inserting data into nodes, we can update the computation they perform by re-adding the node. Node states get updated appropriately automatically. For example, continuing from the previous example, we can change how **intermediate2** is calculated, and we see that nodes **intermediate2**, **result1** and **result2** are no longer marked up-to-date::

    >>> comp.add_node('intermediate2', lambda input1, input2: 5 * input1 + 2 * input2)
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#15b01a" style=filled]
        n3 [label=intermediate2 fillcolor="#9dff00" style=filled]
        n4 [label=intermediate3 fillcolor="#15b01a" style=filled]
        n5 [label=result1 fillcolor="#ffff14" style=filled]
        n6 [label=result2 fillcolor="#ffff14" style=filled]
            n0 -> n2
            n0 -> n3
            n1 -> n4
            n1 -> n3
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
    }

::

    >>> comp.compute_all()
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#15b01a" style=filled]
        n3 [label=intermediate2 fillcolor="#15b01a" style=filled]
        n4 [label=intermediate3 fillcolor="#15b01a" style=filled]
        n5 [label=result1 fillcolor="#15b01a" style=filled]
        n6 [label=result2 fillcolor="#15b01a" style=filled]
            n0 -> n2
            n0 -> n3
            n1 -> n4
            n1 -> n3
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
    }

::

    >>> comp.value('result1')
    18
    >>> comp.value('result2')
    20

Adding new nodes
----------------

We can even add new nodes, and change the dependencies of existing calculations. So for example, we can create a new node called **new_node**, and have **intermediate2** depend on that, rather than **input1**. It's confusing when I describe it with words, but Loman's visualization helps us keep tabs on everything - that's its purpose::

    >>> comp.add_node('new_node', lambda input1, input2: input1 / input2)
    >>> comp.add_node('intermediate2', lambda new_nod, input2: 5 * new_nod + 2 * input2)
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#15b01a" style=filled]
        n3 [label=intermediate2 fillcolor="#0343df" style=filled]
        n4 [label=intermediate3 fillcolor="#15b01a" style=filled]
        n5 [label=result1 fillcolor="#ffff14" style=filled]
        n6 [label=result2 fillcolor="#ffff14" style=filled]
        n7 [label=new_node fillcolor="#9dff00" style=filled]
            n0 -> n2
            n0 -> n7
            n1 -> n4
            n1 -> n7
            n1 -> n3
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
            n7 -> n3
    }

::

    >>> comp.compute_all()
    >>> comp.draw_graphviz()

.. graphviz::

    digraph {
        n0 [label=input1 fillcolor="#15b01a" style=filled]
        n1 [label=input2 fillcolor="#15b01a" style=filled]
        n2 [label=intermediate1 fillcolor="#15b01a" style=filled]
        n3 [label=intermediate2 fillcolor="#15b01a" style=filled]
        n4 [label=intermediate3 fillcolor="#15b01a" style=filled]
        n5 [label=result1 fillcolor="#15b01a" style=filled]
        n6 [label=result2 fillcolor="#15b01a" style=filled]
        n7 [label=new_node fillcolor="#15b01a" style=filled]
            n0 -> n2
            n0 -> n7
            n1 -> n4
            n1 -> n7
            n1 -> n3
            n2 -> n5
            n3 -> n5
            n3 -> n6
            n4 -> n6
            n7 -> n3
    }

::

    >>> comp.value('result1')
    13.0
    >>> comp.value('result2')
    15.0

