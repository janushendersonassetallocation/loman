# Tagging Nodes

Nodes can be tagged with string tags, either when the node is added, using the `tags` parameter of `add_node`, or later, using the `set_tag` or `set_tags` methods, which can take a single node or a list of nodes:

```pycon
>>> from loman import *
>>> comp = Computation()
>>> comp.add_node('a', value=1, tags=['foo'])
>>> comp.add_node('b', lambda a: a + 1)
>>> comp.set_tag(['a', 'b'], 'bar')
```

:::{note}
Tags beginning and ending with double-underscores ("\_\_\[tag\]\_\_") are reserved for internal use by Loman.
:::

The tags associated with a node can be inspected using the `tags` method, or the `t` attribute-style accessor:

```pycon
>>> comp.tags('a')
{'__serialize__', 'bar', 'foo'}
>>> comp.t.b
{'__serialize__', 'bar'}
```

Tags can also be cleared with the `clear_tag` and `clear_tags` methods:

```pycon
>>> comp.clear_tag(['a', 'b'], 'foo')
>>> comp.t.a
{'__serialize__', 'bar'}
```

By design, no error is thrown if a tag is added to a node that already has that tag, nor if a tag is cleared from a node that does not have that tag.

In future, it is intended it will be possible to control graph drawing and calculation using tags (for example, by requesting that only nodes with or without certain tags are rendered or calculated).
