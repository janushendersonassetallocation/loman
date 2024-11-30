# Showing a computation as a DataFrame

`Computation` objects have a method `to_df()` which allows them to be shown as a DataFrame. This provides a quick summary of the states and values of each node, as well as useful timing information: 

```pycon
>>> from loman import *

>>> comp = Computation()
>>> comp.add_node('a', value=1)
>>> comp.add_node('b', lambda a: a + 1)
>>> comp.add_node('c', lambda a: 2 * a)
>>> comp.add_node('d', lambda b, c: b + c)
>>> comp.compute_all()

>>> comp.to_df()
```

|    | state           |   value | start                      | end                        |   duration |
|:---|:----------------|--------:|:---------------------------|:---------------------------|-----------:|
| a  | States.UPTODATE |       1 | NaT                        | NaT                        |        nan |
| b  | States.UPTODATE |       2 | 2024-11-30 18:49:41.626849 | 2024-11-30 18:49:41.626849 |          0 |
| c  | States.UPTODATE |       2 | 2024-11-30 18:49:41.626849 | 2024-11-30 18:49:41.626849 |          0 |
| d  | States.UPTODATE |       4 | 2024-11-30 18:49:41.626849 | 2024-11-30 18:49:41.626849 |          0 |

:::{tip}
If your values are not scalars, it can be useful to drop the value column.
```pycon
>>> comp.to_df().drop(columns='value')
```
:::