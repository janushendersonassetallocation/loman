# Loman

[![PyPI - Version](https://img.shields.io/pypi/v/loman.svg)](https://pypi.python.org/pypi/loman)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/loman.svg)](https://pypi.python.org/pypi/loman)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/loman.svg)](https://pypi.python.org/pypi/loman)
[![PyPI - License](https://img.shields.io/pypi/l/loman.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Github - Test Status](https://github.com/janushendersonassetallocation/loman/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/janushendersonassetallocation/loman/actions/workflows/ci.yml)
[![ReadTheDocs](https://readthedocs.org/projects/loman/badge/?version=latest)](http://loman.readthedocs.io/)

Loman tracks the state of your computations, and the dependencies between them, allowing full and partial recalculations.

## Example

```pycon
>>> comp = Computation()
>>> comp.add_node('a')
>>> comp.add_node('b', lambda a: a+1)
>>> comp.add_node('c', lambda a, b: 2*a)
>>> comp.add_node('d', lambda b, c: b + c)
>>> comp.add_node('e', lambda c: c + 1)
>>> comp.compute('d') # Will not compute e unnecessarily
>>> comp.get_value_dict() # Can see all the intermediates
{'a': 1, 'b': 2, 'c': 2, 'd': 4, 'e': None}
>>> comp.draw_graphviz() # Can quickly see what calculated
```

![Loman Graph Example](https://raw.githubusercontent.com/janusassetallocation/loman/master/docs/_static/example000.png)

For further examples, take a look at the [Quickstart](http://loman.readthedocs.io/en/latest/user/quickstart.html).

## Purpose

Loman makes it easy to ingest data from multiple sources, clean and integrate that data, and then use it to produce results for exporting to databases and other systems, as well as reports or dashboards for humans. Uses of Loman include:

- **Real-time systems**. Inputs to real-time systems frequently tick at different rates. Loman ensures that only what is necessary is recalculated. Furthermore, given some outputs that a slower to produce than others, Loman allows you to control which outputs are computed how frequently. Loman allows you to quickly show status of all items, and keep track of what needs to be updated.
- **Batch systems**. When used as part of a daily process, Loman can serialize some or all nodes of a computation graph, allowing for easy inspection of original inputs, intermediate calculations and tracebacks when failures occur. Original inputs can be replaced, or intermediate calculation methods or values overwritten in-place, allowing easy recovery from failures, without re-acquiring potentially expensive inputs or re-performing time-consuming calculations unnecessarily.
- **Research**. Loman allows you to keep track of complex dependencies as you create new calculation systems, or revisit old ones. Calculate new data, statistics and reports re-using existing raw inputs and calculated intermediates. Improve your productivity by ncreasing the frequency of iterations - make adjustments to methods in-place and re-run only what needs to be re-run.

The [Introduction](http://loman.readthedocs.io/en/latest/user/intro.html) section of the documentation has more details on why Loman might be useful for you.

## Installation

To install loman:

```bash
$ pip install loman
```

Or you can download from github: <https://github.com/janusassetallocation/loman>

## Documentation

Up-to-date and thorough documentation is available on ReadTheDocs at <http://loman.readthedocs.io/>

## Acknowledgments

This project uses configuration templates from [tschm/.config-templates](https://github.com/tschm/.config-templates) for development tooling and code quality standards.
