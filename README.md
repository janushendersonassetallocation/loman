
<div align="center">

# Loman

### Manage complex calculation flow via robust DAG-based dependency logic in Python

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/loman.svg)](https://pypi.python.org/pypi/loman)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/loman.svg)](https://pypi.python.org/pypi/loman)
[![ReadTheDocs](https://readthedocs.org/projects/loman/badge/?version=latest)](http://loman.readthedocs.io/)

![Github](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=white)
![MAC OS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![CodeFactor](https://www.codefactor.io/repository/github/janushendersonassetallocation/loman/badge)](https://www.codefactor.io/repository/github/janushendersonassetallocation/loman)

[![CI](https://github.com/janushendersonassetallocation/loman/actions/workflows/rhiza_ci.yml/badge.svg?event=push)](https://github.com/janushendersonassetallocation/loman/actions/workflows/rhiza_ci.yml)
[![MARIMO](https://github.com/janushendersonassetallocation/loman/actions/workflows/rhiza_marimo.yml/badge.svg?event=push)](https://github.com/janushendersonassetallocation/loman/actions/workflows/rhiza_marimo.yml)
[![DEVCONTAINER](https://github.com/janushendersonassetallocation/loman/actions/workflows/rhiza_devcontainer.yml/badge.svg?event=push)](https://github.com/janushendersonassetallocation/loman/actions/workflows/rhiza_devcontainer.yml)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/janusassetallocation/loman)

**Smart computation management** - Only recalculate what's necessary

</div>

Loman tracks the state of your computations and their dependencies, enabling intelligent partial recalculations that save time and computational resources. Perfect for data pipelines, real-time systems, and complex analytical workflows.

## Table of Contents

- ✨ [Features](#-features)
- 📥 [Installation](#-installation)
- 🛠️ [Development](#️-development)
- 🚀 [Quick Start](#-quick-start)
- 📖 [Documentation](#-documentation)
- ☁️ [Codespaces](#️-try-loman-in-codespaces)
- 👥 [Contributing](#-contributing)

## ✨ Features

**Build smarter computation graphs with automatic dependency tracking**

- 🔄 **Smart Recalculation**: Only compute what's changed - save time and resources
- 📊 **Dependency Tracking**: Automatic graph-based dependency management
- 🎯 **Selective Updates**: Control which outputs to compute and when
- 💾 **State Persistence**: Serialize computation graphs for inspection and recovery
- 🔍 **Visual Debugging**: Built-in GraphViz integration for computation visualization
- ⚡ **Real-time Ready**: Perfect for systems with inputs that tick at different rates
- 🧪 **Research Friendly**: Iterate quickly by updating methods and re-running only necessary computations
- 🔗 **Framework Agnostic**: Works with NumPy, Pandas, and your favorite Python libraries

## 🚀 Quick Start

![Loman Graph Example](https://raw.githubusercontent.com/janusassetallocation/loman/master/docs/_static/example000.png)

The above computation graph was generated with this simple Loman code:

```python
from loman import Computation

# Create a computation graph
comp = Computation()
comp.add_node('a', value=1)                    # Input node
comp.add_node('b', lambda a: a + 1)            # b depends on a
comp.add_node('c', lambda a, b: 2 * a)         # c depends on a and b
comp.add_node('d', lambda b, c: b + c)         # d depends on b and c
comp.add_node('e', lambda c: c + 1)            # e depends on c

# Smart computation - only calculates what's needed
comp.compute('d')  # Will not compute 'e' unnecessarily!

# Inspect intermediate values
comp.to_dict()
# {'a': 1, 'b': 2, 'c': 2, 'd': 4, 'e': None}

# Visualize the computation graph
comp.draw()  # Creates the graph shown above
```

```result

```


📚 **Explore More**: Check out our [Interactive Examples](examples/) including:
- 💰 Interest Rate Swap Pricing
- 📈 Portfolio Valuation
- 🏦 CDO Modeling

## Why Loman?

Loman transforms how you build and maintain complex data processing pipelines by making computational dependencies explicit and manageable. Instead of manually tracking what needs to be recalculated when data changes, Loman handles this automatically.

### Perfect for:

- 🔴 **Real-time Systems**: Handle inputs that tick at different rates efficiently
  - Only recalculate affected downstream computations
  - Control which outputs are computed and when
  - Monitor status of all components in your pipeline

- 📦 **Batch Processing**: Build robust daily/periodic processes
  - Serialize computation state for easy inspection and debugging
  - Replace inputs or override intermediate values without full recomputation
  - Recover from failures efficiently by resuming from last good state

- 🔬 **Research & Analytics**: Accelerate iterative development
  - Modify calculation methods and re-run only what's needed
  - Track complex dependencies in evolving analysis pipelines
  - Reuse expensive intermediate computations across experiments

📖 **Learn More**: The [Introduction](http://loman.readthedocs.io/en/latest/user/intro.html) explains in detail how Loman can transform your workflow.

## 📥 Installation

### Using pip (recommended)

```bash
pip install loman
```

### From source (development)

```bash
git clone https://github.com/janusassetallocation/loman.git
cd loman
pip install -e .
```

**Note**: Use `-e` flag for editable installation during development

## 🛠️ Development

**For contributors and advanced users**

Loman uses modern Python development tools for a smooth developer experience:

```bash
# 📦 Install development dependencies
make install

# 🧪 Run tests with coverage
make test

# ✨ Format and lint code
make check
```

### Development Tools

- **Testing**: pytest with coverage reporting
- **Formatting**: ruff for code formatting and linting
- **Task Management**: Taskfile for build automation
- **Quality**: Pre-commit hooks for code quality

## 📖 Documentation

- 📚 [Complete Documentation](http://loman.readthedocs.io/): Comprehensive guides and API reference
- 🚀 [Quickstart Guide](http://loman.readthedocs.io/en/latest/user/quickstart.html): Get up and running in minutes
- 💡 [User Guide](http://loman.readthedocs.io/en/latest/user/intro.html): In-depth concepts and strategies
- 📊 [Interactive Examples](examples/): Real-world financial modeling examples
- 🔧 [API Reference](http://loman.readthedocs.io/en/latest/api.html): Complete function and class documentation

## ☁️ Try Loman in Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/janusassetallocation/loman?quickstart=1)

**Instant development environment in your browser** Click the button to launch a fully configured development environment in your browser.

### What's Included

- 🐍 **Python 3.13** - Latest Python with all dependencies pre-installed
- 🛠️ **Development Tools** - pytest, ruff, mypy, pre-commit hooks ready to go
- 📓 **Interactive Notebooks** - Marimo notebooks for exploring examples
- ⚡ **Zero Setup** - Everything configured ready to go
- 🎯 **VS Code Extensions** - Python, testing, and productivity extensions pre-installed


## 👥 Contributing

We welcome contributions! 🎉

Whether you're fixing bugs, adding features, or improving documentation, your help makes Loman better for everyone.

### Quick Start for Contributors

1. 🍴 Fork the repository
2. 🌿 Create your feature branch: `git checkout -b feature/amazing-feature`
3. ✨ Make your changes and add tests
4. 🧪 Test your changes: `make test`
5. 📝 Commit your changes: `git commit -m 'Add amazing feature'`
6. 🚀 Push to your branch: `git push origin feature/amazing-feature`
7. 🎯 Open a Pull Request

### Resources

- 📋 [Contributing Guide](CONTRIBUTING.md)
- 🤝 [Code of Conduct](CODE_OF_CONDUCT.md)
- 🐛 [Issue Tracker](https://github.com/janusassetallocation/loman/issues)

### Acknowledgments 🙏

- [Jebel-Quant/rhiza](https://github.com/Jebel-Quant/rhiza) for standardised CI/CD templates and project tooling

---

<div align="center">

⭐ **If you find Loman useful, please consider giving it a star!** ⭐

</div>
