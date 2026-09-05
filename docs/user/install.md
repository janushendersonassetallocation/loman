# Installation Guide

## Requirements

Loman needs **Python 3.11 or newer**. Every release is tested against 3.11, 3.12, 3.13
and 3.14 on Linux, macOS and Windows.

## Installing Loman

### With uv

[uv](https://docs.astral.sh/uv/) is the fastest way to install Loman, and the tool this
project itself is built with. To add Loman to a project:

```bash
$ uv add loman
```

To install it into the environment you already have:

```bash
$ uv pip install loman
```

To try it without installing anything permanently — uv builds a throwaway environment and
discards it afterwards:

```bash
$ uv run --with loman python
```

### With pip

```bash
$ pip install loman
```

If you don't have [pip](https://pip.pypa.io) installed,
[this Python installation guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)
will get you started.

### With conda

Loman is published on PyPI rather than conda-forge, so install it with pip inside your
conda environment:

```bash
$ conda create -n loman python=3.12
$ conda activate loman
$ pip install loman
```

Graphviz, however, *is* on conda-forge, which makes conda a convenient way to get it —
see below.

## Optional extras

Two features are packaged as extras, so a plain install stays small. Neither is needed for
the core computation engine, and nothing in a bare `import loman` imports either one —
both are loaded lazily, at the point you use the feature.

| Extra | What it adds |
|---|---|
| `ui` | `comp.widget()`, the live notebook graph — see [The Interactive Widget](features/querying/interactive_widget.md) |
| `efficient` | Parquet storage for DataFrames when saving a computation |

```bash
$ uv add 'loman[ui,efficient]'      # or: pip install 'loman[ui,efficient]'
```

`efficient` is a storage-format choice rather than a capability: without it, saved frames
are written as `.npy`, which is numpy's own format and needs no extra package. Install it
when you want the saved data readable by other tools; a frame pyarrow cannot represent
falls back to the default encoding rather than failing the save.

## Installing Graphviz

Loman draws dependency graphs by shelling out to [Graphviz](https://graphviz.org/)'s
`dot` command. The Python glue (`pydotplus`) is installed automatically with Loman, but
the Graphviz binaries themselves are a separate, non-Python program that you need to
install yourself — `pip install loman` cannot do it for you.

Everything else in Loman works without Graphviz; you only need it for the visualization
features, for example `Computation.draw()` and the graph widget.

### Install the binaries

Graphviz publishes packages for every major platform. Its
[official download page](https://graphviz.org/download/) has full instructions, but for
most people one of these one-liners is enough:

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install graphviz` |
| Debian / Ubuntu | `sudo apt-get install graphviz` |
| Fedora / RHEL | `sudo dnf install graphviz` |
| conda (any platform) | `conda install -c conda-forge graphviz` |
| Windows (winget) | `winget install Graphviz.Graphviz` |
| Windows (Chocolatey) | `choco install graphviz` |

After installing, confirm the `dot` binary is on your `PATH`:

```bash
$ dot -V
dot - graphviz version 12.0.0 (...)
```

If that prints a version, Loman's visualization features are ready to use. If it reports
that `dot` cannot be found, see the next section.

### Windows: adding the Graphviz binary to your PATH

On Windows, some Graphviz installers place `dot.exe` in a directory that is not on your
`PATH`, so Loman cannot find it even though Graphviz is installed. To fix this, locate
`dot.exe` and add its directory to your `PATH`.

To find where `dot.exe` was installed, use the `where` command (it may not find it if the
directory isn't yet on your `PATH`, in which case look under your Graphviz installation
directory, typically `C:\Program Files\Graphviz\bin`):

```
C:\>where dot
C:\Program Files\Graphviz\bin\dot.exe
```

You can then add that `bin` directory to your `PATH`. The permanent fix is to add it
through the Windows *Environment Variables* control panel. To set it just for the current
Python session, you can run:

```python
import os


def ensure_path(path):
    paths = os.environ["PATH"].split(";")
    if path not in paths:
        paths.append(path)
        os.environ["PATH"] = ";".join(paths)


ensure_path(r"C:\Program Files\Graphviz\bin")
```

## Installing for development

Loman's development environment is managed by
[Rhiza](https://github.com/Jebel-Quant/rhiza), a template that supplies the `Makefile`,
the CI workflows, the pre-commit hooks and the gates behind them. You do not need to
install it: it is synced into the repository, and the `Makefile` is a thin shim that
forwards to the `rhiza-task` CLI, fetched on demand through `uvx`.

So the whole setup is:

```bash
$ git clone https://github.com/janushendersonassetallocation/loman.git
$ cd loman
$ make install
```

`make install` creates the virtual environment, syncs every dependency group and extra
from `uv.lock`, installs the pre-commit hooks, and runs the repository's own
`local-setup.sh` — which installs Graphviz if it is missing, so the visualization tests
can run. uv is installed automatically if you do not already have it.

Then:

```bash
$ make test     # the full suite, with coverage
$ make fmt      # the pre-commit hooks over every file
$ make all      # every gate CI runs
$ make help     # every available target
```

`make help` is worth running once: it lists the tasks Rhiza provides and, separately, the
targets this repository adds in `local.mk`.

For what Rhiza is and where its own documentation lives, see
[Rhiza tooling](../development/rhiza.md).
