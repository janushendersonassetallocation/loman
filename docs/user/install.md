# Installation Guide

## Using Pip

To install Loman, run the following command:

```bash
$ pip install loman
```

If you don't have [pip](https://pip.pypa.io) installed (tisk tisk!),
[this Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/)
can guide you through the process.

## Optional extras

Two features are packaged as extras, so a plain `pip install loman` stays small. Neither
is needed for the core computation engine, and nothing in a bare `import loman` imports
either one — both are loaded lazily, at the point you use the feature.

| Extra | Install | What it adds |
|---|---|---|
| `ui` | `pip install 'loman[ui]'` | `comp.widget()`, the live notebook graph — see [The Interactive Widget](features/querying/interactive_widget.md) |
| `efficient` | `pip install 'loman[efficient]'` | Parquet storage for DataFrames when saving a computation |

Both together:

```bash
$ pip install 'loman[ui,efficient]'
```

`efficient` is a storage-format choice rather than a capability: without it, saved frames
are written as `.npy`, which is numpy's own format and needs no extra package. Install it
when you want the saved data readable by other tools; a frame pyarrow cannot represent
falls back to the default encoding rather than failing the save.

## Dependency on graphviz

Loman uses the [graphviz](http://www.graphviz.org/) tool, and the Python [graphviz library](https://pypi.python.org/pypi/graphviz) to draw dependency graphs. If you are using Continuum's excellent [Anaconda Python](https://www.continuum.io/downloads) distribution (recommended), then you can install them by running these commands:

```bash
$ conda install graphviz
$ python install graphviz
```

### Windows users: Adding the graphviz binary to your PATH

Under Windows, Anaconda's graphviz package installs the graphviz tool's binaries in a subdirectory under the bin directory, but only the bin directory is on the PATH. So we will need to add the subdirectory to the path. To find out where the bin directory is in your installation, use the where command:

```
C:\>where dot
C:\ProgramData\Anaconda3\Library\bin\dot.bat
C:\>dir C:\ProgramData\Anaconda3\Library\bin\graphviz\dot.exe
 Volume in drive C has no label.
 Volume Serial Number is XXXX-XXXX

 Directory of C:\ProgramData\Anaconda3\Library\bin\graphviz

01/03/2017  04:16 PM             7,680 dot.exe
           1 File(s)          7,680 bytes
           0 Dir(s)  xx bytes free
```

You can then add the subdirectory graphviz to your PATH. You can either do this through the Windows Control Panel, or in an interactive session, by running this code:

```python
import sys, os


def ensure_path(path):
    paths = os.environ["PATH"].split(";")
    if path not in paths:
        paths.append(path)
        os.environ["PATH"] = ";".join(paths)


ensure_path(r"C:\ProgramData\Anaconda3\Library\bin\graphviz")
```
