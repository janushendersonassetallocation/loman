# Installation Guide

## Using Pip

To install Loman, run the following command:

```bash
$ pip install loman
```

If you don't have [pip](https://pip.pypa.io) installed (tisk tisk!),
[this Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/)
can guide you through the process.

## Installing Graphviz

Loman draws dependency graphs by shelling out to [Graphviz](https://graphviz.org/)'s
`dot` command. The Python glue (`pydotplus`) is installed automatically with Loman,
but the Graphviz binaries themselves are a separate, non-Python program that you
need to install yourself — `pip install loman` cannot do it for you.

Everything else in Loman works without Graphviz; you only need it for the
visualization features (for example `Computation.draw()` and the graph widgets).

### Install the binaries

Graphviz publishes packages for every major platform. Its
[official download page](https://graphviz.org/download/) has full instructions,
but for most people one of these one-liners is enough:

| Platform            | Command                                       |
| ------------------- | --------------------------------------------- |
| macOS (Homebrew)    | `brew install graphviz`                       |
| Debian / Ubuntu     | `sudo apt-get install graphviz`               |
| Fedora / RHEL       | `sudo dnf install graphviz`                   |
| Windows (Chocolatey)| `choco install graphviz`                      |
| Windows (winget)    | `winget install Graphviz.Graphviz`            |

After installing, confirm the `dot` binary is on your `PATH`:

```bash
$ dot -V
dot - graphviz version 12.0.0 (...)
```

If that prints a version, Loman's visualization features are ready to use. If it
reports that `dot` cannot be found, see the section below.

### Windows users: Adding the graphviz binary to your PATH

On Windows, some Graphviz installers place `dot.exe` in a directory that is not on
your `PATH`, so Loman cannot find it even though Graphviz is installed. To fix this,
locate `dot.exe` and add its directory to your `PATH`.

To find where `dot.exe` was installed, use the `where` command (it may not find it
if the directory isn't yet on your `PATH`, in which case look under your Graphviz
installation directory, typically `C:\Program Files\Graphviz\bin`):

```
C:\>where dot
C:\Program Files\Graphviz\bin\dot.exe
```

You can then add that `bin` directory to your `PATH`. The permanent fix is to add
it through the Windows *Environment Variables* control panel. To set it just for
the current Python session, you can run:

```python
import sys, os
def ensure_path(path):
    paths = os.environ['PATH'].split(';')
    if path not in paths:
        paths.append(path)
        os.environ['PATH'] = ';'.join(paths)
ensure_path(r'C:\Program Files\Graphviz\bin')
```
