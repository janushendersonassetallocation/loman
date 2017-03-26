Installation Guide
==================

Using Pip
---------

To install Loman, run the following command:

.. code-block:: bash

    $ pip install loman

If you don't have `pip <https://pip.pypa.io>`_ installed (tisk tisk!),
`this Python installation guide <http://docs.python-guide.org/en/latest/starting/installation/>`_
can guide you through the process.

Dependency on graphviz
----------------------

Loman uses the `graphviz <http://www.graphviz.org/>` tool, and the Python `graphviz library <https://pypi.python.org/pypi/graphviz>` to draw dependency graphs. If you are using Continuum's excellent `Anaconda Python <https://www.continuum.io/downloads>` distribution (recommended), then you can install them by running these commands:

.. code-block:: bash

    $ conda install graphviz
    $ python install graphviz

Windows users: Adding the graphviz binary to your PATH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under Windows, Anaconda's graphviz package installs the graphviz tool's binaries in a subdirectory under the bin directory, but only the bin directory is on the PATH. So we will need to add the subdirectory to the path. To find out where the bin directory is in your installation, use the where command:

::

    C:\>where dot
    C:\ProgramData\Anaconda3\Library\bin\dot.bat
    C:\>dir C:\ProgramData\Anaconda3\Library\bin\graphviz\dot.exe
     Volume in drive C has no label.
     Volume Serial Number is XXXX-XXXX

     Directory of C:\ProgramData\Anaconda3\Library\bin\graphviz

    01/03/2017  04:16 PM             7,680 dot.exe
               1 File(s)          7,680 bytes
               0 Dir(s)  xx bytes free

You can then add the subdirectory graphviz to your PATH. You can either do this through the Windows Control Panel, or in an interactive session, by running this code:

.. code-block:: python

    import sys, os
    def ensure_path(path):
        paths = os.environ['PATH'].split(';')
        if path not in paths:
            paths.append(path)
            os.environ['PATH'] = ';'.join(paths)
    ensure_path(r'C:\ProgramData\Anaconda3\Library\bin\graphviz')