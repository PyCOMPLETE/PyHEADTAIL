PyHEADTAIL
==========

CERN PyHEADTAIL numerical n-body simulation code
for simulating macro-particle beam dynamics with collective effects.

PyHEADTAIL is written in C and Python.
Currently, PyHEADTAIL is compatible with Python v2.7.

Installation for Users
----------------------

For using PyHEADTAIL without modifying the source code,
we recommend to install the latest version via PyPI:

.. code-block:: bash

    $ pip install PyHEADTAIL

Installation for Developers
---------------------------

For developers of PyHEADTAIL, we recommend to install a stand-alone
package from the source code using git. For GPU usage, the developer
version is required (the Makefile is included in the source code
version only).

We recommend to use the Anaconda package manager (for Python 2.7) to simplify installing.
You can obtain it from anaconda.org .

Installation of PyHEADTAIL on linux (having git installed)
is straight forward.

- Clone the repository in a local folder:

.. code-block:: bash

    $ git clone https://github.com/PyCOMPLETE/PyHEADTAIL

- Go to the folder and run the installation script:

.. code-block:: bash

    $ cd PyHEADTAIL

    $ make

And there you go, start using PyHEADTAIL!

.. code-block:: bash

    $ cd ..

    $ ipython

    ...

    In [1]: import PyHEADTAIL

    PyHEADTAIL v1.13.1

For a single installation of PyHEADTAIL we recommended to add
the PyHEADTAIL path to your PYTHONPATH.
