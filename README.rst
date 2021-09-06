   
PyHEADTAIL
==========

CERN PyHEADTAIL macro-particle code
for simulating beam dynamics in particle accelerators with collective effects.

PyHEADTAIL is written in Python with some C extensions, compiled with Cython.
Currently, PyHEADTAIL is compatible with Python v3.6 or later.

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

We recommend to use the Anaconda package manager to simplify installing.
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

For a single installation of PyHEADTAIL we recommended to add
the PyHEADTAIL path to your PYTHONPATH.
