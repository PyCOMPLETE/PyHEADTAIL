PyHEADTAIL
==========

CERN PyHEADTAIL numerical n-body simulation code
for simulating macro-particle beam dynamics with collective effects.

Installation
------------

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

    PyHEADTAIL v1.11.2


-------------------------------------------------------------------------------

Please use the pre-push script ``prepush.py`` if you want to contribute
to the repository. It only lets you push to the develop and master branch if
no unit tests fail.

To install (creates a symlink): ``ln -s ../../prepush.py .git/hooks/pre-push``
