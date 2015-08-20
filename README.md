PyHEADTAIL
==========

CERN PyHEADTAIL numerical n-body simulation code 
for simulating macro-particle beam dynamics with collective effects.

Currently only a git-clone of the repository is supported.
The available zip files on github are missing the versioning 
and will not allow ./install to work.

# Installation
Installation of PyHEADTAIL on linux (having git installed) 
is straight forward.

- Clone the repository in a local folder:

$ git clone https://github.com/PyCOMPLETE/PyHEADTAIL

- Go to the folder and run the installation script:

$ cd PyHEADTAIL

$ ./install

And there you go, start using PyHEADTAIL!

$ cd ..

$ ipython

...

In [1]: import PyHEADTAIL

PyHEADTAIL v1.4.0-0-g8422af8081


-------------------------------------------------------------------------------

Please use the pre-commit script 'pre-commit.py' if you want to contribute
to the repository. It only lets you commit to the develop and master branch if
no unit tests fail.

To install (creates a symlink): ln -s ../../pre-commit.py .git/hooks/pre-commit
