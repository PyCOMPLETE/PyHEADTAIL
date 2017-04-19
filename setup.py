#!/usr/bin/env python

# Todo:
# - extra cython build command into makefile for local building

# sources: 
# 1. http://stackoverflow.com/questions/4505747/
#    how-should-i-structure-a-python-package-that-contains-cython-code
# 2. http://nfoti.github.io/a-creative-blog-name/posts/2013/02/07/
#    cleaning-cython-build-files/

import numpy as np
import sys

from PyHEADTAIL._version import __version__

from setuptools import setup, Extension, find_packages
# from distutils.core import setup
# from distutils.extension import Extension
from distutils.command.sdist import sdist as _sdist

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cy_extensions = [
    "PyHEADTAIL.cobra_functions.stats",
    "PyHEADTAIL.aperture.aperture_cython",
    "PyHEADTAIL.cobra_functions.c_sin_cos",
    "PyHEADTAIL.cobra_functions.interp_sin_cos",
]

# Set up extension and build configuration
cy_ext_options = {
    "compiler_directives": {
        # "profile": True, # SLOW!!!
        "embedsignature": True
    },
    "annotate": True,
}

use_cython = any('build' in arg for arg in sys.argv[1:])

cmdclass = {}

if use_cython:
    cy_ext_paths = [cy_ext.replace(".", "/") + ".pyx"
                    for cy_ext in cy_extensions]
    cmdclass['build_ext'] = build_ext
else:
    cy_ext_paths = [cy_ext.replace(".", "/") + ".c"
                    for cy_ext in cy_extensions]

cy_ext_modules = [
    Extension(
        cy_ext, [cy_ext_path], libraries=["m"],
        extra_compile_args=["-fopenmp"], 
        extra_link_args=["-fopenmp"],
    )
    for cy_ext, cy_ext_path in zip(cy_extensions, cy_ext_paths)
]

class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(cy_ext_modules, **cy_ext_options)
        _sdist.run(self)
cmdclass['sdist'] = sdist

with open('README.rst', 'rb') as f:
    long_description = f.read().decode('utf-8')

import platform
if platform.system() is 'Darwin':
    print ("Info: since you are running Mac OS, you "
           "may have to install with the following line:\n\n"
           "$ CC=gcc-4.9 ./install\n"
           "(or any equivalent version of gcc)")

setup(
    name='PyHEADTAIL',
    version=__version__,
    description='CERN PyHEADTAIL numerical n-body simulation code '
        'for simulating macro-particle beam dynamics with collective effects.',
    url='https://github.com/PyCOMPLETE/PyHEADTAIL',
    author='Kevin Li',
    author_email='Kevin.Li@cern.ch',
    maintainer='Adrian Oeftiger',
    maintainer_email='Adrian.Oeftiger@cern.ch',
    packages=find_packages(),
    ext_modules=cy_ext_modules,
    cmdclass=cmdclass,
    include_dirs = [np.get_include()],
    long_description=long_description,
    include_package_data=True, # install files matched by MANIFEST.in
    setup_requires=[
        'numpy',
        'scipy',
        'h5py',
        'setuptools',
#        'cython',
    ]
)