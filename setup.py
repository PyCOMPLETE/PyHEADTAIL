#!/usr/bin/env python

# thanks to Nick Foti for his cython skeleton, cf.
# http://nfoti.github.io/a-creative-blog-name/posts/2013/02/07/cleaning-cython-build-files/

import os, sys, subprocess
from setuptools import setup, Extension, find_packages
from pathlib import Path

import numpy as np

from Cython.Build import cythonize
from Cython.Distutils import build_ext


version_file = Path(__file__).parent.absolute().joinpath('PyHEADTAIL/_version.py')
dd = {}
with open(version_file, 'r') as fp:
    exec(fp.read(), dd)
__version__ = dd['__version__']

args = sys.argv[1:]
# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print("Deleting cython and fortran compilation files...")
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf ./build", shell=True, executable="/bin/bash")
    subprocess.Popen("find ./ -name *.c | xargs rm -f", shell=True)
    subprocess.Popen("find ./ -name *.so | xargs rm -f", shell=True)
    subprocess.Popen("find ./ -name *.html | xargs rm -f", shell=True)

    # Now do a normal clean
    sys.argv = [sys.argv[0], 'clean']

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")

with open('README.rst', 'rb') as f:
    long_description = f.read().decode('utf-8')

# Only build for 64-bit target
# os.environ['ARCHFLAGS'] = "-arch x86_64"


# Set up extension and build
cy_ext_options = {
    "compiler_directives": {"profile": False,  # SLOW!!!
                            "embedsignature": True,
                            "linetrace": False,
                            "language_level": sys.version_info[0],
                            },
    "annotate": True,
}
cy_ext = [
    Extension("PyHEADTAIL.cobra_functions.stats",
              ["PyHEADTAIL/cobra_functions/stats.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[], libraries=["m"],
              extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"]),
    Extension("PyHEADTAIL.aperture.aperture_cython",
              ["PyHEADTAIL/aperture/aperture_cython.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[], libraries=["m"]),
    Extension("PyHEADTAIL.cobra_functions.c_sin_cos",
              ["PyHEADTAIL/cobra_functions/c_sin_cos.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[], libraries=["m"],
              extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"]),
    Extension("PyHEADTAIL.cobra_functions.interp_sin_cos",
              ["PyHEADTAIL/cobra_functions/interp_sin_cos.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[], libraries=["m"],
              extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"])
]

setup(
    name='PyHEADTAIL',
    version=__version__,
    description='CERN PyHEADTAIL numerical n-body simulation code '
                'for simulating macro-particle beam dynamics with collective effects.',
    url='https://github.com/PyCOMPLETE/PyHEADTAIL',
    author='Kevin Li',
    author_email='Kevin.Shing.Bruce.Li@cern.ch',
    maintainer='Adrian Oeftiger',
    maintainer_email='Adrian.Oeftiger@cern.ch',
    packages=find_packages(),
    long_description=long_description,
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(cy_ext, **cy_ext_options),
    include_package_data=True,  # install files matched by MANIFEST.in
    setup_requires=[
        'h5py',
        'numpy',
        'scipy',
        'cython',
    ]
)

# from numpy.distutils.core import setup, Extension
# setup(
#     ext_modules = [Extension('PyHEADTAIL.general.errfff',
#                              ['PyHEADTAIL/general/errfff.f90'])],
# )
