#!/usr/bin/python

# thanks to Nick Foti for his cython skeleton, cf.
# http://nfoti.github.io/a-creative-blog-name/posts/2013/02/07/cleaning-cython-build-files/

import numpy as np
from _version import __version__

import re, os, sys, subprocess
import numpy as np

#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize


import platform
if platform.system() is 'Darwin':
    print ("Info: since you are running Mac OS, you "
           "may have to install with the following line:\n\n"
           "$ CC=gcc-4.9 ./install\n"
           "(or any equivalent version of gcc)")
    raw_input('Hit any key to continue...')

VERSIONFILE="_version.py"
#verstrline = open(VERSIONFILE, "rt").read()
#VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
#mo = re.search(VSRE, verstrline, re.M)
#if mo:
#    verstr = mo.group(1)
#else:
#    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
execfile(VERSIONFILE)
verstr = __version__
if not verstr[0].isdigit():
    raise RuntimeError("Unable to determine version from _version.py, "
                       "perhaps no git-describe available?")

args = sys.argv[1:]
# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython and fortran compilation files..."
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf ./build", shell=True, executable="/bin/bash")
    subprocess.Popen("find ./ -name *.c | xargs rm -f", shell=True)
    subprocess.Popen("find ./ -name *.so | xargs rm -f", shell=True)

    # Now do a normal clean
    sys.argv[1] = "clean"
    exit(1)


# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")


# Only build for 64-bit target
# os.environ['ARCHFLAGS'] = "-arch x86_64"


# Set up extension and build
cy_ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
cy_ext = [
        Extension("solvers.grid_functions",
                 ["solvers/grid_functions.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 ),
        Extension("cobra_functions.stats",
                 ["cobra_functions/stats.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 ),
        Extension("solvers.compute_potential_fgreenm2m",
                 ["solvers/compute_potential_fgreenm2m.pyx"],
                  include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 ),
        Extension("trackers.transverse_tracking_cython",
                 ["trackers/transverse_tracking_cython.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 ),
        Extension("trackers.detuners_cython",
                 ["trackers/detuners_cython.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 ),
        Extension("rfq.rfq_cython",
                 ["rfq/rfq_cython.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 ),
        Extension("aperture.aperture_cython",
                 ["aperture/aperture_cython.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 )
          ]

setup(
    name='PyHEADTAIL',
    version=verstr,
    description='CERN PyHEADTAIL numerical n-body simulation code for simulating macro-particle beam dynamics with collective effects.',
    url='http://github.com/PyCOMPLETE/PyHEADTAIL',
    packages=['PyHEADTAIL'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(cy_ext, **cy_ext_options),
    install_requires=[
        'numpy',
        'scipy',
        'hdf5',
        'h5py',
        'cython'
    ]
    )

from numpy.distutils.core import setup, Extension
setup(
    ext_modules = [Extension('general.errfff', ['general/errfff.f'])],
)
