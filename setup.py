#!/usr/bin/python

import numpy as np

import os
import sys
import subprocess
import cython_gsl

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


args = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython files..."
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")

    # Now do a normal clean
    sys.argv[1] = "clean"

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")

# Only build for 64-bit target
# os.environ['ARCHFLAGS'] = "-arch x86_64"

# Set up extension and build
cy_ext = [
        # Extension("beams.bunch",
        #           ["beams/bunch.pyx"],
        #          include_dirs=[np.get_include()],
        #          #extra_compile_args=["-g"],
        #          #extra_link_args=["-g"],
        #          libraries=["m"],
        #          library_dirs=[],
        #          ),
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
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 ),
        Extension("cobra_functions.random",
                 ["cobra_functions/random.pyx"],
                 include_dirs=[np.get_include(), cython_gsl.get_cython_include_dir()],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 library_dirs=[], libraries=["gsl", "gslcblas"],
                 ),
        Extension("solvers.compute_potential_fgreenm2m",
                 ["solvers/compute_potential_fgreenm2m.pyx"],
                  include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 ),
#        Extension("cobra_functions.interp1d",
#                 ["cobra_functions/interp1d.pyx"],
#                  include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
#                 #extra_compile_args=["-g"],
#                 #extra_link_args=["-g"],
#                 ),
        Extension("trackers.cython_tracker",
                 ["trackers/cython_tracker.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 ),
        Extension("rfq.cython_rfq",
                 ["rfq/cython_rfq.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 )
          ]

    # include_dirs = [cython_gsl.get_include()],
    # cmdclass = {'build_ext': build_ext},
    # ext_modules = [Extension("my_cython_script",
    #                          ["src/my_cython_script.pyx"],
    #                          libraries=cython_gsl.get_libraries(),
    #                          library_dirs=[cython_gsl.get_library_dir()],
    #                          include_dirs=[cython_gsl.get_cython_include_dir()])]
    
cy_ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(cmdclass={'build_ext': build_ext},
      ext_modules=cythonize(cy_ext, **cy_ext_options),)

# setup(
#     name="libBunch",
#     ext_modules=cythonize(extensions),
# )
