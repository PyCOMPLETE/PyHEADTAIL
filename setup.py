#!/usr/bin/python

import numpy as np
from _version import __version__

import re, os, sys, subprocess
#import cython_gsl
import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


VERSIONFILE="_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


args = sys.argv[1:]
# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython files..."
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf ./build", shell=True, executable="/bin/bash")
    subprocess.Popen("find ./ -name *.c | xargs rm", shell=True)
    subprocess.Popen("find ./ -name *.so | xargs rm", shell=True)

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
        #Extension("cobra_functions.random",
                 #["cobra_functions/random.pyx"],
                 #include_dirs=[np.get_include(), cython_gsl.get_cython_include_dir()],
                 ##extra_compile_args=["-g"],
                 ##extra_link_args=["-g"],
                 #library_dirs=[], libraries=["gsl", "gslcblas"],
                 #),
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
        Extension("trackers.transverse_tracking_cython",
                 ["trackers/transverse_tracking_cython.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 ),
        Extension("trackers.detuners_cython",
                 ["trackers/detuners_cython.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 ),
        Extension("rfq.rfq",
                 ["rfq/rfq.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"],
                 extra_compile_args=["-fopenmp"],
                 extra_link_args=["-fopenmp"],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 )
          ]

setup(
    name='PyHEADTAIL',
    version=verstr,
    description='CERN macroparticle tracking code for collective effects in circular accelerators.',
    url='http://github.com/like2000/PyHEADTAIL',
    packages=['PyHEADTAIL'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(cy_ext, **cy_ext_options),
    )

# setup(
#     name="libBunch",
#     ext_modules=cythonize(extensions),
# )
