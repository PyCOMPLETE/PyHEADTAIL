import numpy as np

import os
import sys
import subprocess

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
        Extension("beams.bunch",
                  ["beams/bunch.pyx"],
                 include_dirs=[np.get_include()],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 libraries=["m"],
                 library_dirs=[],
                 ),
        Extension("cobra_functions.cobra_functions",
                  ["cobra_functions/cobra_functions.pyx"],
                 include_dirs=[np.get_include()],
                 #extra_compile_args=["-g"],
                 #extra_link_args=["-g"],
                 libraries=["m"],
                 library_dirs=[],
                 )
          ]

cy_ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(cmdclass={'build_ext': build_ext},
      ext_modules=cythonize(cy_ext, **cy_ext_options),)

# setup(
#     name="libBunch",
#     ext_modules=cythonize(extensions),
# )
