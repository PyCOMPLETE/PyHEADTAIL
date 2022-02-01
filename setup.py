# thanks to Nick Foti for his cython skeleton, cf.
# http://nfoti.github.io/a-creative-blog-name/posts/2013/02/07/cleaning-cython-build-files/

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

TOPLEVEL_DIR = Path(__file__).parent.absolute()
VERSION_FILE = TOPLEVEL_DIR / "PyHEADTAIL" / "_version.py"
README_FILE = TOPLEVEL_DIR / "README.md"

PACKAGE_METADATA = {}
with VERSION_FILE.open("r") as fp:
    exec(fp.read(), PACKAGE_METADATA)  # be wary that this will try to run code in the file!

# Get the readme content for PyPI
with README_FILE.open("r") as docs:
    long_description = docs.read()

CALL_ARGS = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
# This kicks in when calling 'python setup.py' directly which is deprecated
if "cleanall" in CALL_ARGS:
    print("[Cleanup] - Deleting Cython and Fortran compilation files.")
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf ./build", shell=True, executable="/bin/bash")
    subprocess.Popen("find ./ -name *.c | xargs rm -f", shell=True)
    subprocess.Popen("find ./ -name *.so | xargs rm -f", shell=True)
    subprocess.Popen("find ./ -name *.html | xargs rm -f", shell=True)

    # Now do a normal clean
    sys.argv = [sys.argv[0], "clean"]

# We want to always use build_ext --inplace
if CALL_ARGS.count("build_ext") > 0 and CALL_ARGS.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")

# Allow the user to install without openmp support (do we use it?), which helps a lot on macOS
# Defaults to include openmp support (not skip), unless 'PYHT_SKIP_OPENMP=1' is explicitely set in the environment
if os.getenv("PYHT_USE_OPENMP", "1") == "1":
    print("[OpenMP] - Enabling OpenMP support during compilation.")
    CYTHON_OPENMP_COMPILE_ARG = CYTHON_OPENMP_LINK_ARGS = "-fopenmp"
else:
    print("[OpenMP] - Skipping OpenMP support during compilation.")
    CYTHON_OPENMP_COMPILE_ARG = CYTHON_OPENMP_LINK_ARGS = ""


# Set up Cython extension and build
CYTHON_OPTIONS = {
    "compiler_directives": {
        "profile": False,  # SLOW!!!
        "embedsignature": True,
        "linetrace": False,
        "language_level": sys.version_info[0],
    },
    "annotate": True,
}
CYTHON_EXTENSION = [
    Extension(
        "PyHEADTAIL.cobra_functions.stats",
        ["PyHEADTAIL/cobra_functions/stats.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        libraries=["m"],
        extra_compile_args=[CYTHON_OPENMP_COMPILE_ARG],
        extra_link_args=[CYTHON_OPENMP_LINK_ARGS],
    ),
]

# Final call to properly package
setup(
    name="PyHEADTAIL",
    version=PACKAGE_METADATA["__version__"],
    description="CERN PyHEADTAIL numerical n-body simulation code "
    "for simulating macro-particle beam dynamics with collective effects.",
    url="https://github.com/PyCOMPLETE/PyHEADTAIL",
    author="Kevin Li",
    author_email="Kevin.Shing.Bruce.Li@cern.ch",
    maintainer="Adrian Oeftiger",
    maintainer_email="Adrian.Oeftiger@cern.ch",
    packages=find_packages(),
    long_description=long_description,
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(CYTHON_EXTENSION, **CYTHON_OPTIONS),
    include_package_data=True,  # make install include files declared in 'MANIFEST.in'
    install_requires=["h5py", "numpy", "scipy", "cython"],
    setup_requires=["h5py", "numpy", "scipy", "cython"],
)

# from numpy.distutils.core import setup, Extension
# setup(
#     ext_modules = [Extension('PyHEADTAIL.general.errfff',
#                              ['PyHEADTAIL/general/errfff.f90'])],
# )
