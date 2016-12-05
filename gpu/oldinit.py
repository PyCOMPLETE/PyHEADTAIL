from ..particles import particles as def_particles

from .particles import ParticlesGPU

## all of the below is from develop, check what is needed!

def_particles.Particles = ParticlesGPU
from .. import Element
from .. import __version__

# monkey patching default classes with GPU variants
# is done on importing this module!

from skcuda.misc import mean, std, diff
from pycuda import gpuarray
from pycuda import cumath

# I. subjects to monkey patching:
from ..particles import particles as def_particles
from ..particles import slicing as def_slicing
from ..trackers import simple_long_tracking as def_simple_long_tracking
from ..trackers import wrapper as def_wrapper

# II. actual monkey patching

# a) Particles rebindings for GPU
from .particles import ParticlesGPU

def_particles.Particles = ParticlesGPU
def_particles.mean = lambda *args, **kwargs: mean(*args, **kwargs).get()
def_particles.std = lambda *args, **kwargs: std(*args, **kwargs).get()

# b) Slicing rebindings for GPU
# def_slicing.min_ = lambda *args, **kwargs: gpuarray.min(*args, **kwargs).get()
# def_slicing.max_ = lambda *args, **kwargs: gpuarray.max(*args, **kwargs).get()
# def_slicing.diff = diff

from .slicing import SlicerGPU

# # to be replaced: find a better solution than monkey patching base classes!
# # (to solve the corresponding need to replace all inheriting classes' parent!)
# def_slicing.Slicer = SlicerGPU
# def_slicing.UniformBinSlicer.__bases__ = (SlicerGPU,)

# c) Longitudinal tracker rebindings for GPU
def_simple_long_tracking.sin = cumath.sin

# d) Wrapper rebindings for GPU
from .wrapper import LongWrapperGPU
def_wrapper.LongWrapper = LongWrapperGPU
