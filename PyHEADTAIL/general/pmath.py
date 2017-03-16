'''
PyHEADTAIL math functions
Dispatches for CPU/GPU versions
@author Stefan Hegglin
@date 05.10.2015
'''
import numpy as np
from ..cobra_functions import stats as cp
try:
    import pycuda.cumath
    import pycuda.gpuarray
    import pycuda.tools
    from ..gpu import gpu_utils
    from ..gpu import gpu_wrap
    has_pycuda = gpu_wrap.has_pycuda
except (ImportError, OSError):
    # print ('No Pycuda in pmath.py import statement found')
    has_pycuda = False
try:
    import skcuda.misc
except ImportError:
    # print ('Skcuda not found. (Scikit-cuda)')
    pass

from functools import wraps

# FADDEEVA error function (wofz) business (used a.o. in spacecharge module)
try:
    from errfff import errf as _errf_f
    _errf = np.vectorize(_errf_f)
except ImportError:
    _errf = None
from scipy.special import erfc as _erfc
from scipy.special import wofz as _scipy_wofz
def _wofz(x, y):
    res = _scipy_wofz(x + 1j*y)
    return res.real, res.imag
def _errfadd(z):
    return np.exp(-z**2) * _erfc(z * -1j)


# # Kevin's sincos interface:
# try:
#     from ..cobra_functions.c_sin_cos import cm_sin, cm_cos

#     def cm_sincos(x):
#         return cm_sin(x), cm_cos(x)

#     sin = cm_sin
#     cos = cm_cos
#     sincos = cm_sincos
# except ImportError as e:
#     # print ('\n' + e.message)
#     # print ("Falling back to NumPy versions...\n")
### defaulting to NumPy sin/cos because Kevin's sincos interface
### results in VisibleDeprecationWarnings with the current transverse
### tracking module (returned objects are memoryviews and not ndarrays)
def np_sincos(x):
    return np.sin(x), np.cos(x)

sin = np.sin
cos = np.cos
sincos = np_sincos


def _mean_per_slice_cpu(sliceset, u, **kwargs):
    '''
    CPU Wrapper for the mean per slice function.
    TODO: Find a good spot where to put this function (equiv to gpu_wrap)
    --> Directly into cobra_functions/stats.pyx?
    '''
    mean_u = np.zeros(sliceset.n_slices)
    cp.mean_per_slice(sliceset.slice_index_of_particle,
                      sliceset.particles_within_cuts,
                      sliceset.n_macroparticles_per_slice,
                      u, mean_u)
    return mean_u

def _std_per_slice_cpu(sliceset, u, **kwargs):
    '''
    CPU Wrapper for the cov per slice function.
    TODO: Find a good spot where to put this function (equiv to gpu_wrap)
    --> Directly into cobra_functions/stats.pyx?
    '''
    std_u = np.zeros(sliceset.n_slices)
    cp.std_per_slice(sliceset.slice_index_of_particle,
                      sliceset.particles_within_cuts,
                      sliceset.n_macroparticles_per_slice,
                      u, std_u)
    return std_u

def _emittance_per_slice_cpu(sliceset, u, up, dp=None, **kwargs):
    '''
    CPU Wrapper for the emittance per slice function.
    TODO: Find a good spot where to put this function (equiv to gpu_wrap)
    --> Directly into cobra_functions/stats.pyx?
    '''
    emittance_u = np.zeros(sliceset.n_slices)
    cp.emittance_per_slice(sliceset.slice_index_of_particle,
                           sliceset.particles_within_cuts,
                           sliceset.n_macroparticles_per_slice,
                           u, up, dp, emittance_u)
    return emittance_u

def _count_macroparticles_per_slice_cpu(sliceset):
    output = np.zeros(sliceset.n_slices, dtype=np.int32)
    cp.count_macroparticles_per_slice(sliceset.slice_index_of_particle,
        sliceset.particles_within_cuts,
        output)
    return output

def _init_bunch_buffer(bunch_stats, buffer_size):
    buf = {}
    for stats in bunch_stats:
        buf[stats] = np.zeros(buffer_size)
    return buf

def _init_slice_buffer(slice_stats, n_slices, buffer_size):
    buf = {}
    for stats in slice_stats:
        buf[stats] = np.zeros((n_slices, buffer_size))
    return buf

def _searchsortedleft(array, values, dest_array=None):
    if dest_array is not None:
        dest_array[:] = np.searchsorted(array, values, side='left')
    else:
        dest_array = np.searchsorted(array, values, side='left')
    return dest_array

def _searchsortedright(array, values, dest_array=None):
    if dest_array is not None:
        dest_array[:] = np.searchsorted(array, values, side='right')
    else:
        dest_array = np.searchsorted(array, values, side='right')
    return dest_array


#### dictionaries storing the CPU and GPU versions of the desired functions ####
_CPU_numpy_func_dict = {
    'sin': np.sin,
    'cos': np.cos,
    'exp': np.exp,
    'log': np.log,
    'arcsin': np.arcsin,
    'mean': np.mean,
    'std': cp.std,
    'emittance': lambda *args, **kwargs: cp.emittance(*args, **kwargs),
    'min': np.min,
    'max': np.max,
    'diff': np.diff,
    'floor': np.floor,
    'argsort': np.argsort,
    'apply_permutation': lambda array, permutation: array[permutation], #auto copy
    'mean_per_slice': _mean_per_slice_cpu,
    #'cov_per_slice': lambda sliceset, u: _cov_per_slice_cpu(sliceset, u),
    'std_per_slice': _std_per_slice_cpu,
    'emittance_per_slice': _emittance_per_slice_cpu,
    'particles_within_cuts': lambda sliceset: np.where(
            (sliceset.slice_index_of_particle < sliceset.n_slices)
            & (sliceset.slice_index_of_particle >= 0)
        )[0].astype(np.int32),
    'particles_outside_cuts': lambda sliceset: np.where(np.logical_not(
            (sliceset.slice_index_of_particle < sliceset.n_slices)
            & (sliceset.slice_index_of_particle >= 0))
        )[0].astype(np.int32),
    'macroparticles_per_slice': lambda sliceset: _count_macroparticles_per_slice_cpu(sliceset),
    'take': np.take,
    'convolve': np.convolve,
    'seq': lambda stop: np.arange(stop, dtype=np.int32),
    'arange': wraps(np.arange)(
        lambda start, stop, step, nslices=None, dtype=np.float64:
            np.arange(start, stop, step, dtype)
    ),
    'zeros': np.zeros,
    'empty': np.empty,
    'empty_like': np.empty_like,
    'ones': np.ones,
    'device': 'CPU',
    'init_bunch_buffer': lambda bunch, bunch_stats, buffer_size: _init_bunch_buffer(bunch_stats, buffer_size),
    'init_slice_buffer': lambda slice_set, slice_stats, buffer_size: _init_slice_buffer(slice_stats, slice_set.n_slices, buffer_size),
    'searchsortedleft': _searchsortedleft,
    'searchsortedright': _searchsortedright,
    'sum': np.sum,
    'cumsum': np.cumsum,
    'wofz': _wofz,
    'all': np.all,
    'any': np.any,
    'indexify': lambda array: array.astype(np.int32),
    'abs': np.abs,
    'sign': np.sign,
    'sqrt': np.sqrt,
    'allclose': np.allclose,
    'put': np.put,
    'atleast_1d': np.atleast_1d,
    'almost_zero': lambda array, *args, **kwargs: np.allclose(array, 0, *args, **kwargs),
    'sincos': sincos,
    '_cpu': None # dummy to have at least one distinction between cpu/gpu
}

if has_pycuda:
    _GPU_func_dict = {
        'sin': pycuda.cumath.sin,
        'cos': pycuda.cumath.cos,
        'exp': pycuda.cumath.exp,
        'log': pycuda.cumath.log,
        'cosh': pycuda.cumath.cosh,
        'mean': gpu_wrap.mean,#lambda *args, **kwargs: skcuda.misc.mean(*args, **kwargs),
        'std': gpu_wrap.std,
        'emittance': lambda u, up, dp=None, **kwargs: gpu_wrap.emittance(u, up, dp, **kwargs),
        'min': lambda *args, **kwargs: pycuda.gpuarray.min(*args, **kwargs).get(),
        'max': lambda *args, **kwargs: pycuda.gpuarray.max(*args, **kwargs).get(),
        'diff': lambda *args, **kwargs: skcuda.misc.diff(*args, **kwargs),
        'floor': lambda *args, **kwargs: pycuda.cumath.floor(*args, **kwargs),
        'argsort': gpu_wrap.argsort,
        'apply_permutation': gpu_wrap.apply_permutation,
        'mean_per_slice': gpu_wrap.sorted_mean_per_slice,
        'std_per_slice': gpu_wrap.sorted_std_per_slice,
        'emittance_per_slice': gpu_wrap.sorted_emittance_per_slice,
        'particles_within_cuts': gpu_wrap.particles_within_cuts,
        'particles_outside_cuts': gpu_wrap.particles_outside_cuts,
        'macroparticles_per_slice': gpu_wrap.macroparticles_per_slice,
        'take': pycuda.gpuarray.take,
        'convolve': gpu_wrap.convolve,
        'seq': lambda stop: pycuda.gpuarray.arange(stop, dtype=np.int32),
        'arange': gpu_wrap.arange,
        'zeros': lambda *args, **kwargs: pycuda.gpuarray.zeros(*args,
            allocator=gpu_utils.memory_pool.allocate, **kwargs),
        'empty': lambda *args, **kwargs: pycuda.gpuarray.empty(*args,
            allocator=gpu_utils.memory_pool.allocate, **kwargs),
        'empty_like': pycuda.gpuarray.empty_like,
        'ones': lambda *args, **kwargs: pycuda.gpuarray.zeros(*args,
            allocator=gpu_utils.memory_pool.allocate, **kwargs) + 1,
        'device': 'GPU',
        'init_bunch_buffer': gpu_wrap.init_bunch_buffer,
        'init_slice_buffer': gpu_wrap.init_slice_buffer,
        'searchsortedleft': gpu_wrap.searchsortedleft,
        'searchsortedright': gpu_wrap.searchsortedright,
        'sum': wraps(pycuda.gpuarray.sum)(
            lambda *args, **kwargs: pycuda.gpuarray.sum(*args, **kwargs).get()
        ),
        'cumsum': gpu_wrap.cumsum,
        'wofz': gpu_wrap.wofz,
        'all': lambda array: pycuda.gpuarray.sum(array == 0).get() == 0,
        'any': lambda array: pycuda.gpuarray.sum(array != 0).get() > 0,
        'indexify': lambda array: array.astype(np.int32).get(), # indices cannot be GPUArrays
        'abs': lambda array: array.__abs__(),
        'sign': gpu_wrap.sign,
        'sqrt': pycuda.cumath.sqrt,
        'allclose': gpu_wrap.allclose,
        'put': skcuda.misc.set_by_index,
        'atleast_1d': gpu_wrap.atleast_1d,
        'almost_zero': lambda array, *args, **kwargs: gpu_wrap.allclose(
            array, pycuda.gpuarray.zeros(
                array.shape, dtype=array.dtype,
                allocator=gpu_utils.memory_pool.allocate),
            *args, **kwargs),
        'sincos': gpu_wrap.sincos,
        '_gpu': None # dummy to have at least one distinction between cpu/gpu
    }
################################################################################
def update_active_dict(new_dict):
    '''
    Update the currently active dictionary. Removes the keys of the currently
    active dictionary from globals() and spills the keys
    from new_dict to globals()
    Args:
        new_dict A dictionary which contents will be spilled to globals()
    '''
    if not hasattr(update_active_dict, 'active_dict'):
        update_active_dict.active_dict = new_dict
    # delete all old implementations/references from globals()
    for key in globals().keys():
        if key in update_active_dict.active_dict.keys():
            del globals()[key]
    # add the new active dict to the globals()
    globals().update(new_dict)
    update_active_dict.active_dict = new_dict

################################################################################
update_active_dict(_CPU_numpy_func_dict)
################################################################################

# print ('Available functions on GPU:\n' + str(_CPU_numpy_func_dict.keys()))
# print ('Available functions on CPU:\n' + str(_GPU_func_dict.keys()))

