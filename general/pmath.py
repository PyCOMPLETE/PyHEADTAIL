'''
PyHEADTAIL math functions
Dispatches for CPU/GPU versions
@author Stefan Hegglin
@date 05.10.2015
'''
import numpy as np
from ..cobra_functions import stats as cp
from ..gpu import gpu_utils
from ..gpu import gpu_wrap as gpu_wrap
try:
    import pycuda.cumath
    import pycuda.gpuarray
    import pycuda.tools
    has_pycuda = True
except ImportError:
    # print ('No Pycuda in pmath.py import statement found')
    has_pycuda = False
try:
    import skcuda.misc
except ImportError:
    # print ('Skcuda not found. (Scikit-cuda)')
    pass


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


#### dictionaries storing the CPU and GPU versions of the desired functions ####
_CPU_numpy_func_dict = {
    'sin' : np.sin,
    'cos' : np.cos,
    'exp' : np.exp,
    'arcsin': np.arcsin,
    'mean' : np.mean,
    'std' : cp.std,
    'emittance' : lambda *args, **kwargs : cp.emittance(*args, **kwargs),
    'min' : np.min,
    'max' : np.max,
    'diff' : np.diff,
    'floor': np.floor,
    'argsort' : np.argsort,
    'apply_permutation' : lambda array, permutation: array[permutation], #auto copy
    'mean_per_slice' : _mean_per_slice_cpu,
    #'cov_per_slice' : lambda sliceset, u: _cov_per_slice_cpu(sliceset, u),
    'std_per_slice' : _std_per_slice_cpu,
    'emittance_per_slice': _emittance_per_slice_cpu,
    'particles_within_cuts' : lambda sliceset: np.where(
        (sliceset.slice_index_of_particle < sliceset.n_slices) & (sliceset.slice_index_of_particle >= 0))[0].astype(np.int32),
    'macroparticles_per_slice': lambda sliceset : _count_macroparticles_per_slice_cpu(sliceset),
    'take' : np.take,
    'convolve': np.convolve,
    'arange': lambda start, stop, step, nslices=None, dtype=np.float64 :np.arange(start, stop, step, dtype),
    'zeros': np.zeros,
    'ones': np.ones,
    'device': 'CPU',
    'init_bunch_buffer': lambda bunch, bunch_stats, buffer_size : _init_bunch_buffer(bunch_stats, buffer_size),
    'init_slice_buffer': lambda slice_set, slice_stats, buffer_size : _init_slice_buffer(slice_stats, slice_set.n_slices, buffer_size),
    'searchsortedleft': lambda a, v, sorter=None, dest_array=None: np.searchsorted(a, v, side='left', sorter=sorter),
    'searchsortedright': lambda a, v, sorter=None, dest_array=None: np.searchsorted(a, v, side='right', sorter=sorter),
    'cumsum': np.cumsum,
    '_cpu' : None # dummy to have at least one distinction between cpu/gpu
}

_GPU_func_dict = {
    'sin' : pycuda.cumath.sin,
    'cos' : pycuda.cumath.cos,
    'exp' : pycuda.cumath.exp,
    'cosh': pycuda.cumath.cosh,
    'mean': gpu_wrap.mean,#lambda *args, **kwargs : skcuda.misc.mean(*args, **kwargs),
    'std': gpu_wrap.std,
    'emittance' : lambda u, up, dp=None, **kwargs: gpu_wrap.emittance(u, up, dp, **kwargs),
    'min': lambda *args, **kwargs : pycuda.gpuarray.min(*args, **kwargs).get(),
    'max': lambda *args, **kwargs : pycuda.gpuarray.max(*args, **kwargs).get(),
    'diff' : lambda *args, **kwargs : skcuda.misc.diff(*args, **kwargs),
    'floor': lambda *args, **kwargs : pycuda.cumath.floor(*args, **kwargs),
    'argsort': gpu_wrap.argsort,
    'apply_permutation' : gpu_wrap.apply_permutation,
    'mean_per_slice' : gpu_wrap.sorted_mean_per_slice,
    'std_per_slice' : gpu_wrap.sorted_std_per_slice,
    'emittance_per_slice' : gpu_wrap.sorted_emittance_per_slice,
    'particles_within_cuts': gpu_wrap.particles_within_cuts,
    'macroparticles_per_slice' : gpu_wrap.macroparticles_per_slice,
    'take': pycuda.gpuarray.take,
    'convolve': gpu_wrap.convolve,
    'arange': lambda start, stop, step, n_slices, dtype=np.float64: gpu_wrap.arange_startstop_gpu(start, stop, step, n_slices, dtype) if isinstance(start, pycuda.gpuarray.GPUArray) else pycuda.gpuarray.arange(start, stop, step, dtype=np.float64),
    'zeros': lambda *args, **kwargs : pycuda.gpuarray.zeros(*args,
        allocator=gpu_utils.memory_pool.allocate, **kwargs),
    'ones': lambda *args, **kwargs : pycuda.gpuarray.zeros(*args,
        allocator=gpu_utils.memory_pool.allocate, **kwargs) + 1,
    'device': 'GPU',
    'init_bunch_buffer': gpu_wrap.init_bunch_buffer,
    'init_slice_buffer': gpu_wrap.init_slice_buffer,
    'searchsortedleft': gpu_wrap.searchsortedleft,
    'searchsortedright': gpu_wrap.searchsortedright,
    'cumsum': skcuda.misc.cumsum,
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

