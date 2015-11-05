'''
PyHEADTAIL math functions
Dispatches for CPU/GPU versions
@author Stefan Hegglin
@date 05.10.2015
'''
import numpy as np
from ..cobra_functions import stats as cp
from ..gpu import gpu_wrap as gpu_wrap
try:
    import pycuda.cumath
    import pycuda.gpuarray
except ImportError:
    print ('No Pycuda in math.py import statement found')
try:
    import skcuda.misc
except ImportError:
    print ('Skcuda not found. (Scikit-cuda)')


def _mean_per_slice_cpu(sliceset, u):
    '''
    CPU Wrapper for the mean per slice function.
    TODO: Find a good spot where to put this function (equiv to gpu_wrap)
    '''
    mean_u = np.zeros(sliceset.n_slices)
    cp.mean_per_slice(sliceset.slice_index_of_particle,
                      sliceset.particles_within_cuts,
                      sliceset.n_macroparticles_per_slice,
                      u, mean_u)
    return mean_u

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
    'mean_per_slice' : lambda sliceset,  u: _mean_per_slice_cpu(sliceset, u),
    '_cpu' : None # dummy to have at least one distinction between cpu/gpu
}

_GPU_func_dict = {
    'sin' : pycuda.cumath.sin,
    'cos' : pycuda.cumath.cos,
    'exp' : pycuda.cumath.exp,
    'cosh': pycuda.cumath.cosh,
    'mean': lambda *args, **kwargs : skcuda.misc.mean(*args, **kwargs).get(),
    'std': lambda *args, **kwargs : skcuda.misc.std(*args, **kwargs).get(),
    'emittance' : lambda u, up, dp=None : gpu_wrap.emittance(u, up, dp),
    'min': lambda *args, **kwargs : pycuda.gpuarray.min(*args, **kwargs).get(),
    'max': lambda *args, **kwargs : pycuda.gpuarray.max(*args, **kwargs).get(),
    'diff' : lambda *args, **kwargs : skcuda.misc.diff(*args, **kwargs),
    'floor': lambda *args, **kwargs : pycuda.cumath.floor(*args, **kwargs),
    'argsort': gpu_wrap.argsort,
    'apply_permutation' : gpu_wrap.apply_permutation,
    'mean_per_slice' : gpu_wrap.sorted_mean_per_slice,
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

print ('Available functions on GPU:\n' + str(_CPU_numpy_func_dict.keys()))
print ('Available functions on CPU:\n' + str(_GPU_func_dict.keys()))
