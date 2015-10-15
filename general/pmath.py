'''
PyHEADTAIL math functions
Dispatches for CPU/GPU versions
@author Stefan Hegglin
@date 05.10.2015
'''

import numpy as np

try:
    import pycuda.cumath
except ImportError:
    print ('No Pycuda in math.py import statement found')



#### dictionaries storing the CPU and GPU versions of the desired functions ####
_CPU_numpy_func_dict = {
    'sin' : np.sin,
    'cos' : np.cos,
    'exp' : np.exp,
    'arcsin': np.arcsin,
    '_cpu' : None # dummy to have at least one distinction between cpu/gpu
}

_GPU_func_dict = {
    'sin' : pycuda.cumath.sin,
    'cos' : pycuda.cumath.cos,
    'exp' : pycuda.cumath.exp,
    'cosh': pycuda.cumath.cosh,
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
