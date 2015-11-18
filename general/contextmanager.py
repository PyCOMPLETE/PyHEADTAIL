'''
Context manager classes
@author Stefan Hegglin
@data 30.09.2015
'''
import numpy as np
import pmath as pm
try:
    import pycuda.gpuarray as gpuarray
    import pycuda
    import pycuda.tools
    has_pycuda = True
except ImportError:
    print('pycuda not found, GPU context unavailable')
    has_pycuda = False

if has_pycuda:
    GPU_utils = dict()
    GPU_utils['memory_pool'] = pycuda.tools.DeviceMemoryPool()
    #GPU_utils['memory_pool'] = pycuda.tools.PageLockedMemoryPool()



class Context(object):
    '''
    Example contextmanager class providing enter and exit methods
    '''
    def __init__(self):
        print('Context() created')

    def __enter__(self):
        print('Entered context')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Exited context')

class CPU(object):
    '''
    Dummy class to run the code on the CPU.
    Does nothing but has the same interface as the GPU contextmanager
    '''
    def __init__(self, bunch):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class GPU(object):
    '''
    Class providing enter/exit methods to move/get data from/to the gpu or
    provide a general base framework for all decorated function calls
    All data after must be in the same state after exiting as before entering
    this context!
    '''
    def __init__(self, bunch):
        '''
        Pass the bunch to the context s.t. the context knows what to copy
        to the gpu. The problem with this approach: not very nice for the user:
        with GPU(bunch) as context:
        '''
        # replace above line with bunch.coords_n_momenta!
        # 'id' is required for the sorting and has to be transformed as well
        self.bunch = bunch #reference!
        self.to_move = self.bunch.coords_n_momenta | set(['id'])
        self.previous_state = dict()


    def __enter__(self):
        '''
        Move all data to the GPU (and monkey patch methods?)
        Returns self (eg. to provide info about gpu/status/...)
        '''
        for coord in self.to_move:
            obj = getattr(self.bunch, coord, None)
            if isinstance(obj, np.ndarray):
                setattr(self.bunch, coord, gpuarray.to_gpu(obj, GPU_utils['memory_pool'].allocate))

        # replace functions in general.math.py
        pm.update_active_dict(pm._GPU_func_dict)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        Move all data back to the CPU (and un-patch the methods?)
        Reestablish state of everything as it was before entering
        '''
        for coord in self.to_move:
            obj = getattr(self.bunch, coord, None)
            if isinstance(obj, pycuda.gpuarray.GPUArray):
                setattr(self.bunch, coord, obj.get())
        pm.update_active_dict(pm._CPU_numpy_func_dict)
