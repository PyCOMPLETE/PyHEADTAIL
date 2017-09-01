'''
Context manager classes
@author Stefan Hegglin
@data 30.09.2015
'''
import numpy as np

class UnknownContextManagerError(Exception):
    '''Raise if context manager is not found, e.g. cannot determine
    whether on CPU or on GPU.
    '''
    def __init__(self, message='Failed to determine current context, e.g. '
                               'whether pmath.device is "CPU" or "GPU".'):
        self.message = message

import pmath as pm
from ..gpu import gpu_utils
try:
    import pycuda.gpuarray as gpuarray
    import pycuda
    import pycuda.tools
    import pycuda.elementwise
    has_pycuda = True
except ImportError:
    # print ('pycuda not found, GPU context unavailable')
    has_pycuda = False

if has_pycuda:
    def create_kernel(operator):
        '''Return a elementwisekernel with the operator being one of
        +, -, /, * as a string.
        '''
        _ker = pycuda.elementwise.ElementwiseKernel(
                'double* out, double* a, const double* b',
                'out[i] = a[i] %s b[0]' %operator
            )
        return _ker

    def create_ckernel(operator):
        '''Return a complex elementwisekernel with the operator being
        one of +, -, /, * as a string.
        '''
        _ker = pycuda.elementwise.ElementwiseKernel(
                'pycuda::complex<double>* out, pycuda::complex<double>* a, '
                'const pycuda::complex<double>* b',
                'out[i] = a[i] %s b[0]' % operator
            )
        return _ker

    def patch_op(op, func_name):
        ''' Monkey patch the function: Wrap it with an extra check for shape ()
        func_name: one of __isub__, __iadd__, __imul__, __idiv__
        '''
        _kernel = create_kernel(op)
        _ckernel = create_ckernel(op)
        old_v = getattr(pycuda.gpuarray.GPUArray, func_name)
        def _patch(self, other):
            if isinstance(other, pycuda.gpuarray.GPUArray) and other.shape in [(), (1,)]:
                if 'c' in (self.dtype.kind, other.dtype.kind):
                    self = self.astype(complex)
                    _ckernel(self, self, other.astype(complex))
                else:
                    if self.dtype != np.float64 or other.dtype != np.float64:
                        raise NotImplementedError(
                            'pmath: Oops. So far only double or complex '
                            'operations with GPU scalars have been '
                            'implemented. Please convert both operands at '
                            'least to np.float64. Or implement a more '
                            'general monkey patching of GPUArray operators.'
                        )
                    _kernel(self, self, other)
            else:
                old_v(self, other)
            return self
        return _patch

    def patch_binop(op, func_name):
        '''monkey patch __sub__, __add__, ...'''
        _kernel = create_kernel(op)
        _ckernel = create_ckernel(op)
        old_v = getattr(pycuda.gpuarray.GPUArray, func_name)
        def _patch_binop(self, other):
            if isinstance(other, pycuda.gpuarray.GPUArray) and other.shape in [(),(1,)]:
                if 'c' in (self.dtype.kind, other.dtype.kind):
                    self = self.astype(complex)
                    out = pycuda.gpuarray.empty_like(self)
                    _ckernel(out, self, other.astype(complex))
                else:
                    if self.dtype != np.float64 or other.dtype != np.float64:
                        raise NotImplementedError(
                            'pmath: Oops. So far only double or complex '
                            'operations with GPU scalars have been '
                            'implemented. Please convert both operands at '
                            'least to np.float64. Or implement a more '
                            'general monkey patching of GPUArray operators.'
                        )
                    out = pycuda.gpuarray.empty_like(self)
                    _kernel(out, self, other)
                return out
            else:
                return old_v(self, other)
        return _patch_binop

    # patch the GPUArray to be able to cope with gpuarrays of size 1 as ops
    pycuda.gpuarray.GPUArray.__isub__ = patch_op('-', '__isub__')
    pycuda.gpuarray.GPUArray.__iadd__ = patch_op('+', '__iadd__')
    pycuda.gpuarray.GPUArray.__imul__ = patch_op('*', '__imul__')
    pycuda.gpuarray.GPUArray.__idiv__ = patch_op('/', '__idiv__')
    pycuda.gpuarray.GPUArray.__sub__ = patch_binop('-', '__sub__')
    pycuda.gpuarray.GPUArray.__add__ = patch_binop('+', '__add__')
    pycuda.gpuarray.GPUArray.__mul__ = patch_binop('*', '__mul__')
    pycuda.gpuarray.GPUArray.__div__ = patch_binop('/', '__div__')
    pycuda.gpuarray.GPUArray.__truediv__ = pycuda.gpuarray.GPUArray.__div__


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
        self.bunch = bunch

    def __enter__(self):
        '''Remove slice records from bunch.'''
        self.bunch.clean_slices()

    def __exit__(self, exc_type, exc_value, traceback):
        '''Remove slice records from bunch.'''
        self.bunch.clean_slices()


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

        Remove slice records from bunch.
        '''
        self.bunch.clean_slices()
        for coord in self.to_move:
            obj = getattr(self.bunch, coord, None)
            if isinstance(obj, np.ndarray):
                setattr(self.bunch, coord, gpuarray.to_gpu(obj,
                        gpu_utils.memory_pool.allocate))

        # replace functions in general.math.py
        pm.update_active_dict(pm._GPU_func_dict)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        Move all data back to the CPU (and un-patch the methods?)
        Reestablish state of everything as it was before entering

        Remove slice records from bunch.
        '''
        self.bunch.clean_slices()
        for coord in self.to_move:
            obj = getattr(self.bunch, coord, None)
            if isinstance(obj, pycuda.gpuarray.GPUArray):
                setattr(self.bunch, coord, obj.get())
        pm.update_active_dict(pm._CPU_numpy_func_dict)
