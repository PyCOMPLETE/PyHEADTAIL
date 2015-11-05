'''
@author Stefan Hegglin
@date 20.10.2015
Python functions which wrap GPU functionality.
Use in dispatch of general/pmath
All functions assume GPU arrays as arguments!
'''
from __future__ import division
import numpy as np
try:
    import skcuda.misc
    import pycuda.gpuarray
    import pycuda.compiler
except ImportError:
    pass

### Thrust
import thrust_interface
thrust = thrust_interface.compiled_module

### CUDA Kernels
import os
where = os.path.dirname(os.path.abspath(__file__)) + '/'
with open(where + 'stats.cu') as stream:
    source = stream.read()
stats_kernels = pycuda.compiler.SourceModule(source) # compile
sorted_mean_per_slice_kernel = stats_kernels.get_function('sorted_mean_per_slice')
sorted_cov_per_slice_kernel = stats_kernels.get_function('sorted_cov_per_slice')


def covariance(a, b):
    '''Covariance (not covariance matrix)
    Args:
        a: pycuda.GPUArray
        b: pycuda.GPUArray
    '''
    n = len(a)
    mean_a = skcuda.misc.mean(a).get()
    x = a - mean_a
    mean_b = skcuda.misc.mean(b).get()
    y = b - mean_b
    covariance = skcuda.misc.mean(x * y) * n / (n + 1)
    return covariance.get()

def emittance(u, up, dp):
    '''
    Compute the emittance of GPU arrays
    Args:
        u coordinate array
        up conjugate momentum array
        dp longitudinal momentum variation
    '''
    sigma11 = 0.
    sigma12 = 0.
    sigma22 = 0.
    cov_u2 = covariance(u,u)
    cov_up2 = covariance(up, up)
    cov_u_up = covariance(up, u)
    cov_u_dp = 0.
    cov_up_dp = 0.
    cov_dp2 = 1.
    if dp is not None: #if not None, assign values to variables involving dp
        cov_u_dp = covariance(u, dp)
        cov_up_dp = covariance(up,dp)
        cov_dp2 = covariance(dp,dp)
    sigma11 = cov_u2 - cov_u_dp*cov_u_dp/cov_dp2
    sigma12 = cov_u_up - cov_u_dp*cov_up_dp/cov_dp2
    sigma22 = cov_up2 - cov_up_dp*cov_up_dp/cov_dp2
    sigma11 * sigma22 - sigma12 * sigma12
    return np.sqrt(sigma11 * sigma22 - sigma12 * sigma12)

def argsort(to_sort):
    '''
    Return the permutation required to sort the array.
    Args:
        to_sort gpuarray for which the permutation array to sort it is returned
    Returns the permutation
    '''
    dtype = to_sort.dtype
    permutation = pycuda.gpuarray.empty(to_sort.shape, dtype=np.int32)
    if dtype.itemsize == 8 and dtype.kind is 'f':
        thrust.get_sort_perm_double(to_sort.copy(), permutation)
    elif dtype.itemsize == 4 and dtype.kind is 'i':
        thrust.get_sort_perm_int(to_sort.copy(), permutation)
    else:
        print array.dtype
        print array.dtype.itemsize
        print array.dtype.kind
        raise TypeError('Currently only float64 and int32 types can be sorted')
    return permutation

def apply_permutation(array, permutation):
    '''
    Permute the entries in array according to the permutation array.
    Returns a new (permuted) array which is equal to array[permutation]
    Args:
        array gpuarray to be permuted. Either float64 or int32
        permutation permutation array: must be np.int32 (or int32), is asserted
    '''
    assert(permutation.dtype.itemsize == 4 and permutation.dtype.kind is 'i')
    tmp = pycuda.gpuarray.empty_like(array)
    dtype = array.dtype
    if dtype.itemsize == 8 and dtype.kind is 'f':
        thrust.apply_sort_perm_double(array, tmp, permutation)
    elif dtype.itemsize == 4 and dtype.kind is 'i':
        thrust.apply_sort_perm_int(array, tmp, permutation)
    else:
        print array.dtype
        print array.dtype.itemsize
        print array.dtype.kind
        raise TypeError('Currently only float64 and int32 types can be sorted')
    return tmp

def sorted_mean_per_slice(sliceset, u):
    '''
    Computes the mean per slice of the array u
    Args:
        sliceset specifying slices
        u the array of which to compute the mean
    Returns the an array, res[i] stores the mean of slice i
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        #print 'Adding the upper/lower_bounds to the sliceset'
        #print 'Assuming a sorted beam'
        seq = pycuda.gpuarray.arange(sliceset.n_slices, dtype=np.int32)
        upper_bounds = pycuda.gpuarray.empty_like(seq)
        lower_bounds = pycuda.gpuarray.empty_like(seq)
        thrust.upper_bound_int(sliceset.slice_index_of_particle,
                                                seq, upper_bounds)
        thrust.lower_bound_int(sliceset.slice_index_of_particle,
                                                seq, lower_bounds)
        sliceset.upper_bounds = upper_bounds
        sliceset.lower_bounds = lower_bounds
    #print 'upper bounds ',sliceset.upper_bounds
    #print 'lower bounds ',sliceset.lower_bounds
    block = (256, 1, 1)
    grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
    mean_u = pycuda.gpuarray.zeros(sliceset.n_slices, dtype=np.float64) + 99.
    sorted_mean_per_slice_kernel(sliceset.lower_bounds.gpudata,
                                 sliceset.upper_bounds.gpudata,
                                 u.gpudata, np.int32(sliceset.n_slices),
                                 mean_u.gpudata,
                                 block=block, grid=grid)
    return mean_u
