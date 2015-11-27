'''
@author Stefan Hegglin
@date 20.10.2015
Python functions which wrap GPU functionality.
Use in dispatch of general/pmath
All functions assume GPU arrays as arguments!
'''
from __future__ import division
import numpy as np
import os
import gpu_utils
try:
    import skcuda.misc
    import pycuda.gpuarray
    import pycuda.compiler
    import pycuda.driver as drv
    import thrust_interface
    import pycuda.elementwise

    # if pycuda is there, try to compile things. If no context available,
    # throw error to tell the user that he should import pycuda.autoinit
    # at the beginning of the script if he wants to use cuda functionalities
    try:
        ### Thrust
        thrust = thrust_interface.compiled_module

        ### CUDA Kernels
        where = os.path.dirname(os.path.abspath(__file__)) + '/'
        with open(where + 'stats.cu') as stream:
            source = stream.read()
        stats_kernels = pycuda.compiler.SourceModule(source) # compile
        sorted_mean_per_slice_kernel = stats_kernels.get_function('sorted_mean_per_slice')
        sorted_std_per_slice_kernel = stats_kernels.get_function('sorted_std_per_slice')
        sorted_cov_per_slice_kernel = stats_kernels.get_function('sorted_cov_per_slice')
        has_pycuda = True
    except pycuda._driver.LogicError: #the error pycuda throws if no context initialized
        print ('No context initialized. Please import pycuda.autoinit at the '
               'beginning of your script if you want to use GPU functionality')
        has_pycuda=False


except ImportError:
    print 'Either pycuda, skcuda or thrust not found! No GPU capabilites available'

if has_pycuda:
    _sub_1dgpuarr = pycuda.elementwise.ElementwiseKernel(
        'double* out, double* a, const double* b',
        'out[i] = a[i] - b[0]',
        '_sub_1dgpuarr'
    )
    def sub_scalar(gpuarr, scalar):
        out = pycuda.gpuarray.empty_like(gpuarr)
        _sub_1dgpuarr(out, gpuarr, scalar)
        return out

    _mul_1dgpuarr = pycuda.elementwise.ElementwiseKernel(
        'double* out, double* x, const double* a',
        'out[i] = a[0] * x[i]',
        '_mul_1dgpuarr'
    )


def _inplace_pow(x_gpu, p, stream=None):
    '''
    Performs an in-place x_gpu = x_gpu ** p
    Courtesy: scikits.cuda
    '''
    func = pycuda.elementwise.get_pow_kernel(x_gpu.dtype)
    func.prepared_async_call(x_gpu._grid, x_gpu._block, stream,
        p, x_gpu.gpudata, x_gpu.gpudata, x_gpu.mem_size)



def covariance_old(a, b):
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

def covariance(a,b, stream=None):
    '''Covariance (not covariance matrix)
    Args:
        a: pycuda.GPUArray
        b: pycuda.GPUArray
    '''
    n = len(a)
    x = pycuda.gpuarray.empty_like(a)
    y = pycuda.gpuarray.empty_like(b)
    mean_a = skcuda.misc.mean(a)
    #x -= mean_a
    _sub_1dgpuarr(x, a, mean_a, stream=stream)
    mean_b = skcuda.misc.mean(b)
    #y -= mean_b
    _sub_1dgpuarr(y, b, mean_b, stream=stream)
    covariance = skcuda.misc.mean(x * y) * (n / (n + 1))
    return covariance

def std(a):
    '''Std of a vector'''
    #return skcuda.misc.std(a, ddof=1)
    n = len(a)
    mean_a = skcuda.misc.mean(a)
    x = pycuda.gpuarray.empty_like(a)
    _sub_1dgpuarr(x, a, mean_a)
    _inplace_pow(x, 2)
    res =  pycuda.gpuarray.sum(x) / (n - 1)
    _inplace_pow(res, 0.5)
    return res

def emittance_reference(u, up, dp):
    '''
    Compute the emittance of GPU arrays. Reference implementation, slow
    but readable
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
    sigma12 = sigma12.get()
    return np.sqrt(sigma11.get() * sigma22.get() - sigma12 * sigma12)

def emittance_getasync(u, up, dp):
    '''
    Compute the emittance of GPU arrays. Test with streams. Not faster than
    version below.
    Args:
        u coordinate array
        up conjugate momentum array
        dp longitudinal momentum variation
    '''

    n = len(u)
    sigma11 = 0.
    sigma12 = 0.
    sigma22 = 0.
    cov_u_dp = 0.
    cov_up_dp = 0.
    cov_dp2 = 1.
    mean_u = skcuda.misc.mean(u)
    mean_up = skcuda.misc.mean(up)

    tmp_u = sub_scalar(u, mean_u)
    tmp_up = sub_scalar(up, mean_up)
    cov_u2 = skcuda.misc.mean(tmp_u*tmp_u).get_async(stream=gpu_utils.streams[0]) * (n / (n + 1.))
    cov_u_up = skcuda.misc.mean(tmp_u*tmp_up).get_async(stream=gpu_utils.streams[1]) * (n / (n + 1.))
    cov_up2 = skcuda.misc.mean(tmp_up*tmp_up).get_async(stream=gpu_utils.streams[2]) * (n / (n + 1.))

    if dp is not None: #if not None, assign values to variables involving dp
        mean_dp = skcuda.misc.mean(dp)
        tmp_dp = sub_scalar(dp, mean_dp)
        cov_u_dp = skcuda.misc.mean(tmp_u*tmp_dp).get_async(stream=gpu_utils.streams[0]) * (n / (n + 1.))
        cov_up_dp = skcuda.misc.mean(tmp_up*tmp_dp).get_async(stream=gpu_utils.streams[1]) * (n / (n + 1.))
        cov_dp2 = skcuda.misc.mean(tmp_dp*tmp_dp).get_async(stream=gpu_utils.streams[2]) * (n / (n + 1.))
    for i in xrange(3):
        gpu_utils.streams[i].synchronize()
    sigma11 = cov_u2 - cov_u_dp*cov_u_dp/cov_dp2
    sigma12 = cov_u_up - cov_u_dp*cov_up_dp/cov_dp2
    sigma22 = cov_up2 - cov_up_dp*cov_up_dp/cov_dp2
    return np.sqrt(sigma11 * sigma22 - sigma12*sigma12)


def emittance(u, up, dp):
    '''
    Compute the emittance of GPU arrays. Check the algorithm above for
    a more readable version, this one has been 'optimized', e.g. mean->sum
    and multiplication at the end to minimize kernel calls/inits of gpuarrs
    Args:
        u coordinate array
        up conjugate momentum array
        dp longitudinal momentum variation
    '''
    n = len(u)
    mean_u = skcuda.misc.mean(u)
    mean_up = skcuda.misc.mean(up)
    tmp_u = sub_scalar(u, mean_u)
    tmp_up = sub_scalar(up, mean_up)
    cov_u2 = pycuda.gpuarray.sum(tmp_u * tmp_u)
    cov_u_up = pycuda.gpuarray.sum(tmp_u * tmp_up)
    cov_up2 = pycuda.gpuarray.sum(tmp_up * tmp_up)
    if dp is not None: #if not None, assign values to variables involving dp
        mean_dp = skcuda.misc.mean(dp)
        tmp_dp = sub_scalar(dp, mean_dp)
        cov_u_dp = pycuda.gpuarray.sum(tmp_u * tmp_dp)
        cov_up_dp = pycuda.gpuarray.sum(tmp_up * tmp_dp)
        cov_dp2 = pycuda.gpuarray.sum(tmp_dp * tmp_dp)

        sigma11 = cov_u2 - cov_u_dp*cov_u_dp/cov_dp2
        sigma12 = cov_u_up - cov_u_dp*cov_up_dp/cov_dp2
        sigma22 = cov_up2 - cov_up_dp*cov_up_dp/cov_dp2
    else:
        sigma11 = cov_u2
        sigma12 = cov_u_up
        sigma22 = cov_up2
    return pycuda.cumath.sqrt((1./(n*n+n))*(sigma11 * sigma22 - sigma12*sigma12))

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

def particles_within_cuts(sliceset):
    '''
    Returns np.where((array >= minimum) and (array <= maximum))
    Assumes a sorted beam!
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)
    idx = pycuda.gpuarray.arange(sliceset.pidx_begin, sliceset.pidx_end, dtype=np.int32)
    return idx

def macroparticles_per_slice(sliceset):
    '''
    Returns the number of macroparticles per slice. Assumes a sorted beam!
    '''
    # simple: upper_bounds - lower_bounds!
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)
    return sliceset.upper_bounds - sliceset.lower_bounds


def _add_bounds_to_sliceset(sliceset):
    '''
    Adds the lower_bounds and upper_bounds members to the sliceset
    They must not present before the function call, otherwise undefined behaviour
    '''
    seq = pycuda.gpuarray.arange(sliceset.n_slices, dtype=np.int32)
    upper_bounds = pycuda.gpuarray.empty_like(seq)
    lower_bounds = pycuda.gpuarray.empty_like(seq)
    thrust.upper_bound_int(sliceset.slice_index_of_particle,
                                            seq, upper_bounds)
    thrust.lower_bound_int(sliceset.slice_index_of_particle,
                                            seq, lower_bounds)
    sliceset.upper_bounds = upper_bounds
    sliceset.lower_bounds = lower_bounds
    sliceset._pidx_begin = lower_bounds.get()[0] # set those properties now!
    sliceset._pidx_end = upper_bounds.get()[-1]  # this way .get() gets called only once
    #print 'upper bounds ',sliceset.upper_bounds
    #print 'lower bounds ',sliceset.lower_bounds

def sorted_mean_per_slice(sliceset, u, stream=None):
    '''
    Computes the mean per slice of the array u
    Args:
        sliceset specifying slices, has .nslices and .slice_index_of_particle
        u the array of which to compute the mean
    Returns the an array, res[i] stores the mean of slice i
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)

    block = (256, 1, 1)
    grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
    #!!! managed memory, requires comp. capability >=3.0 (not on TeslaC2075)!
    #mean_u = drv.managed_zeros(sliceset.n_slices, dtype=np.float64, mem_flags=drv.mem_attach_flags.GLOBAL)
    mean_u = pycuda.gpuarray.zeros(sliceset.n_slices, dtype=np.float64)
    sorted_mean_per_slice_kernel(sliceset.lower_bounds.gpudata,
                                 sliceset.upper_bounds.gpudata,
                                 u.gpudata, np.int32(sliceset.n_slices),
                                 mean_u.gpudata,
                                 block=block, grid=grid, stream=stream)
    return mean_u

def sorted_std_per_slice(sliceset, u, stream=None):
    '''
    Computes the cov per slice of the array u
    Args:
        sliceset specifying slices
        u the array of which to compute the cov
    Returns an array, res[i] stores the cov of slice i
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)
    block = (256, 1, 1)
    grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
    std_u = pycuda.gpuarray.zeros(sliceset.n_slices, dtype=np.float64)
    sorted_std_per_slice_kernel(sliceset.lower_bounds.gpudata,
                                sliceset.upper_bounds.gpudata,
                                u.gpudata, np.int32(sliceset.n_slices),
                                std_u.gpudata,
                                block=block, grid=grid, stream=stream)
    return std_u

def sorted_cov_per_slice(sliceset, u, v, stream=None):
    '''
    Computes the covariance of the quantities u,v per slice
    Args:
        sliceset specifying slices
        u, v the arrays of which to compute the covariance
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)
    block = (256, 1, 1)
    grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
    cov_uv = pycuda.gpuarray.zeros(sliceset.n_slices, dtype=np.float64)
    sorted_cov_per_slice_kernel(sliceset.lower_bounds.gpudata,
                                sliceset.upper_bounds.gpudata,
                                u.gpudata, v.gpudata,
                                np.int32(sliceset.n_slices),
                                cov_uv.gpudata,
                                block=block, grid=grid, stream=stream)
    return cov_uv

def sorted_emittance_per_slice(sliceset, u, up, dp=None):
    '''
    Computes the emittance per slice.
    If dp is None, the effective emittance is computed
    Args:
        sliceset specifying slices
        u, up the quantities of which to compute the emittance, e.g. x,xp
    '''
    ### computes the covariance on different streams
    n_streams = 3 #HARDCODED FOR NOW
    streams = [drv.Stream() for i in xrange(n_streams)]
    cov_u2 = sorted_cov_per_slice(sliceset, u, u, stream=streams[0])
    cov_up2= sorted_cov_per_slice(sliceset, up, up, stream=streams[1])
    cov_u_up = sorted_cov_per_slice(sliceset, u, up, stream=streams[2])
    if dp is not None:
        cov_u_dp = sorted_cov_per_slice(sliceset, u, dp, stream=streams[0])
        cov_up_dp= sorted_cov_per_slice(sliceset, up, dp, stream=streams[1])
        cov_dp2 = sorted_cov_per_slice(sliceset, dp, dp, stream=streams[2])
    else:
        cov_dp2 = pycuda.gpuarray.zeros_like(cov_u2) + 1.
        cov_u_dp = pycuda.gpuarray.zeros_like(cov_u2)
        cov_up_dp = pycuda.gpuarray.zeros_like(cov_u2)
    for i in xrange(n_streams):
        streams[i].synchronize()
    # TODO: change this to elementwise kernels or .mul_add()
    sigma11 = cov_u2 - cov_u_dp*cov_u_dp/cov_dp2
    sigma12 = cov_u_up - cov_u_dp*cov_up_dp/cov_dp2
    sigma22 = cov_up2 - cov_up_dp*cov_up_dp/cov_dp2
    emittance = pycuda.cumath.sqrt(sigma11*sigma22 - sigma12*sigma12)
    return emittance

def convolve(a, v, mode='full'):
    '''
    Compute the convolution of the two arrays a,v. See np.convolve
    '''
    #HACK: use np.convolve for now, make sure both arguments are np.arrays!
    try:
        a = a.get()
    except:
        pass
    try:
        v = v.get()
    except:
        pass
    c = np.convolve(a, v, mode)
    return pycuda.gpuarray.to_gpu(c)

def init_bunch_buffer(bunch, bunch_stats, buffer_size):
    '''Call bunch.[stats], match the buffer type with the returned type'''
    buf = {}
    for stats in bunch_stats:
        try:
            res = getattr(bunch, stats)()
        except TypeError:
            res = getattr(bunch, stats)
        if isinstance(res, pycuda.gpuarray.GPUArray):
            buf[stats] = pycuda.gpuarray.zeros(buffer_size,
                dtype=res.dtype, allocator=gpu_utils.memory_pool.allocate)
        else: # is already on CPU, e.g. macroparticlenumber
            buf[stats] = np.zeros(buffer_size, dtype=type(res))
    return buf

def init_slice_buffer(slice_set, slice_stats, buffer_size):
    '''Call sliceset.['stats'], match the buffer type with the returned type'''
    n_slices = slice_set.n_slices
    buf = {}
    for stats in slice_stats:
        res = getattr(slice_set, stats)
        if isinstance(res, pycuda.gpuarray.GPUArray):
            buf[stats] = pycuda.gpuarray.zeros(shape=(n_slices, buffer_size),
                dtype=res.dtype, allocator=gpu_utils.memory_pool.allocate)
        else: #already on CPU
            buf[stats] = np.zeros((n_slices, buffer_size), dtype=type(res))
    return buf
