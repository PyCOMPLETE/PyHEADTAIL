'''
@author Stefan Hegglin, Adrian Oeftiger
@date 20.10.2015
Python functions which wrap GPU functionality.
Use in dispatch of general/pmath
All functions assume GPU arrays as arguments!
'''
from __future__ import division
import numpy as np
import os
import gpu_utils
import math
from functools import wraps
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
        thrust = thrust_interface

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
        print ('Warning: GPU is in principle available but no context has been '
               'initialized. Please import pycuda.autoinit at the '
               'beginning of your script before importing PyHEADTAIL '
               'if you want to use GPU functionality.\n')
        has_pycuda = False

except ImportError as e:
    # print ('Either pycuda, skcuda or thrust not found! '
    #        'No GPU capabilities available')
    has_pycuda = False

def _empty_like(gpuarray):
    return pycuda.gpuarray.empty(
        shape=gpuarray.shape, dtype=gpuarray.dtype,
        allocator=gpu_utils.memory_pool.allocate)


if has_pycuda:
    # define all compilation depending functions (e.g. ElementwiseKernel)
    _sub_1dgpuarr = pycuda.elementwise.ElementwiseKernel(
        'double* out, double* a, const double* b',
        'out[i] = a[i] - b[0]',
        '_sub_1dgpuarr'
    )
    def sub_scalar(gpuarr, scalar, out=None, stream=None):
        if out is None:
            out = _empty_like(gpuarr)
        _sub_1dgpuarr(out, gpuarr, scalar, stream=stream)
        return out

    _mul_with_factor = pycuda.elementwise.ElementwiseKernel(
        'double* out, double *in, const double c',
        'out[i] = c * in[i]',
        '_mul_with_factor'
    )
    def _mul_scalar(gpuarr, scalar, out=None, stream=None):
        '''Multiply the gpuarr by a scalar, use this if you need
        to specify a stream
        '''
        if out is None:
            out = _empty_like(gpuarr)
        _mul_with_factor(out, gpuarr, scalar, stream=stream)

    def _multiply(a, b, out=None, stream=None):
        '''Elementwise multiply of two gpuarray specifying a stream
        Required because gpuarray.__mul__ has no stream argument'''
        if out is None:
            out = _empty_like(a)
        func = pycuda.elementwise.get_binary_op_kernel(a.dtype, b.dtype,
            out.dtype, "*")
        func.prepared_async_call(a._grid, a._block, stream, a.gpudata,
            b.gpudata, out.gpudata, a.mem_size)
        return out

    _arange_gpu_float64 = pycuda.elementwise.ElementwiseKernel(
        'double* out, double* start, const double* step',
        'const double s = step[0];'
        'out[i] = start[0] + i * s',
        '_arange_gpu_float64'
    )
    _arange_gpu_int32 = pycuda.elementwise.ElementwiseKernel(
        'int* out, int* start, const int* step',
        'const int s = step[0];'
        'out[i] = start[0] + i * s',
        '_arange_gpu_int32'
    )
    def arange_startstop_gpu(start_gpu, stop_gpu, step_gpu, n_slices_cpu,
                             dtype=np.float64, stream=None):
        if dtype is np.float64:
            out = pycuda.gpuarray.empty(n_slices_cpu, dtype=np.float64,
                allocator=gpu_utils.memory_pool.allocate)
            _arange_gpu_float64(out, start_gpu, step_gpu, stream=stream)
        elif dtype is np.int32:
            out = pycuda.gpuarray.empty(n_slices_cpu, dtype=np.int32,
                allocator=gpu_utils.memory_pool.allocate)
            _arange_gpu_int32(out, start_gpu, step_gpu, stream=stream)
        else:
            raise TypeError('currently only np.float64 and np.int32 supported.')
        return out

    def arange(start, stop, step=1, n_slices=None, dtype=np.float64,
               stream=None):
        """Create an array filled with numbers spaced `step` apart,
        starting from `start` and ending at `stop`.

        For floating point arguments, the length of the result is
        `ceil((stop - start)/step)`.  This rule may result in the last
        element of the result being greater than stop.
        """
        if n_slices is None:
            n_slices = int(np.ceil((stop - start) / step))
        if isinstance(start, pycuda.gpuarray.GPUArray):
            return arange_startstop_gpu(start, stop, step, n_slices, dtype)
        else:
            return pycuda.gpuarray.arange(start, stop, step, dtype=dtype)

    _comp_sigma = pycuda.elementwise.ElementwiseKernel(
        'double* out, double* a, double* b, double* c, double* d',
        'out[i] = a[i] - b[i]*c[i]/d[i]',
        '_comp_sigma'
    )
    def _compute_sigma(a, b, c, d, out=None, stream=None):
        '''Computes elementwise a - b*c/d as required in compute sigma for
        the emittance '''
        if out is None:
            out = _empty_like(a)
        _comp_sigma(out, a, b, c, d, stream=stream)
        return out

    _mul_1dgpuarr = pycuda.elementwise.ElementwiseKernel(
        'double* out, double* x, const double* a',
        'out[i] = a[0] * x[i]',
        '_mul_1dgpuarr'
    )
    _emitt_disp = pycuda.elementwise.ElementwiseKernel(
        arguments='double* out, double* cov_u2, double* cov_u_up, '
                  'double* cov_up2, double* cov_u_dp, double* cov_up_dp, '
                  'double* cov_dp2, double nn',
        #operation='out[i] = nn',
        operation='double sigma11 = cov_u2[i]   - cov_u_dp[i] *cov_u_dp[i] / cov_dp2[i];'
                  'double sigma12 = cov_u_up[i] - cov_u_dp[i] *cov_up_dp[i]/ cov_dp2[i];'
                  'double sigma22 = cov_up2[i]  - cov_up_dp[i]*cov_up_dp[i]/ cov_dp2[i];'
                  'out[i] = sqrt((1./(nn*nn+nn))*(sigma11 * sigma22 - sigma12*sigma12))',
        name='_emitt_disp',
    )
    def _emittance_dispersion(
            n, cov_u2, cov_u_up, cov_up2, cov_u_dp, cov_up_dp,
            cov_dp2, out=None, stream=None):
        if out is None:
            out = _empty_like(cov_u2)
        _emitt_disp(out, cov_u2, cov_u_up, cov_up2, cov_u_dp, cov_up_dp,
                    cov_dp2, np.float64(n), stream=stream)
        return out

    _emitt_nodisp = pycuda.elementwise.ElementwiseKernel(
        arguments='double* out, double* cov_u2, double* cov_u_up, '
                  'double* cov_up2, double nn',
        operation=
            'out[i] = sqrt((1./(nn*nn+nn)) * '
                     '(cov_u2[i] * cov_up2[i] - cov_u_up[i]*cov_u_up[i]))',
        name='_emitt_nodisp'
    )
    def _emittance_no_dispersion(
            n, cov_u2, cov_u_up, cov_up2, out=None, stream=None):
        if out is None:
            out = _empty_like(cov_u2)
        _emitt_nodisp(out, cov_u2, cov_u_up, cov_up2, np.float64(n),
                      stream=stream)
        return out

    _wofz = pycuda.elementwise.ElementwiseKernel(
        arguments='double* in_real, double* in_imag, double* out_real, '
                  'double* out_imag',
        operation='wofz(in_real[i], in_imag[i], &out_real[i], &out_imag[i]);',
        name='wofz_kernel',
        preamble=open(where + 'wofz.cu', 'r').read()
    )
    def wofz(in_real, in_imag, out_real=None, out_imag=None, stream=None):
        '''Faddeeva error function, equivalent to scipy.special.wofz.
        Instead of a complex argument z, it takes the real and imaginary
        part of z.
        '''
        if out_real is None:
            out_real = _empty_like(in_real)
        if out_imag is None:
            out_imag = _empty_like(in_imag)
        _wofz(in_real, in_imag, out_real, out_imag, stream=stream)
        return out_real, out_imag

    _sign = pycuda.elementwise.ElementwiseKernel(
        arguments='double* in, double* out',
        operation='out[i] = ((in[i]) > 0. ? +1. : ((in[i]) < 0. ? -1. : 0.));',
        name='sign_kernel',
    )
    def sign(array, out=None, stream=None):
        if out is None:
            out = _empty_like(array)
        _sign(array, out, stream=stream)
        return out

    _allclose = pycuda.elementwise.ElementwiseKernel(
        arguments='double* a, double* b, int* notclose, '
                  'const double atol, const double rtol',
        # max(x, -x) is the quickest implementation of abs, cf. CUDA doc
        operation='const double absdiff = max(a[i] - b[i], b[i] - a[i]);'
                  'const double limit = atol + rtol * max(b[i], -b[i]);'
                  'if (absdiff > limit) notclose[i] = 1;'
                  'else notclose[i] = 0;',
        name='allclose_kernel'
    )
    np_allclose_defaults = np.allclose.func_defaults # (rtol, atol, equal_nan)
    @wraps(np.allclose)
    def allclose(a, b, rtol=np_allclose_defaults[0],
                 atol=np_allclose_defaults[1], out=None, stream=None):
        assert a.shape == b.shape
        if out is None:
            out = pycuda.gpuarray.empty(a.shape, dtype=np.int32)
        _allclose(a, b, out, atol, rtol)
        how_many_not_close = pycuda.gpuarray.sum(out).get()
        return how_many_not_close == 0

    # assigns stat_noncontained entries to stat_contained
    # only if the respective slice_ids entry is within the slicing region
    _reassign_valid = pycuda.elementwise.ElementwiseKernel(
        arguments="const int* slice_ids_noncontained, "
                  "const double* stat_noncontained, "
                  "const int n_slices, " # end inputs
                  "double* stat_contained", # output
        operation="if (0 <= slice_ids_noncontained[i] "
                      "&& slice_ids_noncontained[i] < n_slices) "
                  "stat_contained[slice_ids_noncontained[i]] = stat_noncontained[i];"
    )
    def thrust_mean_and_std_per_slice(sliceset, u, stream=None):
        '''Compute mean and standard deviation per slice
        using the thrust reduce_by_key reduction.
        Args:
            - sliceset: SliceSet instance specifying slices, has .n_slices
              and .slice_index_of_particle
            - u: the array of which to compute the mean and std

        Return mean_u and sigma_u.
        Does not make use of the thrust streams yet...

        N.B.: memory-hungry as it allows all particles to lie outside of
        slicing region.
        '''
        p_sids = sliceset.slice_index_of_particle
        # slice_index_of_particle may have slice indices outside of slicing area,
        # the following arrays therefore can comprise non valid slice entries
        slice_ids_noncontained = _empty_like(p_sids)
        slice_means_noncontained = _empty_like(u)
        slice_stds_noncontained = _empty_like(u)

        (_, _, _, new_end) = thrust.thrust_stats_per_slice(
            p_sids, u, slice_ids_noncontained, slice_means_noncontained,
            slice_stds_noncontained)

        mean_u = pycuda.gpuarray.zeros(sliceset.n_slices, dtype=np.float64,
                                       allocator=gpu_utils.memory_pool.allocate)
        sigma_u = pycuda.gpuarray.zeros(sliceset.n_slices, dtype=np.float64,
                                        allocator=gpu_utils.memory_pool.allocate)
        _reassign_valid(slice_ids_noncontained[:new_end],
                        slice_means_noncontained[:new_end],
                        sliceset.n_slices, mean_u)
        _reassign_valid(slice_ids_noncontained[:new_end],
                        slice_stds_noncontained[:new_end],
                        sliceset.n_slices, sigma_u)
        return (mean_u, sigma_u)


    _slice_to_particles = pycuda.elementwise.ElementwiseKernel(
        "const int* slice_index_of_particle, "
        "const double* input_slice_quantity, "
        "const int n_slices, double* output_particle_array",
        # i is the particle index within slice_index_of_particle
        "const int sid = slice_index_of_particle[i];\n"
        "if (sid >= 0 && sid < n_slices) {"
            "output_particle_array[i] = "
                "input_slice_quantity[sid];"
        "} else {"
            "output_particle_array[i] = 0;"
        "};",
        "slice_to_particles_kernel"
    )
    def slice_to_particles(sliceset, slice_array, particle_array=None):
        '''Convert slice_array with entries for each slice to a
        particle array with the respective entry of each particle
        given by its slice_array value via the slice that the
        particle belongs to. If the particle is located outside
        the slicing area, its particle_array entry is assigned to zero.
        '''
        if particle_array == None:
            particle_array = pycuda.gpuarray.empty(
                sliceset.slice_index_of_particle.shape,
                dtype=np.float64, allocator=gpu_utils.memory_pool.allocate)
        _slice_to_particles(sliceset.slice_index_of_particle,
                            slice_array, sliceset.n_slices, particle_array)
        return particle_array


def _inplace_pow(x_gpu, p, stream=None):
    '''
    Perform an in-place x_gpu = x_gpu ** p
    Courtesy: scikits.cuda
    '''
    func = pycuda.elementwise.get_pow_kernel(x_gpu.dtype)
    func.prepared_async_call(x_gpu._grid, x_gpu._block, stream,
        p, x_gpu.gpudata, x_gpu.gpudata, x_gpu.mem_size)

def atleast_1d(*arrays):
    '''Return input arrays unless they are scalars. Ensure the results have
    ndim >= 1.
    '''
    results = []
    for array in arrays:
        if len(array.shape) == 0:
            results.append(array.reshape((1,)))
        else:
            results.append(array)
    if len(results) == 1:
        return results[0]
    else:
        return results

def sincos(array):
    '''Return a tuple with the sin and the cos of the input array.'''
    # CUDA indeed has a sincos function which should be faster than
    # separate sin and cos calls, however it is not exposed in PyCUDA...
    return pycuda.cumath.sin(array), pycuda.cumath.cos(array)



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
    x = _empty_like(a)
    y = _empty_like(b)
    mean_a = skcuda.misc.mean(a)
    #x -= mean_a
    _sub_1dgpuarr(x, a, mean_a, stream=stream)
    mean_b = skcuda.misc.mean(b)
    #y -= mean_b
    _sub_1dgpuarr(y, b, mean_b, stream=stream)
    covariance = skcuda.misc.mean(x * y) * (n / (n + 1))
    return covariance

def mean(a, stream=None):
    '''Compute the mean of the gpuarray a
    Replacement for skcuda.misc.mean(), which does not allow to specify
    the stream (because gpuarray.__div__ does not have a stream
    argument).
    '''
    #out = pycuda.gpuarray.empty(1, dtype=np.float64, allocator=gpu_utils.memory_pool.allocate)
    n = len(a)
    out = pycuda.gpuarray.sum(a, stream=stream,
                              allocator=gpu_utils.memory_pool.allocate)
    _mul_scalar(out=out, gpuarr=out, scalar=np.float64(1./n), stream=stream)
    return out

def std(a, stream=None):
    '''Std of a vector'''
    #return skcuda.misc.std(a, ddof=1)
    n = len(a)
    #mean_a = skcuda.misc.mean(a)
    x = _empty_like(a)
    mean_a = mean(a, stream=stream)
    _sub_1dgpuarr(x, a, mean_a, stream=stream)
    _inplace_pow(x, 2, stream=stream)
    res =  pycuda.gpuarray.sum(x, stream=stream)
    _mul_scalar(out=res, gpuarr=res, scalar=np.float64(1./(n-1)), stream=stream)
    _inplace_pow(res, 0.5, stream=stream)
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


def emittance_(u, up, dp):
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


def emittance(u, up, dp, stream=None):
    '''
    Compute the emittance of GPU arrays. Check the algorithm above for
    a more readable version, this one has been 'optimized', e.g. mean->sum
    and multiplication at the end to minimize kernel calls/inits of gpuarrs
    Args:
        u coordinate array
        up conjugate momentum array
        dp longitudinal momentum variation
        stream: In which cuda stream to perform the computations
    '''
    n = len(u)
    mean_u = mean(u, stream=stream)
    mean_up = mean(up, stream=stream)
    out = _empty_like(mean_u)
    tmp_u = sub_scalar(u, mean_u, stream=stream)
    tmp_up = sub_scalar(up, mean_up, stream=stream)
    tmp_space = _multiply(tmp_u, tmp_u, stream=stream)
    cov_u2 = pycuda.gpuarray.sum(tmp_space, stream=stream)
    tmp_space = _multiply(tmp_u, tmp_up, out=tmp_space, stream=stream) #specify out to reuse memory, the stream implicitly serializes everything s.t. nothing bad happens...
    cov_u_up = pycuda.gpuarray.sum(tmp_space, stream=stream)
    tmp_space = _multiply(tmp_up, tmp_up, out=tmp_space, stream=stream)
    cov_up2 = pycuda.gpuarray.sum(tmp_space, stream=stream)
    if dp is not None: #if not None, assign values to variables involving dp
        mean_dp = mean(dp, stream=stream)
        tmp_dp = sub_scalar(dp, mean_dp, stream=stream)
        tmp_space = _multiply(tmp_u, tmp_dp, out=tmp_space, stream=stream)
        cov_u_dp = pycuda.gpuarray.sum(tmp_space, stream=stream)
        tmp_space = _multiply(tmp_up, tmp_dp, out=tmp_space, stream=stream)
        cov_up_dp = pycuda.gpuarray.sum(tmp_space, stream=stream)
        tmp_space = _multiply(tmp_dp, tmp_dp, out=tmp_space, stream=stream)
        cov_dp2 = pycuda.gpuarray.sum(tmp_space, stream=stream)
        #em = _emittance_dispersion(n, cov_u2, cov_u_up, cov_up2, cov_u_dp, cov_up_dp, cov_dp2, out=out, stream=stream)
        _emitt_disp(out, cov_u2, cov_u_up, cov_up2, cov_u_dp, cov_up_dp, cov_dp2,np.float64(n), stream=stream)
    else:
        #em = _emittance_no_dispersion(n, cov_u2, cov_u_up, cov_up2, out=out,stream=stream)
        _emitt_nodisp(out, cov_u2, cov_u_up, cov_up2, np.float64(n), stream=stream)
    return out

# spawn multiple streams for each direction/mean/...
# is not faster than above, since the mean() functions use up all the kernels!
def emittance_multistream(u, up, dp, stream=None):
    '''
    Compute the emittance of GPU arrays. Check the algorithm above for
    a more readable version, this one has been 'optimized', e.g. mean->sum
    and multiplication at the end to minimize kernel calls/inits of gpuarrs
    Args:
        u coordinate array
        up conjugate momentum array
        dp longitudinal momentum variation
        stream: In which cuda stream to perform the computations
    '''
    n = len(u)
    streams = gpu_utils.stream_emittance
    mean_u = mean(u, stream=streams[0])
    mean_up = mean(up, stream=streams[1])
    tmp_u = sub_scalar(u, mean_u, stream=streams[0])
    tmp_space = _multiply(tmp_u, tmp_u, stream=streams[0])
    cov_u2 = pycuda.gpuarray.sum(tmp_space, stream=streams[0])
    out = _empty_like(mean_u)
    tmp_up = sub_scalar(up, mean_up, stream=streams[1])
    streams[0].synchronize()
    streams[1].synchronize()
    tmp_space = _multiply(tmp_u, tmp_up, out=tmp_space, stream=stream) #specify out to reuse memory, the stream implicitly serializes everything s.t. nothing bad happens...
    cov_u_up = pycuda.gpuarray.sum(tmp_space, stream=stream)
    tmp_space = _multiply(tmp_up, tmp_up, out=tmp_space, stream=stream)
    cov_up2 = pycuda.gpuarray.sum(tmp_space, stream=stream)
    if dp is not None: #if not None, assign values to variables involving dp
        mean_dp = mean(dp, stream=streams[2])
        tmp_dp = sub_scalar(dp, mean_dp, stream=streams[2])
        streams[2].synchronize()
        tmp_space = _multiply(tmp_u, tmp_dp, out=tmp_space, stream=stream)
        cov_u_dp = pycuda.gpuarray.sum(tmp_space, stream=stream)
        tmp_space = _multiply(tmp_up, tmp_dp, out=tmp_space, stream=stream)
        cov_up_dp = pycuda.gpuarray.sum(tmp_space, stream=stream)
        tmp_space = _multiply(tmp_dp, tmp_dp, out=tmp_space, stream=stream)
        cov_dp2 = pycuda.gpuarray.sum(tmp_space, stream=stream)
        #em = _emittance_dispersion(n, cov_u2, cov_u_up, cov_up2, cov_u_dp, cov_up_dp, cov_dp2, out=out, stream=stream)
        _emitt_disp(out, cov_u2, cov_u_up, cov_up2, cov_u_dp, cov_up_dp, cov_dp2,np.float64(n), stream=stream)
        gpu_utils.dummy_1(mean_u, stream=stream)
    else:
        #em = _emittance_no_dispersion(n, cov_u2, cov_u_up, cov_up2, out=out,stream=stream)
        _emitt_nodisp(out, cov_u2, cov_u_up, cov_up2, np.float64(n), stream=stream)
    return out

def cumsum(array, dest=None):
    '''Return cumulative sum of 1-dimensional GPUArray data.
    Works for dtypes np.int32 and np.float64. Wrapper for thrust
    prefix sum via thrust::inclusive_scan.
    '''
    if array.dtype == np.int32:
        if dest is None:
            dest = _empty_like(array)
        thrust_interface.thrust_cumsum_int(array, dest)
    elif array.dtype == np.float64:
        if dest is None:
            dest = _empty_like(array)
        thrust_interface.thrust_cumsum_double(array, dest)
    else:
        dest = array.copy()
        skcuda.misc.cumsum(dest)
    return dest

#@profile
def argsort(to_sort):
    '''
    Return the permutation required to sort the array.
    Args:
        to_sort: gpuarray for which the permutation array to sort
                 it is returned
    '''
    dtype = to_sort.dtype
    permutation = pycuda.gpuarray.empty(
        to_sort.shape, dtype=np.int32,
        allocator=gpu_utils.memory_pool.allocate)
    if dtype.itemsize == 8 and dtype.kind is 'f':
        thrust.get_sort_perm_double(to_sort.copy(), permutation)
    elif dtype.itemsize == 4 and dtype.kind is 'i':
        thrust.get_sort_perm_int(to_sort.copy(), permutation)
    else:
        print to_sort.dtype
        print to_sort.dtype.itemsize
        print to_sort.dtype.kind
        raise TypeError('Currently only float64 and int32 types can be sorted')
    return permutation

def searchsortedleft(array, values, dest_array=None):
    if dest_array is None:
        dest_array = pycuda.gpuarray.empty(
            shape=values.shape, dtype=np.int32,
            allocator=gpu_utils.memory_pool.allocate)
    thrust.lower_bound_int(array, values, dest_array)
    return dest_array

def searchsortedright(array, values, dest_array=None):
    if dest_array is None:
        dest_array = pycuda.gpuarray.empty(
            shape=values.shape, dtype=np.int32,
            allocator=gpu_utils.memory_pool.allocate)
    thrust.upper_bound_int(array, values, dest_array)
    return dest_array

def apply_permutation(array, permutation):
    '''
    Permute the entries in array according to the permutation array.
    Return a new (permuted) array which is equal to array[permutation]
    Args:
        array gpuarray to be permuted. Either float64 or int32
        permutation permutation array: must be np.int32 (or int32), is asserted
    '''
    assert(permutation.dtype.itemsize == 4 and permutation.dtype.kind is 'i')
    tmp = _empty_like(array)
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
    Return np.where((array >= minimum) and (array <= maximum))
    Assumes a sorted beam!
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)
    idx = arange(sliceset.pidx_begin, sliceset.pidx_end, dtype=np.int32)
    return idx

def particles_outside_cuts(sliceset):
    '''
    Return np.where((array < minimum) and (array > maximum))
    Assumes a sorted beam!
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)
    n_part_inside = sliceset.pidx_end - sliceset.pidx_begin
    n_part_outside = len(sliceset.slice_index_of_particle) - n_part_inside
    idx = arange(0, n_part_outside, dtype=np.int32)
    idx[sliceset.pidx_begin:] += n_part_inside
    return idx

def macroparticles_per_slice(sliceset):
    '''
    Return the number of macroparticles per slice. Assumes a sorted beam!
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
    seq = pycuda.gpuarray.arange(sliceset.n_slices, dtype=np.int32)# not supported in pycuda:, allocator=gpu_utils.memory_pool.allocate)
    upper_bounds = pycuda.gpuarray.empty(shape=seq.shape, dtype=np.int32, allocator=gpu_utils.memory_pool.allocate)
    lower_bounds = pycuda.gpuarray.empty(shape=seq.shape, dtype=np.int32, allocator=gpu_utils.memory_pool.allocate)
    thrust.upper_bound_int(sliceset.slice_index_of_particle,
                                            seq, upper_bounds)
    thrust.lower_bound_int(sliceset.slice_index_of_particle,
                                            seq, lower_bounds)
    sliceset.upper_bounds = upper_bounds
    sliceset.lower_bounds = lower_bounds
    sliceset._pidx_begin = lower_bounds[0].get() # set those properties now!
    sliceset._pidx_end = upper_bounds[-1].get()  # this way .get() gets called only once
    #print 'upper bounds ',sliceset.upper_bounds
    #print 'lower bounds ',sliceset.lower_bounds

def sorted_mean_per_slice(sliceset, u, stream=None):
    '''
    Computes the mean per slice of the array u
    Args:
        sliceset specifying slices, has .n_slices and .slice_index_of_particle
        u the array of which to compute the mean
    Return the an array, res[i] stores the mean of slice i
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)

    block = (256, 1, 1)
    grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
    #!!! managed memory, requires comp. capability >=3.0 (not on TeslaC2075)!
    #mean_u = drv.managed_zeros(sliceset.n_slices, dtype=np.float64, mem_flags=drv.mem_attach_flags.GLOBAL)
    mean_u = pycuda.gpuarray.empty(sliceset.n_slices, dtype=np.float64, allocator=gpu_utils.memory_pool.allocate)
    sorted_mean_per_slice_kernel(sliceset.lower_bounds.gpudata,
                                 sliceset.upper_bounds.gpudata,
                                 u.gpudata, np.int32(sliceset.n_slices),
                                 mean_u.gpudata,
                                 block=block, grid=grid, stream=stream)
    #pycuda.autoinit.context.synchronize()
    #gpu_utils.context.synchronize()
    return mean_u

def sorted_std_per_slice(sliceset, u, stream=None):
    '''
    Computes the cov per slice of the array u
    Args:
        sliceset specifying slices
        u the array of which to compute the cov
    Return an array, res[i] stores the cov of slice i
    '''
    if (not hasattr(sliceset, 'upper_bounds')) and (not hasattr(sliceset, 'lower_bounds')):
        _add_bounds_to_sliceset(sliceset)
    block = (256, 1, 1)
    grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
    std_u = pycuda.gpuarray.empty(sliceset.n_slices, dtype=np.float64, allocator=gpu_utils.memory_pool.allocate)
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
    cov_uv = pycuda.gpuarray.empty(sliceset.n_slices, dtype=np.float64, allocator=gpu_utils.memory_pool.allocate)
    sorted_cov_per_slice_kernel(sliceset.lower_bounds.gpudata,
                                sliceset.upper_bounds.gpudata,
                                u.gpudata, v.gpudata,
                                np.int32(sliceset.n_slices),
                                cov_uv.gpudata,
                                block=block, grid=grid, stream=stream)
    return cov_uv

def sorted_emittance_per_slice_slow(sliceset, u, up, dp=None, stream=None):
    '''
    Computes the emittance per slice.
    If dp is None, the effective emittance is computed
    Args:
        sliceset specifying slices
        u, up the quantities of which to compute the emittance, e.g. x,xp
    '''
    ### computes the covariance on different streams
    #n_streams = 3 #HARDCODED FOR NOW
    cov_u2 = sorted_cov_per_slice(sliceset, u, u, stream=stream)
    cov_up2= sorted_cov_per_slice(sliceset, up, up, stream=stream)
    cov_u_up = sorted_cov_per_slice(sliceset, u, up, stream=stream)
    if dp is not None:
        cov_u_dp = sorted_cov_per_slice(sliceset, u, dp, stream=stream)
        cov_up_dp= sorted_cov_per_slice(sliceset, up, dp, stream=stream)
        cov_dp2 = sorted_cov_per_slice(sliceset, dp, dp, stream=stream)
    else:
        cov_dp2 = pycuda.gpuarray.zeros_like(cov_u2) + 1.
        cov_u_dp = pycuda.gpuarray.zeros_like(cov_u2)
        cov_up_dp = pycuda.gpuarray.zeros_like(cov_u2)

    sigma11 = cov_u2 - cov_u_dp*cov_u_dp/cov_dp2
    sigma12 = cov_u_up - cov_u_dp*cov_up_dp/cov_dp2
    sigma22 = cov_up2 - cov_up_dp*cov_up_dp/cov_dp2
    emittance = pycuda.cumath.sqrt(sigma11*sigma22 - sigma12*sigma12)
    return emittance


def sorted_emittance_per_slice(sliceset, u, up, dp=None, stream=None):
    '''
    Computes the emittance per slice.
    If dp is None, the effective emittance is computed
    Args:
        sliceset specifying slices
        u, up the quantities of which to compute the emittance, e.g. x,xp
    '''
    ### computes the covariance on different streams
    streams = gpu_utils.stream_emittance
    cov_u2 = sorted_cov_per_slice(sliceset, u, u, stream=streams[0])
    cov_up2= sorted_cov_per_slice(sliceset, up, up, stream=streams[1])
    cov_u_up = sorted_cov_per_slice(sliceset, u, up, stream=streams[2])
    out = _empty_like(cov_u2)
    # use this factor in emitt_disp: the code has a 1/(n*n+n) factor which is not
    # required here since the scaling is done in the cov_per_slice
    # --> 1/(n*n + n) must be 1. ==> n = sqrt(5)/2 -0.5
    n = np.sqrt(5.)/2. - 0.5
    if dp is not None:
        cov_u_dp = sorted_cov_per_slice(sliceset, u, dp, stream=streams[3])
        cov_up_dp= sorted_cov_per_slice(sliceset, up, dp, stream=streams[4])
        cov_dp2 = sorted_cov_per_slice(sliceset, dp, dp, stream=streams[5])
        for s in streams:
            s.synchronize()
        _emitt_disp(out, cov_u2, cov_u_up, cov_up2, cov_u_dp, cov_up_dp, cov_dp2,np.float64(n), stream=stream)
    else:
        for s in [streams[0], streams[1], streams[2]]:
            s.synchronize()
        _emitt_nodisp(out, cov_u2, cov_u_up, cov_up2, np.float64(n), stream=stream)
    return out


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
