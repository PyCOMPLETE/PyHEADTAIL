import ctypes
import numpy as np
import os

_libthrustwrap = ctypes.cdll.LoadLibrary(
    os.path.dirname(os.path.abspath(__file__)) + '/thrust.so')

####thrust_lower_bound_int ####################################################
#void thrust_lower_bound_int(int* sorted_ptr, int sorted_length, int* bounds_ptr,
#                            int bounds_length, int* output_ptr)

_libthrustwrap.thrust_lower_bound_int.restype = None
_libthrustwrap.thrust_lower_bound_int.argtypes = [
    ctypes.c_void_p, #sorted_ptr
    ctypes.c_int, #sorted_length
    ctypes.c_void_p, #boudns_ptr
    ctypes.c_int,    #bounds_length
    ctypes.c_void_p #output_ptr
]
def lower_bound_int(position, bounds, out):
    '''
    Returns: nothing
    Args: Particle positions, mesh bounds, output array (GPUArrays)
    '''
    _libthrustwrap.thrust_lower_bound_int(int(position.gpudata),
                                          np.int32(len(position)),
                                          int(bounds.gpudata),
                                          np.int32(len(bounds)),
                                          int(out.gpudata))

#### thrust_upper_bound_int ####################################################
#void thrust_upper_bound_int(int* sorted_ptr, int sorted_length, int* bounds_ptr,
#                            int bounds_length, int* output_ptr)

_libthrustwrap.thrust_upper_bound_int.restype = None
_libthrustwrap.thrust_upper_bound_int.argtypes = [
    ctypes.c_void_p, #sorted_ptr
    ctypes.c_int, #sorted_length
    ctypes.c_void_p, #boudns_ptr
    ctypes.c_int,    #bounds_length
    ctypes.c_void_p #output_ptr
]
def upper_bound_int(position, bounds, out):
    '''
    Returns: nothing
    Args: Particle positions, mesh bounds, output array (GPUArrays)
    '''
    _libthrustwrap.thrust_upper_bound_int(int(position.gpudata),
                                          np.int32(len(position)),
                                          int(bounds.gpudata),
                                          np.int32(len(bounds)),
                                          int(out.gpudata))


#### thrust_apply_sort_perm_double #############################################
_libthrustwrap.thrust_apply_sort_perm_double.restype = None
_libthrustwrap.thrust_apply_sort_perm_double.argtypes = [
    ctypes.c_void_p, #in
    ctypes.c_int, #length
    ctypes.c_void_p, #out
    ctypes.c_void_p #perm
]
def apply_sort_perm_double(to_sort, out, permutation):
    '''
    Permutes the array to_sort using permutation and stores it into out
    Returns: nothing
    All arguments are GPUArrays with a double datatype
    '''
    _libthrustwrap.thrust_apply_sort_perm_double(int(to_sort.gpudata),
                                                 np.int32(len(to_sort)),
                                                 int(out.gpudata),
                                                 int(permutation.gpudata))


#### thrust_apply_sort_perm_double #############################################
_libthrustwrap.thrust_apply_sort_perm_int.restype = None
_libthrustwrap.thrust_apply_sort_perm_int.argtypes = [
    ctypes.c_void_p, #in
    ctypes.c_int, #length
    ctypes.c_void_p, #out
    ctypes.c_void_p #perm
]
def apply_sort_perm_int(to_sort, out, permutation):
    '''
    Permutes the array to_sort using permutation and stores it into out
    Returns: nothing
    All arguments are GPUArrays with an int datatype
    '''
    _libthrustwrap.thrust_apply_sort_perm_int(int(to_sort.gpudata),
                                              np.int32(len(to_sort)),
                                              int(out.gpudata),
                                              int(permutation.gpudata))

#### thrust_sort_double #############################################
#void thrust_sort_double(double* input_ptr, int length);
_libthrustwrap.thrust_sort_double.restype = None
_libthrustwrap.thrust_sort_double.argtypes = [
    ctypes.c_void_p, #input
    ctypes.c_int #len
]
def sort_double(array):
    '''
    Sort the (double) GPUArray array in place
    Returns: nothing
    '''
    _libthrustwrap.thrust_sort_double(int(array.gpudata), np.int32(len(array)))

#### thrust_sort_by_key_double #################################################
#void thrust_sort_by_key_double(double* key_ptr, int length, double* val_ptr)
_libthrustwrap.thrust_sort_by_key_double.restype = None
_libthrustwrap.thrust_sort_by_key_double.argtypes = [
    ctypes.c_void_p, #key
    ctypes.c_int, #len
    ctypes.c_void_p #val
]
def sort_by_key_double(key, value):
    '''
    Sorts the GPUArray value by the GPUArray key key[i] <-> value[i], double
    Returns: nothing
    '''
    _libthrustwrap.thrust_sort_by_key_double(int(key.gpudata),
                                             np.int32(len(key)),
                                             int(value.gpudata)
                                             )
#### thrust_get_sort_perm_double ###############################################
#void thrust_get_sort_perm_double(double* input_ptr, int length, int* perm_ptr)
_libthrustwrap.thrust_get_sort_perm_double.restype = None
_libthrustwrap.thrust_get_sort_perm_double.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p
]
def get_sort_perm_double(input, out):
    '''
    Sort the GPUArray (double) input and store the used permutation in out (int)
    '''
    _libthrustwrap.thrust_get_sort_perm_double(int(input.gpudata),
                                               np.int32(len(input)),
                                               int(out.gpudata))

#### thrust_get_sort_perm_int ###############################################
#void thrust_get_sort_perm_int(int* input_ptr, int length, int* perm_ptr)
_libthrustwrap.thrust_get_sort_perm_int.restype = None
_libthrustwrap.thrust_get_sort_perm_int.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p
]
def get_sort_perm_int(input, out):
    '''
    Sort the GPUArray (int) input and store the used permutation in out (int)
    '''
    _libthrustwrap.thrust_get_sort_perm_int(int(input.gpudata),
                                            np.int32(len(input)),
                                            int(out.gpudata))


#### thrust_cumsum_double ###############################################
#void thrust_cumsum_double(double* data_ptr, int length, double* sum_ptr)
_libthrustwrap.thrust_cumsum_double.restype = None
_libthrustwrap.thrust_cumsum_double.argtypes = [
    ctypes.c_void_p, #data_ptr
    ctypes.c_int, #data_length
    ctypes.c_void_p, #sum_ptr
]
def thrust_cumsum_double(data, out):
    '''Return cumulative sum of 1-dimensional GPUArray data with dtype
    np.float64.
    '''
    assert data.dtype == np.float64
    assert data.ndim == 1
    _libthrustwrap.thrust_cumsum_double(
        int(data.gpudata), np.int32(len(data)), int(out.gpudata))
    return out


#### thrust_cumsum_int ###############################################
#void thrust_cumsum_int(int* data_ptr, int length, int* sum_ptr)
_libthrustwrap.thrust_cumsum_int.restype = None
_libthrustwrap.thrust_cumsum_int.argtypes = [
    ctypes.c_void_p, #data_ptr
    ctypes.c_int, #data_length
    ctypes.c_void_p, #sum_ptr
]
def thrust_cumsum_int(data, out):
    '''Return cumulative sum of 1-dimensional GPUArray data with dtype
    np.int32.
    '''
    assert data.dtype == np.int32
    assert data.ndim == 1
    _libthrustwrap.thrust_cumsum_int(
        int(data.gpudata), np.int32(len(data)), int(out.gpudata))
    return out

_libthrustwrap.thrust_stats_per_slice.restype = None
_libthrustwrap.thrust_stats_per_slice.argtypes = [
    ctypes.c_void_p, #particle_slice_id_ptr
    ctypes.c_void_p, #u
    ctypes.c_int,    #int n_mp
    ctypes.c_void_p, #slice_id_ptr
    ctypes.c_void_p, #slice_mean_ptr
    ctypes.c_void_p, #slice_std_ptr
    ctypes.c_void_p, #n_relevant_entries
]
def thrust_stats_per_slice(particle_slice_ids, u, slice_ids,
                           slice_means, slice_stds):
    '''
    Calculate slice means and standard deviations using Chan et al.'s
    parallel algorithm (cf. https://en.wikipedia.org/wiki/
    Algorithms_for_calculating_variance#Parallel_algorithm).

    Arguments:
        - particle_slice_ids: GPUArray of dtype np.int32 with the
          slice ids of each particle (macroparticlenumber length)
        - u: GPUArray of dtype np.float64, coordinate or momentum
          array of the beam (x, x', ..., macroparticlenumber length)
    '''
    assert particle_slice_ids.dtype == slice_ids.dtype == np.int32
    assert u.dtype == slice_means.dtype == slice_stds.dtype == np.float64
    assert len(particle_slice_ids) == len(u)
    new_end = np.empty((), dtype=np.int32)
    _libthrustwrap.thrust_stats_per_slice(
        int(particle_slice_ids.gpudata),
        int(u.gpudata),
        np.int32(len(u)),
        int(slice_ids.gpudata),
        int(slice_means.gpudata),
        int(slice_stds.gpudata),
        int(new_end.ctypes.data),
    )
    return (slice_ids, slice_means, slice_stds, new_end)
