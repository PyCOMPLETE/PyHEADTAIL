import numpy as np

from libc.math cimport sin, cos
from libc.stdlib cimport malloc, free
# from libcpp.vector cimport vector

from cython.parallel import prange

cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def cm_sin(double[::1] x):

    cdef int i
    cdef int n = x.shape[0]
    cdef double[::1] s = np.zeros(n, dtype='float64')

    for i in prange(n, nogil=True, num_threads=1):
        s[i] = sin(x[i])

    return s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def cm_cos(double[::1] x):

    cdef int i
    cdef int n = x.shape[0]
    cdef double[::1] c = np.zeros(n, dtype='float64')

    for i in prange(n, nogil=True, num_threads=1):
        c[i] = cos(x[i])

    return c
