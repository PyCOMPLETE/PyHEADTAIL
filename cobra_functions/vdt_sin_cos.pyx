cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange


cdef extern from "sin.h" namespace "vdt":
    cdef double fast_sin(double x) nogil
    # cdef void fast_sinv(int n, double *x, double *s) nogil

cdef extern from "cos.h" namespace "vdt":
    cdef double fast_cos(double x) nogil
    # cdef void fast_cosv(int n, double *x, double *s) nogil

cdef extern from "sincos.h" namespace "vdt":
    cdef void fast_sincos(double x, double &s, double &c) nogil


@cython.boundscheck(False)
cdef _vdt_sin(double[::1] x, double[::1] s):

    cdef int n = x.shape[0]
    cdef int i
    for i in prange(n, nogil=True, num_threads=4):
        s[i] = fast_sin(x[i])

@cython.boundscheck(False)
cdef _vdt_cos(double[::1] x, double[::1] c):

    cdef int n = x.shape[0]
    cdef int i
    for i in prange(n, nogil=True, num_threads=4):
        c[i] = fast_cos(x[i])

@cython.boundscheck(False)
cdef _vdt_sincos(double[::1] x, double[::1] s, double[::1] c):

    cdef int n = x.shape[0]
    cdef int i
    for i in prange(n, nogil=True, num_threads=4):
        fast_sincos(x[i], s[i], c[i])


def vdt_sin(double[::1] x):

    cdef int n = x.shape[0]
    _vdt_sin(x, s)
    return s

def vdt_cos(double[::1] x):

    cdef int n = x.shape[0]
    _vdt_cos(x, c)
    return c

def vdt_sincos(double[::1] x):

    cdef int n = x.shape[0]
    _vdt_sincos(x, s, c)
    return s, c

# @cython.boundscheck(False)
# def vdt_sinv(double[::1] x, double[::1] s):

#     cdef int n = x.shape[0]
#     fast_sinv(n, &x[0], &s[0])
