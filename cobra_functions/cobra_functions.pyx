'''
Created on 31.01.2014

@author: Kevin Li
'''


# import numpy as np
cimport numpy as np


cpdef double mean(double[::1] u):

    cdef int n = u.shape[0]
    cdef double mean_u = 0

    for i in xrange(n):
        mean_u += u[i]
    if n:
        mean_u *= 1. / n

    return mean_u

cpdef double std(double[::1] u):

    cdef int n = u.shape[0]
    cdef double mean_u = mean(u)
    cdef double std_u = 0

    for i in xrange(n):
        std_u += (u[i] - mean_u) * (u[i] - mean_u)
    if n:
        std_u *= 1. / n

    std_u = np.sqrt(std_u)

    return std_u

cpdef double emittance(double[::1] u, double[::1] v):

    cdef int i, n = u.shape[0]
    cdef double mean_u = mean(u)
    cdef double mean_v = mean(v)

    cdef double u2 = 0
    cdef double v2 = 0
    cdef double uv = 0

    for i in xrange(n):
        u2 += (u[i] - mean_u) * (u[i] - mean_u)
        v2 += (v[i] - mean_v) * (v[i] - mean_v)
        uv += (u[i] - mean_u) * (v[i] - mean_v)
    if n:
        u2 *= 1. / n
        v2 *= 1. / n
        uv *= 1. / n

    return np.sqrt(u2 * v2 - uv * uv)
