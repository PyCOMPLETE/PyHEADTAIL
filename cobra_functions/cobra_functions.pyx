'''
Created on 31.01.2014

@author: kli
'''


import numpy as np
cimport numpy as np


cpdef double emittance(double[::1] u, double mean_u,
                       double[::1] v, double mean_v):

	cdef int i, k
	cdef int n = len(u)
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
