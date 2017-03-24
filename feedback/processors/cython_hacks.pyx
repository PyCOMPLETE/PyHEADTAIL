#TODO: maybe this could be simplified/avoided by using cimport scipy.linalg.cython_blas

import numpy as np
cimport numpy as np
cimport cython

""" The functions in this file have been written, because the dot product function of NumPy slowed down PyHEADTAIL
    simulation in the CERN batch system by a factor of two or more. The only working solution which was found was to
    write a new function for matrix product in Cython.
"""

@cython.boundscheck(False)
@cython.wraparound(False)

def cython_matrix_product(double[:, ::1] matrix not None, double[::1] vector not None):

    cdef np.intp_t i, j, dim_0, dim_1
    dim_0 = matrix.shape[0]
    dim_1 = matrix.shape[1]
    cdef double[::1] D = np.zeros(dim_0)
    
    for i in range(dim_0):
        for j in range(dim_1):
            D[i] += matrix[i,j]* vector[j]
            
    return D