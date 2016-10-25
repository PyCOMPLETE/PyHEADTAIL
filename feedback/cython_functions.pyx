import numpy as np
cimport numpy as np
cimport cython

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