import numpy as np

cimport numpy as np
cimport cython.boundscheck
cimport cython.wraparound
cimport cython.nonecheck
cimport cython.cdivision


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef interpolate(double[::1] x_samples, double[::1] y_samples,
                 double[::1] x, double bias=0):

    cdef int i, ix
    cdef double fx, vx
    cdef int ni = x.shape[0]
    cdef double[::1] yi = np.zeros(ni)
    cdef double dx = x_samples[1] - x_samples[0]

    for i in range(ni):
        vx = (x[i] - bias)/dx
        ix = int(vx)
        fx = vx - ix

        # yi[i] = y_samples[ix] + (fx)*dx
        yi[i] = y_samples[ix] * (1-fx) + y_samples[ix+1] * (fx)  # = fx((i)) * (1-hx)+ fx((i+1)) * hx;

    return yi


def interpolated_mod2pi(fn, xmin, xmax, steps):
    x_map = np.linspace(xmin, xmax, steps)
    fn_map = fn(x_map)
    bias_x = np.min(x_map)

    def function(x):

        return np.reshape(interpolate(
            x_map, fn_map, np.mod(x, 2 * np.pi), bias_x), x.shape)

    return function


# sin_interpolated = interpolated_mod2pi(np.sin, -2.1 * np.pi, 2.1 * np.pi, 2000000)
# cos_interpolated = interpolated_mod2pi(np.cos, -2.1 * np.pi, 2.1 * np.pi, 2000000)
