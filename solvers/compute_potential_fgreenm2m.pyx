import numpy as np
cimport numpy as np

from libc.math cimport log
from cython.parallel import parallel, prange


# from types import MethodType
# Point.incx = MethodType(incx, None, Point)


cpdef compute_potential_fgreenm2m(self, double[:,:] x, double[:,:] y,
                                        double[:,:] phi, double[:,:] rho):

    cdef int k, l, m, n
    cdef int nx = self.nx, ny = self.ny
    cdef double xi, yi, xj, yj, r2
    cdef double da = self.dx * self.dy

    for k in prange(ny, nogil=True, num_threads=2):
        for l in xrange(nx):
            xi, yi = x[k,l], y[k,l]
            for m in xrange(ny):
                for n in xrange(nx):
                    xj, yj = x[m,n], y[m,n]
                    r2 = (xi - xj) ** 2 + (yi - yj) ** 2 + 1e-8
                    phi[k,l] += (-rho[m,n] * da * 1 / 2. * log(r2))

    # for i in xrange(n_macroparticles):
    #     xi, yi = x[i], y[i]
    #     for j in xrange(n_macroparticles):
    #         xj, yj = x[j], y[j]
    #         r2 = (xi - xj) ** 2 + (yi - yj) ** 2 + 1e-6
    #         phi[i] += (-rho[j] * da * 1 / 2 * np.log(r2)) # * self.dx * self.dy / (2 * np.pi))

    # self.phi = np.reshape(phi, (ny, nx))
