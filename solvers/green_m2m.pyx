import numpy as np
cimport numpy as np


# from types import MethodType
# Point.incx = MethodType(incx, None, Point)


cpdef green_m2m(self):

    cdef int i, j
    cdef int n_points = self.n_points
    cdef int nx = self.nx, ny = self.ny
    cdef double xi, yi, xj, yj, r2
    cdef double[::1] x, y, phi, rho

    self.phi = np.zeros((ny, nx))

    x, y = self.x.flatten(), self.y.flatten()
    phi, rho = self.phi.flatten(), self.rho.flatten()

    for i in xrange(n_points):
        xi, yi = x[i], y[i]
        for j in xrange(n_points):
            xj, yj = x[j], y[j]
            r2 = (xi - xj) ** 2 + (yi - yj) ** 2 + 1e-6
            phi[i] += (-rho[j] * 1 / 2 * np.log(r2)) # * self.dx * self.dy / (2 * np.pi))

    self.phi = np.reshape(phi, (ny, nx))