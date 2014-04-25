

import numpy as np
cimport numpy as np


from libc.math cimport floor
from cython.parallel import parallel, prange


# Try using numpy arrays
# cdef fastgather(self, np.ndarray[double, ndim=1] x,
#                      np.ndarray[double, ndim=1] y,
#                      np.ndarray[double, ndim=2] rho):
def gather_from(self, double[:] x, double[:] y, double[:,:] rho):
    '''
    Cell
    3 ------------ 4
    |     |        |
    |     |        |
    |     |        |
    |     |        |
    |-----x--------|
    |     |        |
    1 ------------ 2
    '''

    # Line charge density
    # lambda_e = self.density / self.n_macroparticles * (max_x - min_x) * (max_y - min_y)
    # lambda_p = bunch.n_particles / bunch.n_macroparticles / dz;

    # Initialise
    rho[:] = 0

    # On regular mesh
    cdef double x0 = self.x[0,0]
    cdef double x1 = self.x[0,1]
    cdef double dx = x1 - x0
    cdef double y0 = self.y[0,0]
    cdef double y1 = self.y[1,0]
    cdef double dy = y1 - y0
    cdef double dxi = 1 / dx
    cdef double dyi = 1 / dy
    cdef double ai = 1 / (dx * dy)
    # TODO: on adaptive mesh

    cdef int i, n = len(x)
    cdef int ix, iy
    cdef double fx, fy
    cdef double a1, a2, a3, a4
    cdef double q = 1 # if taken into account as coefficient for kx in push
    # for i in prange(n, nogil=True, num_threads=2):
    for i in xrange(n):
        fx, fy = (x[i] - x0) * dxi, (y[i] - y0) * dyi
        ix, iy = <int> fx, <int> fy
        fx, fy = fx - ix, fy - iy

        a1 = (1 - fx) * (1 - fy)
        a2 = fx * (1 - fy)
        a3 = (1 - fx) * fy
        a4 = fx * fy

        rho[iy, ix] += a1 * ai * q
        rho[iy + 1, ix] += a2 * ai * q
        rho[iy, ix + 1] += a3 * ai * q
        rho[iy + 1, ix + 1] += a4 * ai * q
        # remember: 1 / dz missing here

    # H, xedges, yedges = np.histogram2d(ix, iy, bins=self.rho.shape)
    # self.rho += H

# def weight(self):

#     pass

# def fastscatter(self):
#     '''
#     Cell
#     3 ------------ 4
#     |     |        |
#     |     |        |
#     |     |        |
#     |     |        |
#     |-----x--------|
#     |     |        |
#     1 ------------ 2
#     '''

#     # Initialise
#     cdef double[:,:] rho = self.rho
#     rho[:] = 0

#     # On regular mesh
#     cdef double x0 = self.x[0,0]
#     cdef double x1 = self.x[0,1]
#     cdef double dx = x1 - x0
#     cdef double y0 = self.y[0,0]
#     cdef double y1 = self.y[1,0]
#     cdef double dy = y1 - y0
#     cdef double dxi = 1 / dx
#     cdef double dyi = 1 / dy
#     cdef double ai = 1 / (dx * dy)
#     # TODO: on adaptive mesh

#     cdef double fx, fy
#     cdef double a1, a2, a3, a4
#     cdef int ix, iy
#     cdef int i, n = len(x)
#     cdef int lambda_ = 1 # number_of_particles / dz

#     x, y = bunch.x, bunch.y
#     n = bunch.n_macroparticles
#     lambda_p = bunch.n_particles / bunch.n_macroparticles / dz;
#     # for i in prange(n, nogil=True, num_threads=2):
#     for i in xrange(n):
#         fx, fy = (x[i] - x0) * dxi, (y[i] - y0) * dyi
#         ix, iy = <int> fx, <int> fy
#         fx, fy = fx - ix, fy - iy

#         a1 = (1 - fx) * (1 - fy)
#         a2 = fx * (1 - fy)
#         a3 = (1 - fx) * fy
#         a4 = fx * fy

#         # Scatter fields
#         bunch.kx[i] = 1
#         bunch.ky[i] = 1
# #       t.kx[ip[j]] = (u.ex_g[k1] * a1 + u.ex_g[k2] * a2
# #                      + u.ex_g[k3] * a3 + u.ex_g[k4] * a4);
# #       t.ky[ip[j]] = (u.ey_g[k1] * a1 + u.ey_g[k2] * a2
# #                      + u.ey_g[k3] * a3 + u.ey_g[k4] * a4);
# #     }

#         rho[iy, ix] += a1 * ai * lambda_
#         rho[iy + 1, ix] += a2 * ai * lambda_
#         rho[iy, ix + 1] += a3 * ai * lambda_
#         rho[iy + 1, ix + 1] += a4 * ai * lambda_

#     x, y = self.x, self.y
#     n = self.n_macroparticles
#     lambda_e = self.density / self.n_macroparticles * (max_x - min_x) * (max_y - min_y)
#     # for i in prange(n, nogil=True, num_threads=2):
#     for i in xrange(n):
#         fx, fy = (x[i] - x0) * dxi, (y[i] - y0) * dyi
#         ix, iy = <int> fx, <int> fy
#         fx, fy = fx - ix, fy - iy

#         a1 = (1 - fx) * (1 - fy)
#         a2 = fx * (1 - fy)
#         a3 = (1 - fx) * fy
#         a4 = fx * fy

#         # Scatter fields
#         self.kx[i] = 1
#         self.ky[i] = 1
# #       u.kx[ip[j]] = (t.ex_g[k1] * a1 + t.ex_g[k2] * a2
# #                      + t.ex_g[k3] * a3 + t.ex_g[k4] * a4);
# #       u.ky[ip[j]] = (t.ey_g[k1] * a1 + t.ey_g[k2] * a2
# #                      + t.ey_g[k3] * a3 + t.ey_g[k4] * a4);

def scatter_to(self, other):
    '''
    Cell
    3 ------------ 4
    |     |        |
    |     |        |
    |     |        |
    |     |        |
    |-----x--------|
    |     |        |
    1 ------------ 2
    '''

    # Initialise
    cdef double[:,:] ex = self.ex
    cdef double[:,:] ey = self.ey

    # On regular mesh
    cdef double x0 = self.x[0,0]
    cdef double x1 = self.x[0,1]
    cdef double dx = x1 - x0
    cdef double y0 = self.y[0,0]
    cdef double y1 = self.y[1,0]
    cdef double dy = y1 - y0
    cdef double dxi = 1 / dx
    cdef double dyi = 1 / dy
    cdef double ai = 1 / (dx * dy)
    # TODO: on adaptive mesh

    cdef int ix, iy
    cdef double fx, fy
    cdef double a1, a2, a3, a4
    cdef int i, n = other.n_macroparticles
    cdef double[::1] x = other.x
    cdef double[::1] y = other.y
    cdef double[::1] kx = other.kx
    cdef double[::1] ky = other.ky
    # for i in prange(n, nogil=True, num_threads=2):
    for i in xrange(n):
        fx, fy = (x[i] - x0) * dxi, (y[i] - y0) * dyi
        ix, iy = <int> fx, <int> fy
        fx, fy = fx - ix, fy - iy

        a1 = (1 - fx) * (1 - fy)
        a2 = fx * (1 - fy)
        a3 = (1 - fx) * fy
        a4 = fx * fy

    	# size_t k1 = iy * n_points_x + ix;
    	# size_t k2 = iy * n_points_x + ix + 1;
    	# size_t k3 = (iy + 1) * n_points_x + ix;
    	# size_t k4 = (iy + 1) * n_points_x + ix + 1;

    	# // Compute normalized area
    	# fx -= ix;
    	# fy -= iy;

    	# double a1 = (1 - fx) * (1 - fy);
    	# double a2 = fx * (1 - fy);
    	# double a3 = (1 - fx) * fy;
    	# double a4 = fx * fy;

    	# // Scatter fields
    	# t.kx[ip[j]] = (u.ex_g[k1] * a1 + u.ex_g[k2] * a2
        #              + u.ex_g[k3] * a3 + u.ex_g[k4] * a4);
    	# t.ky[ip[j]] = (u.ey_g[k1] * a1 + u.ey_g[k2] * a2
        #              + u.ey_g[k3] * a3 + u.ey_g[k4] * a4);


        # Scatter fields
        kx[i] = ex[iy, ix] * a1  + ex[iy + 1, ix] * a2 * ex[iy, ix + 1] * a3 + ex[iy + 1, ix + 1] * a4
        ky[i] = ey[iy, ix] * a1  + ey[iy + 1, ix] * a2 * ey[iy, ix + 1] * a3 + ey[iy + 1, ix + 1] * a4
#       t.kx[ip[j]] = (u.ex_g[k1] * a1 + u.ex_g[k2] * a2
#                      + u.ex_g[k3] * a3 + u.ex_g[k4] * a4);
#       t.ky[ip[j]] = (u.ey_g[k1] * a1 + u.ey_g[k2] * a2
#                      + u.ey_g[k3] * a3 + u.ey_g[k4] * a4);
