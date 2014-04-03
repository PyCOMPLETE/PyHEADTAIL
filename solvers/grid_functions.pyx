

import numpy as np
cimport numpy as np


from libc.math cimport floor
from cython.parallel import parallel, prange


# Try using numpy arrays
# cdef fastgather(self, np.ndarray[double, ndim=1] x,
#                      np.ndarray[double, ndim=1] y,
#                      np.ndarray[double, ndim=2] rho):
def fastgather(self, double[:] x, double[:] y):

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
    cdef double[:,:] rho = self.rho
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

    # Line charge density and particle selection
    # double lambda;
    # std::vector<int> index;
    # t.get_slice(i_slice, lambda, index);
    # int np = index.size();

    cdef double fx, fy
    cdef double a1, a2, a3, a4
    cdef int ix, iy
    cdef int i, n = len(x)
    cdef int l=1
    # for i in prange(n, nogil=True, num_threads=2):
    for i in xrange(n):
        fx, fy = (x[i] - x0) * dxi, (y[i] - y0) * dyi
        ix, iy = <int> fx, <int> fy
        fx, fy = fx - ix, fy - iy

        a1 = (1 - fx) * (1 - fy)
        a2 = fx * (1 - fy)
        a3 = (1 - fx) * fy
        a4 = fx * fy

        rho[iy, ix] += l * a1 * ai
        rho[iy + 1, ix] += l * a2 * ai
        rho[iy, ix + 1] += l * a3 * ai
        rho[iy + 1, ix + 1] += l * a4 * ai

        # rho[ix, iy] += l * a1 * ai
        # rho[ix + 1, iy] += l * a2 * ai
        # rho[ix, iy + 1] += l * a3 * ai
        # rho[ix + 1, iy + 1] += l * a4 * ai

    # H, xedges, yedges = np.histogram2d(ix, iy, bins=self.rho.shape)
    # self.rho += H

# template<typename T, typename U>
# void PoissonBase::fastscatter(T t, U u, int i_slice)
# {
#     int np;
#     double qp;
#     std::vector<int> ip;

#     /*
#      * Cell
#      *
#      *  3 ------------ 4
#      *  |     |        |
#      *  |     |        |
#      *  |     |        |
#      *  |     |        |
#      *  |-----x--------|
#      *  |     |        |
#      *  1 ------------ 2
#      */

#     // On regular mesh
#     const double dx = (mx.back() - mx.front()) / (n_points_x - 1);
#     const double dy = (my.back() - my.front()) / (n_points_y - 1);
#     const double dxi = 1 / dx;
#     const double dyi = 1 / dy;
#     // TODO: on adaptive mesh

#     // Line charge density and particle selection
#     t.get_slice(i_slice, np, qp, ip);

#     // t impacting fields
#     for (int j=0; j<np; j++)
#     {
#       double xp = t.x[ip[j]];
#       double yp = t.y[ip[j]];

#       // Compute points
#       double fx = (xp - mx[0]) * dxi;
#       double fy = (yp - my[0]) * dyi;
#       int ix = std::floor(fx);
#       int iy = std::floor(fy);

#       size_t k1 = iy * n_points_x + ix;
#       size_t k2 = iy * n_points_x + ix + 1;
#       size_t k3 = (iy + 1) * n_points_x + ix;
#       size_t k4 = (iy + 1) * n_points_x + ix + 1;

#       // Compute normalized area
#       fx -= ix;
#       fy -= iy;

#       double a1 = (1 - fx) * (1 - fy);
#       double a2 = fx * (1 - fy);
#       double a3 = (1 - fx) * fy;
#       double a4 = fx * fy;

#       // Scatter fields
#       t.kx[ip[j]] = (u.ex_g[k1] * a1 + u.ex_g[k2] * a2
#                      + u.ex_g[k3] * a3 + u.ex_g[k4] * a4);
#       t.ky[ip[j]] = (u.ey_g[k1] * a1 + u.ey_g[k2] * a2
#                      + u.ey_g[k3] * a3 + u.ey_g[k4] * a4);
#     }

#     // Line charge density and particle selection
#     u.get_slice(i_slice, np, qp, ip);

#     // u impacting fields
#     for (int j=0; j<np; j++)
#     {
#       double xp = u.x[ip[j]];
#       double yp = u.y[ip[j]];

#       // Compute points
#       double fx = (xp - mx[0]) * dxi;
#       double fy = (yp - my[0]) * dyi;
#       int ix = std::floor(fx);
#       int iy = std::floor(fy);

#       size_t k1 = iy * n_points_x + ix;
#       size_t k2 = iy * n_points_x + ix + 1;
#       size_t k3 = (iy + 1) * n_points_x + ix;
#       size_t k4 = (iy + 1) * n_points_x + ix + 1;

#       // Compute normalized area
#       fx -= ix;
#       fy -= iy;

#       double a1 = (1 - fx) * (1 - fy);
#       double a2 = fx * (1 - fy);
#       double a3 = (1 - fx) * fy;
#       double a4 = fx * fy;

#       // Scatter fields
#       u.kx[ip[j]] = (t.ex_g[k1] * a1 + t.ex_g[k2] * a2
#                      + t.ex_g[k3] * a3 + t.ex_g[k4] * a4);
#       u.ky[ip[j]] = (t.ey_g[k1] * a1 + t.ey_g[k2] * a2
#                      + t.ey_g[k3] * a3 + t.ey_g[k4] * a4);
#     }
# }

# template<typename T, typename U>
# void PoissonBase::parallelscatter(T t, U u, int i_slice)
# {
#     /*
#      * Cell
#      *
#      *  3 ------------ 4
#      *  |     |        |
#      *  |     |        |
#      *  |     |        |
#      *  |     |        |
#      *  |-----x--------|
#      *  |     |        |
#      *  1 ------------ 2
#      */

#     // On regular mesh
#     const double dx = (mx.back() - mx.front()) / (n_points_x - 1);
#     const double dy = (my.back() - my.front()) / (n_points_y - 1);
#     const double dxi = 1 / dx;
#     const double dyi = 1 / dy;
#     const double mx0 = mx[0];
#     const double my0 = my[0];
#     // TODO: on adaptive mesh

# //    // Separate variable for parallel processing
# //	// Line charge density and particle selection
# //	t.get_slice(i_slice, np, qp, ip);
# //
# //    std::vector<double> t_x(np);
# //    std::vector<double> t_y(np);
# //    std::vector<double> t_kx(np);
# //    std::vector<double> t_ky(np);
# //    std::vector<double> u_ex_g(n_points);
# //    std::vector<double> u_ey_g(n_points);
# //
# //    for (int j=0; j<np; j++)
# //    {
# //        t_x[j] = t.x[ip[j]];
# //        t_y[j] = t.y[ip[j]];
# ////      t_kx[j] = t.kx[ip[j]];
# ////      t_ky[j] = t.ky[ip[j]];
# //    }
# //
# //    for (size_t k=0; k<n_points; k++)
# //    {
# //        u_ex_g[k] = u.ex_g[k];
# //        u_ey_g[k] = u.ey_g[k];
# //    }
# //
# //    // Line charge density and particle selection
# //    u.get_slice(i_slice, np, qp, ip);
# //
# //    std::vector<double> u_x(np);
# //    std::vector<double> u_y(np);
# //    std::vector<double> u_kx(np);
# //    std::vector<double> u_ky(np);
# //    std::vector<double> t_ex_g(n_points);
# //    std::vector<double> t_ey_g(n_points);
# //
# //    for (int j=0; j<np; j++)
# //    {
# //        u_x[j] = u.x[ip[j]];
# //        u_y[j] = u.y[ip[j]];
# ////      u_kx[j] = u.kx[ip[j]];
# ////      u_ky[j] = u.ky[ip[j]];
# //    }
# //
# //    for (size_t k=0; k<n_points; k++)
# //    {
# //        t_ex_g[k] = t.ex_g[k];
# //        t_ey_g[k] = t.ey_g[k];
# //    }

# //#pragma omp parallel
# //{
# //#pragma omp sections nowait private(np, qp, ip)
# //{
# //#pragma omp section
# //{
#     int np;
#     double lambda;
#     std::vector<int> index;

#   // Line charge density and particle selection
#   t.get_slice(i_slice, lambda, index);
#   np = index.size();

#     // t impacting fields
#     for (int j=0; j<np; j++)
#     {
#         double xp = t.x[index[j]];
#         double yp = t.y[index[j]];

#         // Compute points
#         double fx = (xp - mx[0]) * dxi;
#         double fy = (yp - my[0]) * dyi;
#         int ix = std::floor(fx);
#         int iy = std::floor(fy);

#         size_t k1 = iy * n_points_x + ix;
#         size_t k2 = iy * n_points_x + ix + 1;
#         size_t k3 = (iy + 1) * n_points_x + ix;
#         size_t k4 = (iy + 1) * n_points_x + ix + 1;

#         // Compute normalized area
#         fx -= ix;
#         fy -= iy;

#         double a1 = (1 - fx) * (1 - fy);
#         double a2 = fx * (1 - fy);
#         double a3 = (1 - fx) * fy;
#         double a4 = fx * fy;

#         // Scatter fields
#         t.kx[index[j]] = (u.ex_g[k1] * a1 + u.ex_g[k2] * a2
#                      + u.ex_g[k3] * a3 + u.ex_g[k4] * a4);
#         t.ky[index[j]] = (u.ey_g[k1] * a1 + u.ey_g[k2] * a2
#                      + u.ey_g[k3] * a3 + u.ey_g[k4] * a4);
#     }
# //}
# //
# //#pragma omp section
# //{
#   // Line charge density and particle selection
#     u.get_slice(i_slice, lambda, index);
#     np = index.size();

#     // u impacting fields
#     for (int j=0; j<np; j++)
#     {
#         double xp = u.x[index[j]];
#         double yp = u.y[index[j]];

#         // Compute points
#         double fx = (xp - mx[0]) * dxi;
#         double fy = (yp - my[0]) * dyi;
#         int ix = std::floor(fx);
#         int iy = std::floor(fy);

#         size_t k1 = iy * n_points_x + ix;
#         size_t k2 = iy * n_points_x + ix + 1;
#         size_t k3 = (iy + 1) * n_points_x + ix;
#         size_t k4 = (iy + 1) * n_points_x + ix + 1;

#         // Compute normalized area
#         fx -= ix;
#         fy -= iy;

#         double a1 = (1 - fx) * (1 - fy);
#         double a2 = fx * (1 - fy);
#         double a3 = (1 - fx) * fy;
#         double a4 = fx * fy;

#         // Scatter fields
#         u.kx[index[j]] = (t.ex_g[k1] * a1 + t.ex_g[k2] * a2
#                      + t.ex_g[k3] * a3 + t.ex_g[k4] * a4);
#         u.ky[index[j]] = (t.ey_g[k1] * a1 + t.ey_g[k2] * a2
#                      + t.ey_g[k3] * a3 + t.ey_g[k4] * a4);
#     }
# //}
# //}
# //}
# }
