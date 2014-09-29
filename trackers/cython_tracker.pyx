'''
Created on 20.08.2014

@author: Kevin Li, Michael Schenk
'''
from cython.parallel import prange
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, M_PI


@cython.boundscheck(False)
cpdef track_transverse_without_detuners(double[:,::1] I, double[:,::1] J, double dphi_x, double dphi_y,
                                        double[::1] x, double[::1] xp, double[::1] y, double[::1] yp):

    # Calculate transport matrix M.
    cdef double dphix_2pi = 2. * M_PI * dphi_x
    cdef double dphiy_2pi = 2. * M_PI * dphi_y

    cdef double cos_dphi_x = cos(dphix_2pi)
    cdef double sin_dphi_x = sin(dphix_2pi)
    cdef double cos_dphi_y = cos(dphiy_2pi)
    cdef double sin_dphi_y = sin(dphiy_2pi)

    cdef double M00 = I[0,0] * cos_dphi_x + J[0,0] * sin_dphi_x
    cdef double M01 = I[0,1] * cos_dphi_x + J[0,1] * sin_dphi_x
    cdef double M10 = I[1,0] * cos_dphi_x + J[1,0] * sin_dphi_x
    cdef double M11 = I[1,1] * cos_dphi_x + J[1,1] * sin_dphi_x
    cdef double M22 = I[2,2] * cos_dphi_y + J[2,2] * sin_dphi_y
    cdef double M23 = I[2,3] * cos_dphi_y + J[2,3] * sin_dphi_y
    cdef double M32 = I[3,2] * cos_dphi_y + J[3,2] * sin_dphi_y
    cdef double M33 = I[3,3] * cos_dphi_y + J[3,3] * sin_dphi_y

    # Track particles.
    cdef double xtmp, xptmp, ytmp, yptmp
    cdef int n = x.shape[0]
    cdef int i
    for i in prange(n, nogil=True):

        xtmp  = x[i]
        xptmp = xp[i]
        x[i]  = M00 * xtmp + M01 * xptmp
        xp[i] = M10 * xtmp + M11 * xptmp

        ytmp  = y[i]
        yptmp = yp[i]
        y[i]  = M22 * ytmp + M23 * yptmp
        yp[i] = M32 * ytmp + M33 * yptmp


@cython.boundscheck(False)
cpdef track_transverse_with_detuners(double[:,::1] I, double[:,::1] J, double[::1] dphi_x, double[::1] dphi_y,
                                     double[::1] x, double[::1] xp, double[::1] y, double[::1] yp):

    cdef double cos_dphi_x, sin_dphi_x, cos_dphi_y, sin_dphi_y
    cdef double dphix_2pi, dphiy_2pi
    cdef double xtmp, xptmp, ytmp, yptmp

    cdef double I00 = I[0,0]
    cdef double I01 = I[0,1]
    cdef double I10 = I[1,0]
    cdef double I11 = I[1,1]
    cdef double I22 = I[2,2]
    cdef double I23 = I[2,3]
    cdef double I32 = I[3,2]
    cdef double I33 = I[3,3]

    cdef double J00 = J[0,0]
    cdef double J01 = J[0,1]
    cdef double J10 = J[1,0]
    cdef double J11 = J[1,1]
    cdef double J22 = J[2,2]
    cdef double J23 = J[2,3]
    cdef double J32 = J[3,2]
    cdef double J33 = J[3,3]

    cdef int n = dphi_x.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        dphix_2pi = 2. * M_PI * dphi_x[i]
        dphiy_2pi = 2. * M_PI * dphi_y[i]

        cos_dphi_x = cos(dphix_2pi)
        sin_dphi_x = sin(dphix_2pi)
        cos_dphi_y = cos(dphiy_2pi)
        sin_dphi_y = sin(dphiy_2pi)

        xtmp  = x[i]
        xptmp = xp[i]
        x[i]  = (I00 * cos_dphi_x + J00 * sin_dphi_x) * xtmp + \
                (I01 * cos_dphi_x + J01 * sin_dphi_x) * xptmp
        xp[i] = (I10 * cos_dphi_x + J10 * sin_dphi_x) * xtmp + \
                (I11 * cos_dphi_x + J11 * sin_dphi_x) * xptmp

        ytmp  = y[i]
        yptmp = yp[i]
        y[i]  = (I22 * cos_dphi_y + J22 * sin_dphi_y) * ytmp + \
                (I23 * cos_dphi_y + J23 * sin_dphi_y) * yptmp
        yp[i] = (I32 * cos_dphi_y + J32 * sin_dphi_y) * ytmp + \
                (I33 * cos_dphi_y + J33 * sin_dphi_y) * yptmp


@cython.boundscheck(False)
cpdef chromaticity_detune(double[::1] dphi_x, double[::1] dphi_y, \
                          double dQp_x, double dQp_y, double[::1] dp):

    cdef int n = dphi_x.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        # W/o factor 2 np.pi. See track method.
        dphi_x[i] += dQp_x * dp[i]
        dphi_y[i] += dQp_y * dp[i]


@cython.boundscheck(False)
cpdef amplitude_detune(double[::1] dphi_x, double[::1] dphi_y, \
                       double dapp_x, double dapp_y, double dapp_xy, \
                       double[::1] x, double[::1] xp, double beta_x, \
                       double[::1] y, double[::1] yp, double beta_y):

    cdef double betax2 = beta_x * beta_x
    cdef double betay2 = beta_y * beta_y
    cdef double jx
    cdef double jy

    cdef int n = dphi_x.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        jx = ( x[i] * x[i] + xp[i] * xp[i] * betax2 ) / ( 2. * beta_x)
        jy = ( y[i] * y[i] + yp[i] * yp[i] * betay2 ) / ( 2. * beta_y)

        # W/o factor 2 np.pi. See track method.
        dphi_x[i] += dapp_x * jx + dapp_xy * jy
        dphi_y[i] += dapp_y * jy + dapp_xy * jx
