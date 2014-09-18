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
cpdef rfq_detune(double[::1] dphi_x, double[::1] dphi_y,
                 double dapp_xz, double dapp_yz, double omega_RF, double phi_0_RF,
                 double[::1] z, double[::1] dp, double p0, double beta):

    cdef double c = 299792458.
    cdef double omegaRF_beta_c = omega_RF / (beta * c)
    cdef double cos_dependence

    cdef unsigned int n = dphi_x.shape[0]
    cdef unsigned int i
#    for i in prange(n, nogil=True):
    for i in xrange(n):
        cos_dependence = cos(z[i] * omegaRF_beta_c + phi_0_RF) / ((1. + dp[i]) * p0)
        dphi_x[i] += dapp_xz * cos_dependence
        dphi_y[i] += dapp_yz * cos_dependence


@cython.boundscheck(False)
cpdef rfq_transverse_kicks(double[::1] x, double[::1] xp, double[::1] y, double[::1] yp,
                           double[::1] z, double p0, double beta,
                           double omega_RF, double v2_RF, double phi_0_RF):

    cdef double c = 299792458.
    cdef double e = 1.60217657e-19
    cdef double omegaRF_beta_c = omega_RF / (beta * c)
    cdef double e_2_v2RF_omegaRF = 2. * e * v2_RF / omega_RF
    
    cdef double cos_dependence
    cdef double delta_p_x, delta_p_y
    
    cdef unsigned int n = xp.shape[0]
    cdef unsigned int i
#    for i in prange(n, nogil=True):
    for i in xrange(n):
        cos_dependence = cos(omegaRF_beta_c * z[i] + phi_0_RF)
        
        delta_p_x = x[i] * e_2_v2RF_omegaRF * cos_dependence
        delta_p_y = -y[i] * e_2_v2RF_omegaRF * cos_dependence

        xp[i] += delta_p_x / p0
        yp[i] += delta_p_y / p0


@cython.boundscheck(False)
cpdef rfq_longitudinal_kick(double[::1] x, double[::1] y, double[::1] z, double[::1] dp,
                            double beta, double p0, double omega_RF, double phi_0_RF,
                            double v2_RF):

    cdef double c = 299792458.
    cdef double e = 1.60217657e-19
    cdef double omegaRF_beta_c = omega_RF / (beta * c)
    cdef double e_v2RF_beta_c = e * v2_RF / (beta * c)
    cdef double delta_p

    cdef unsigned int n = dp.shape[0]
    cdef unsigned int i
#    for i in prange(n, nogil=True):
    for i in xrange(n):
        delta_p = - e_v2RF_beta_c * (x[i] ** 2 - y[i] ** 2) * sin(omegaRF_beta_c * z[i] + phi_0_RF)
        dp[i] += delta_p / p0
