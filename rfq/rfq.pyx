"""
This module contains the Cython implementation of a pillbox-cavity RF
quadrupole - referred to as the RFQ - as it was proposed by Alexej
Grudiev in 'Radio frequency quadrupole for Landau damping in
accelerators', Phys. Rev. Special Topics - Accelerators and Beams 17,
011001 (2014) [1]. Similar to a 'Landau' octupole magnet, this device
is intended to introduce an incoherent tune spread such that Landau
damping can prevent the growth of transverse collective instabilities.

The formulae that are used are based on [1] and make use of the thin-
lens approximation. On the one hand, the RFQ introduces a longitudinal
spread of the betatron frequency and on the other hand, a transverse
spread of the synchrotron frequency.

The effect in the transverse plane is modelled in two different
ways

(I)  RFQ as a detuner acting directly on each particles' betatron
     tunes,
(II) RFQ as a localized kick acting on each particles' momenta xp
     and yp.

The effect in the longitudinal plane is always modelled as a localized
kick, i.e. a change in a particle's normalized momentum dp. For model
(II), the incoherent betatron detuning is not applied directly, but is
a consequence of the change in momenta xp and yp.

@author Michael Schenk
@date July, 10th 2014
@brief Cython implementation of a pillbox cavity RF quadrupole for
       Landau damping.
@copyright CERN
"""
from __future__ import division
from cython.parallel import prange
cimport cython.boundscheck
cimport cython.cdivision

from libc.math cimport cos, sin
from scipy.constants import c, e
import numpy as np
cimport numpy as np

from trackers.detuners import DetunerCollection


class RFQTransverseDetuner(DetunerCollection):
    """
    Collection class to contain/manage the segment-wise defined
    RFQ elements RFQTransverseDetunerSegment acting on the
    betatron tunes (detuner model of the RFQ). This is a pure
    Python class and it derives from the DetunerCollection class
    defined in the module PyHEADTAIL.trackers.detuners.
    """
    def __init__(self, v_2, omega, phi_0, beta_x_RFQ, beta_y_RFQ,
                 n_threads=1):
        """
        An RFQ element is fully characterized by the parameters
          v_2:   quadrupolar expansion coefficient of the accelerating
                 voltage (~strength of the RFQ), in [V/m^2].
          omega: Angular frequency of the RF wave, in [rad/s].
          phi_0: Constant phase offset wrt. bunch center (z=0), in
                 [rad].

        beta_x_RFQ and beta_y_RFQ are the beta functions at the
        position of the RFQ, although in the detuner model of the RFQ,
        the RFQ should not actually be understood as being localized.

        n_threads defines the number of threads to be used to execute
        the for-loop on the number of particles with cython OpenMP. It
        is set to 1 by default.
        """
        self.v_2 = v_2
        self.omega = omega
        self.phi_0 = phi_0

        self.beta_x_RFQ = beta_x_RFQ
        self.beta_y_RFQ = beta_y_RFQ

        self.n_threads = n_threads

    def generate_segment_detuner(self, segment_length, **kwargs):
        dapp_xz = self.beta_x_RFQ * self.v_2 * e / (2.*np.pi*self.omega)
        dapp_yz = -self.beta_y_RFQ * self.v_2 * e / (2.*np.pi*self.omega)
        dapp_xz *= segment_length
        dapp_yz *= segment_length

        detuner = RFQTransverseDetunerSegment(
            dapp_xz, dapp_yz, self.omega, self.phi_0, n_threads=self.n_threads)
        self.segment_detuners.append(detuner)


cdef class RFQTransverseDetunerSegment(object):
    """
    Cython implementation of the RFQ element acting directly on the
    particles' betatron tunes (i.e. RFQ detuner model).
    """
    cdef double dapp_xz, dapp_yz, omega, phi_0
    cdef int n_threads

    def __init__(self, dapp_xz, dapp_yz, omega, phi_0, n_threads):
        """
          omega:   Angular frequency of the RF wave, in [rad/s].
          phi_0:   Constant phase offset wrt. bunch center (z=0), in
                   [rad].
          dapp_xz: Strength of detuning in the horizontal plane, scaled
                   to the relative segment length.
          dapp_yz: Strength of detuning in the vertical plane, scaled
                   to the relative segment length.

        n_threads defines the number of threads to be used to execute
        the for-loop on the number of particles with cython OpenMP. It
        is set to 1 by default.
        """
        self.dapp_xz = dapp_xz
        self.dapp_yz = dapp_yz

        self.omega = omega
        self.phi_0 = phi_0

        self.n_threads = n_threads

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def detune(self, beam):
        """
        Calculates for each particle its betatron detuning dQ_x, dQ_y
        according to formulae taken from [1] (see above).
            dQ_x = dapp_xz / p * \cos(omega / (beta c) z + phi_0)
            dQ_y = dapp_yz / p * \cos(omega / (beta c) z + phi_0)
        with
            dapp_xz = beta_x_RFQ  * v_2 * e / (2 Pi * omega)
            dapp_yz = -beta_y_RFQ  * v_2 * e / (2 Pi * omega)
        and p the particle momentum p = (1 + dp) p0.
        (Probably, it would make sense to approximate p by p0 for better
        performance).
        """
        cdef double[::1] z = beam.z
        cdef double[::1] dp = beam.dp
        cdef double p0 = beam.p0

        cdef unsigned int n_particles = z.shape[0]
        cdef double[::1] dQ_x = np.zeros(n_particles, dtype=np.double)
        cdef double[::1] dQ_y = np.zeros(n_particles, dtype=np.double)

        cdef double cos_arg = self.omega / (beam.beta * c)
        cdef double cos_term, p

        cdef unsigned int i
        for i in prange(n_particles, nogil=True, num_threads=self.n_threads):
            cos_term = cos(cos_arg * z[i] + self.phi_0)
            p = (1. + dp[i]) * p0
            cos_term = cos_term / p

            dQ_x[i] = self.dapp_xz * cos_term
            dQ_y[i] = self.dapp_yz * cos_term

        return dQ_x, dQ_y


cdef class RFQKick(object):
    """
    Cython base class to describe the RFQ element in the localized kick
    model for both the transverse and the longitudinal coordinates.
    """
    cdef double v_2, omega, phi_0
    cdef int n_threads

    def __init__(self, v_2, omega, phi_0, n_threads=1):
        """
        An RFQ element is fully characterized by the parameters
          v_2:   quadrupolar expansion coefficient of the accelerating
                 voltage (~strength of the RFQ), in [V/m^2].
          omega: Angular frequency of the RF wave, in [rad/s].
          phi_0: Constant phase offset wrt. bunch center (z=0), in
                 [rad].

        n_threads defines the number of threads to be used to execute
        the for-loop on the number of particles with cython OpenMP. It
        is set to 1 by default.
        """
        self.v_2 = v_2
        self.omega = omega
        self.phi_0 = phi_0

        self.n_threads = n_threads

    def track(self, beam):
        pass


cdef class RFQTransverseKick(RFQKick):
    """
    Cython implementation of the RFQ element acting on the particles'
    transverse coordinates (i.e. localized kick model).
    """

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def track(self, beam):
        """
        The formula that describes the transverse kick experienced by
        an ultra-relativistic particle traversing the RFQ longitudinally
        is based on the thin-lens approximation
            \Delta p_x = -x*(2 e v_2 / omega) *
                cos(omega z / (beta c) + phi_0),
            \Delta p_y =  y*(2 e v_2 / omega) *
                cos(omega z / (beta c) + phi_0).

        The for loop on the number of particles can make use of cython
        OpenMP with the number of threads defined by self.n_threads. It
        is set to 1 by default.
        """
        cdef double[::1] x = beam.x
        cdef double[::1] y = beam.y
        cdef double[::1] z = beam.z
        cdef double[::1] xp = beam.xp
        cdef double[::1] yp = beam.yp
        cdef double p0 = beam.p0

        cdef double cos_arg = self.omega / (beam.beta * c)
        cdef double pre_factor = 2. * e * self.v_2 / self.omega
        cdef double delta_p_x, delta_p_y, cos_term

        cdef unsigned int i
        cdef unsigned int n_particles = z.shape[0]

        for i in prange(n_particles, nogil=True, num_threads=self.n_threads):
            cos_term = pre_factor * cos(cos_arg * z[i] + self.phi_0)

            delta_p_x = -x[i] * cos_term
            delta_p_y = y[i] * cos_term

            xp[i] += delta_p_x / p0
            yp[i] += delta_p_y / p0


cdef class RFQLongitudinalKick(RFQKick):
    """
    Cython implementation of the RFQ element acting on the particles'
    longitudinal coordinate dp.
    """

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def track(self, beam):
        """
        The formula used to describe the longitudinal kick is given by
            \Delta p_z = -(x^2 - y^2) (e v_2 / (beta c)) *
                sin(omega z / (beta c) + phi_0).

        The for loop on the number of particles can make use of cython
        OpenMP with the number of threads defined by self.n_threads. It
        is set to 1 by default.
        """
        cdef double[::1] x = beam.x
        cdef double[::1] y = beam.y
        cdef double[::1] z = beam.z
        cdef double[::1] dp = beam.dp
        cdef double p0 = beam.p0

        cdef double sin_arg = self.omega / (beam.beta * c)
        cdef double pre_factor = e * self.v_2 / (beam.beta * c)
        cdef double delta_p, sin_term

        cdef unsigned int i
        cdef unsigned int n_particles = z.shape[0]

        for i in prange(n_particles, nogil=True, num_threads=self.n_threads):
            sin_term = pre_factor * sin(sin_arg * z[i] + self.phi_0)
            delta_p = -(x[i]*x[i] - y[i]*y[i]) * sin_term

            dp[i] += delta_p / p0
