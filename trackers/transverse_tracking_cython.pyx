"""
@author Kevin Li, Michael Schenk
@date 07. January 2014
@brief Description of the transport of transverse phase spaces.
@copyright CERN
"""
from __future__ import division

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin

from .transverse_tracking import TransverseMap


class TransverseSegmentMap(object):
    """ Class to transport/track the particles of the beam in the
    transverse plane through an accelerator ring segment defined by its
    boundaries [s0, s1]. To calculate the transverse linear transport
    matrix M that transports each particle's transverse phase space
    coordinates (x, xp, y, yp) from position s0 to position s1 in the
    accelerator, the TWISS parameters alpha and beta at positions s0
    and s1 must be provided. The betatron phase advance of each
    particle in the present segment is given by their betatron tune
    Q_{x,y} and possibly by an incoherent tune shift introduced e.g. by
    amplitude detuning or chromaticity effects (see trackers.detuners
    module).

    For this implementation of the class, which uses Cython functions,
    two different cases must be considered.

    (1) There are no detuners present in the segment.
        The transport matrix M is a constant and the same for all the
        particles. Hence it is calculated only once. Also, the
        self.track(self, beam) method is bound to
        self.track_without_detuners(self, beam) which calls the
        corresponding Cython function.
    (2) Detuners are present in the segment.
        The transport matrix M is not constant and must be calculated
        for each particle individually. The self.track(self, beam)
        method is bound to self.track_with_detuners(self, beam) which
        calls the corresponding Cython function.

    TODO
    Implement dispersion effects, i.e. the change of a particle's
    transverse phase space coordinates on its relative momentum offset.
    For the moment, a NotImplementedError is raised if dispersion
    coefficients are non-zero.
    Improve the interface between the TransverseSegmentMap class and the
    Cython tracking functions. Think of a way to make the whole
    TransverseSegmentMap class a Cython class.
    Optimize the Cython functions. """

    def __init__(self,
            alpha_x_s0, beta_x_s0, D_x_s0, alpha_x_s1, beta_x_s1, D_x_s1,
            alpha_y_s0, beta_y_s0, D_y_s0, alpha_y_s1, beta_y_s1, D_y_s1,
            dQ_x, dQ_y, *segment_detuners):
        """ Return an instance of the TransverseSegmentMap class. The
        values of the TWISS parameters alpha_{x,y} and beta_{x,y} as
        well as of the dispersion coefficients D_{x,y} (not yet
        implemented) are given at the beginning s0 and at the end s1 of
        the corresponding segment. The dQ_{x,y} denote the betatron
        tunes normalized to the (relative) segment length. The
        SegmentDetuner objects present in this segment are passed and
        zipped to a list via the argument *segment_detuners.
        The matrices self.I and self.J are constant and are calculated
        only once at instantiation of the TransverseSegmentMap.
        In case the list of segment_detuners is empty, the transport matrix
        M is a constant and the same for all the particles. Hence it is
        calculated only once. Also, the self.track(self, beam) method is
        bound to self.track_without_detuners(self, beam) which calls the
        corresponding Cython function.
        In the other case of a non-empty list, the transport matrix M
        is not constant and must be calculated for each particle
        individually. The self.track(self, beam) method is bound to
        self.track_with_detuners(self, beam) which calls the
        corresponding Cython function. """
        self.dQ_x = dQ_x
        self.dQ_y = dQ_y

        if (D_x_s0 != 0 or D_x_s1 != 0 or D_y_s0 != 0 or D_y_s1 != 0):
            raise NotImplementedError('Non-zero values have been \n' +
                'specified for the dispersion coefficients D_{x,y}.\n' +
                'But, the effects of dispersion are not yet implemented. \n')

        self._build_segment_map(alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1,
                                alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1)

        if segment_detuners:
            self.segment_detuners = segment_detuners
            self.track = self.track_with_detuners
        else:
            self._calculate_transport_matrix()
            self.track = self.track_without_detuners

    def _build_segment_map(self, alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1,
                           alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1):
        """ Calculate matrices I and J which are decoupled from the
        phase advance dependency and only depend on the TWISS parameters
        at the boundaries of the accelerator segment. These matrices are
        constant and hence need to be calculated only once at
        instantiation of the TransverseSegmentMap. """
        self.I = np.zeros((4, 4))
        self.J = np.zeros((4, 4))

        # Sine component.
        self.I[0,0] = np.sqrt(beta_x_s1 / beta_x_s0)
        self.I[0,1] = 0.
        self.I[1,0] = (np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
                      (alpha_x_s0 - alpha_x_s1))
        self.I[1,1] = np.sqrt(beta_x_s0 / beta_x_s1)
        self.I[2,2] = np.sqrt(beta_y_s1 / beta_y_s0)
        self.I[2,3] = 0.
        self.I[3,2] = (np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
                      (alpha_y_s0 - alpha_y_s1))
        self.I[3,3] = np.sqrt(beta_y_s0 / beta_y_s1)

        # Cosine component.
        self.J[0,0] = np.sqrt(beta_x_s1 / beta_x_s0) * alpha_x_s0
        self.J[0,1] = np.sqrt(beta_x_s0 * beta_x_s1)
        self.J[1,0] = -(np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
                      (1. + alpha_x_s0 * alpha_x_s1))
        self.J[1,1] = -np.sqrt(beta_x_s0 / beta_x_s1) * alpha_x_s1
        self.J[2,2] = np.sqrt(beta_y_s1 / beta_y_s0) * alpha_y_s0
        self.J[2,3] = np.sqrt(beta_y_s0 * beta_y_s1)
        self.J[3,2] = -(np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
                      (1. + alpha_y_s0 * alpha_y_s1))
        self.J[3,3] = -np.sqrt(beta_y_s0 / beta_y_s1) * alpha_y_s1

    def _calculate_transport_matrix(self):
        """ If the list of segment_detuners is empty, the phase advance
        is the same for all particles and is simply given by the betatron
        tune. It makes sense to calculate the transport matrix M only
        once at the beginninig and store it. """
        dphi_x = 2. * np.pi * self.dQ_x
        dphi_y = 2. * np.pi * self.dQ_y

        c_dphi_x = np.cos(dphi_x)
        c_dphi_y = np.cos(dphi_y)
        s_dphi_x = np.sin(dphi_x)
        s_dphi_y = np.sin(dphi_y)

        self.M = np.zeros((4,4))
        self.M[0,0] = self.I[0,0] * c_dphi_x + self.J[0,0] * s_dphi_x
        self.M[0,1] = self.I[0,1] * c_dphi_x + self.J[0,1] * s_dphi_x
        self.M[1,0] = self.I[1,0] * c_dphi_x + self.J[1,0] * s_dphi_x
        self.M[1,1] = self.I[1,1] * c_dphi_x + self.J[1,1] * s_dphi_x
        self.M[2,2] = self.I[2,2] * c_dphi_y + self.J[2,2] * s_dphi_y
        self.M[2,3] = self.I[2,3] * c_dphi_y + self.J[2,3] * s_dphi_y
        self.M[3,2] = self.I[3,2] * c_dphi_y + self.J[3,2] * s_dphi_y
        self.M[3,3] = self.I[3,3] * c_dphi_y + self.J[3,3] * s_dphi_y

    def track_with_detuners(self, beam):
        """ This method is bound to the self.track(self, beam) method
        in case the self.segment_detuners is not empty. The dphi_{x,y}
        denote the phase advance in the horizontal and vertical plane
        respectively for the given accelerator segment. They are
        composed of the betatron tunes dQ_{x,y} and a possible
        incoherent tune shift introduced by detuner elements / effects
        defined in the list self.segment_detuners (they are all instances
        of the SegmentDetuner child classes). Hence, they are arrays of
        length beam.n_macroparticles. """

        # Calculate phase advance for this segment (betatron motion in
        # this segment and incoherent tune shifts introduced by detuning
        # effects).
        dphi_x = self.dQ_x
        dphi_y = self.dQ_y

        for element in self.segment_detuners:
            detune_x, detune_y = element.detune(beam)
            dphi_x += detune_x
            dphi_y += detune_y

        dphi_x *= 2. * np.pi
        dphi_y *= 2. * np.pi

        # Call Cython method to do the tracking.
        cytrack_with_detuners(beam.x, beam.xp, beam.y, beam.yp,
                              dphi_x, dphi_y, self.I, self.J)

    def track_without_detuners(self, beam):
        """ This method is bound to the self.track(self, beam) method
        in case the self.segment_detuners list is empty. The dphi_{x,y}
        denote the phase advance in the horizontal and vertical plane
        respectively for the given accelerator segment. They are
        composed of the betatron tunes dQ_{x,y} only. They are scalar
        values. For this case, the transport matrix self.M is a constant
        and can be directly used. """

        # Calculate phase advance for this segment (betatron motion in
        # this segment and incoherent tune shifts introduced by detuning
        # effects).
        dphi_x = 2. * np.pi * self.dQ_x
        dphi_y = 2. * np.pi * self.dQ_y

        # Call Cython method to do the tracking.
        cytrack_without_detuners(beam.x, beam.xp, beam.y, beam.yp,
                                 dphi_x, dphi_y, self.M)


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef cytrack_with_detuners(double[::1] x, double[::1] xp, double[::1] y,
        double[::1] yp, double[::1] dphi_x, double[::1] dphi_y,
        double[:,::1] I, double[:,::1] J):
    """ Cython method to perform the transverse tracking / transport
    of the bunch for the case with an non-empty
    TransverseSegmentMap.segment_detuners list. In this case, the
    transport matrix M is not a constant and is different for all
    the particles (due to incoherent detuning). It must hence be
    computed for each individual particle from the matrices I, J and
    their particular phase advance.
    OpenMP support is prepared (and working), but set to one thread
    only for the moment to avoid large CPU time use and CPU time
    limit exceed on LSF. """

    cdef unsigned int n_particles = x.shape[0]

    cdef double c_dphi_x, s_dphi_x, c_dphi_y, s_dphi_y
    cdef double x_tmp, xp_tmp, y_tmp, yp_tmp

    cdef unsigned int i
    for i in prange(n_particles, nogil=True, num_threads=1):
        c_dphi_x = cos(dphi_x[i])
        s_dphi_x = sin(dphi_x[i])
        c_dphi_y = cos(dphi_y[i])
        s_dphi_y = sin(dphi_y[i])

        # Store coordinates in temporary variables before tracking.
        # Imitate Python's a, b = b, a.
        x_tmp = x[i]
        y_tmp = y[i]
        xp_tmp = xp[i]
        yp_tmp = yp[i]

        x[i] = ((I[0,0] * c_dphi_x + J[0,0] * s_dphi_x) * x_tmp +
                (I[0,1] * c_dphi_x + J[0,1] * s_dphi_x) * xp_tmp)
        y[i] = ((I[2,2] * c_dphi_y + J[2,2] * s_dphi_y) * y_tmp +
                (I[2,3] * c_dphi_y + J[2,3] * s_dphi_y) * yp_tmp)

        xp[i] = ((I[1,0] * c_dphi_x + J[1,0] * s_dphi_x) * x_tmp +
                 (I[1,1] * c_dphi_x + J[1,1] * s_dphi_x) * xp_tmp)
        yp[i] = ((I[3,2] * c_dphi_y + J[3,2] * s_dphi_y) * y_tmp +
                 (I[3,3] * c_dphi_y + J[3,3] * s_dphi_y) * yp_tmp)


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef cytrack_without_detuners(double[::1] x, double[::1] xp, double[::1] y,
        double[::1] yp, double[::1] dphi_x, double[::1] dphi_y,
        double[:,::1] M):
    """ Cython method to perform the transverse tracking / transport
    of the bunch for the case with an empty
    TransverseSegmentMap.segment_detuners list. In this case, the
    transport matrix M is a constant and the same for all the
    particles. It is hence computed only once at instantiation of
    the TransverseSegmentMap and directly passed to this cytrack
    method.
    OpenMP is disabled for this case as it does not lead to any
    speedup. """

    # Store coordinates in temporary variables before tracking.
    # Imitate Python's a, b = b, a.
    cdef double x_tmp, xp_tmp, y_tmp, yp_tmp

    cdef unsigned int n_particles = x.shape[0]
    cdef unsigned int i
    for i in xrange(n_particles):
        x_tmp = x[i]
        y_tmp = y[i]
        xp_tmp = xp[i]
        yp_tmp = yp[i]

        x[i] = M[0,0] * x_tmp + M[0,1] * xp_tmp
        y[i] = M[2,2] * y_tmp + M[2,3] * yp_tmp

        xp[i] = M[1,0] * x_tmp + M[1,1] * xp_tmp
        yp[i] = M[3,2] * y_tmp + M[3,3] * yp_tmp
