"""
The PyHEADTAIL.trackers.detuners module implemented in CYTHON (incl.
support for OpenMP). Collection classes are imported from the Python
PyHEADTAIL.trackers.detuners module.

NB. Cython does not allow the use of abstract classes which is why there
is no ABC 'Detuner' in this module. When adding a new detuning element
or effect, it must be made sure that the detune(beam) method is
implemented!

Description from PyHEADTAIL.trackers.detuners:
Module to describe devices/effects, such as chromaticity or octupole
magnets, leading to an incoherent detuning of the particles in the beam.
A detuner is (in general) present along the full circumference of the
accelerator and the detuning is applied proportionally along the ring.

The structure of this module is such that there is a DetunerCollection
object for each type of detuning effect present in the accelerator. It
provides a description of the detuning along the full circumference. The
accelerator is divided into segments (1 or more) and the
DetunerCollection can create and store a SegmentDetuner object of the
given type of detuning for each of these segments. A SegmentDetuner
object has a detune(beam) method that defines how the phase advance of
each particle in the beam is changed according to the formula describing
the effect.

@author Kevin Li, Michael Schenk
@date 12. October 2014
@brief Cython implementation of the detuners module to describe the
       elements/effects in an accelerator ring leading to an incoherent
       detuning.
@copyright CERN
"""
from cython.parallel import prange
cimport cython.boundscheck

import numpy as np
cimport numpy as np
from scipy.constants import e, c
from libc.math cimport cos, sin

from .detuners import AmplitudeDetuning, Chromaticity


cdef class ChromaticitySegment(object):
    """ Cython implementation of a detuning object for a segment of the
    accelerator ring to describe the incoherent detuning introduced by
    chromaticity effects.
    TODO Implement second and third order chromaticity effects. """

    cdef double dQp_x, dQp_y
    cdef int n_threads

    def __init__(self, dQp_x, dQp_y, n_threads=1):
        """ Return an instance of a ChromaticitySegment. The dQp_{x,y}
        denote the first order chromaticity coefficients scaled to the
        segment length.
        Note: There is no satisfying solution yet to set the number of
        threads self.n_threads when instantiating the segment detuners
        via the DetunerCollection class (which is the usual way to do
        it). """
        self.dQp_x = dQp_x
        self.dQp_y = dQp_y

        self.n_threads = n_threads

    @cython.boundscheck(False)
    def detune(self, beam):
        """ Calculate for every particle the change in phase advance
        (detuning) dQ_{x,y}  caused by first order chromaticity
        effects. """
        cdef double[::1] dp = beam.dp
        cdef unsigned int n_particles = dp.shape[0]
        cdef double[::1] dQ_x = np.zeros(n_particles, dtype=np.double)
        cdef double[::1] dQ_y = np.zeros(n_particles, dtype=np.double)

        cdef unsigned int i
        for i in prange(n_particles, nogil=True, num_threads=self.n_threads):
            dQ_x[i] = self.dQp_x * dp[i]
            dQ_y[i] = self.dQp_y * dp[i]

        return dQ_x, dQ_y


cdef class AmplitudeDetuningSegment(object):
    """ Cython implementation of a detuning object for a segment of
    the accelerator ring to describe detuning with amplitude (e.g.
    introduced by octupoles). """

    cdef double dapp_x, dapp_y, dapp_xy
    cdef double beta_x, beta_y
    cdef int n_threads

    def __init__(self, dapp_x, dapp_y, dapp_xy, beta_x, beta_y, n_threads=1):
        """ Return an instance of an AmplitudeDetuningSegment by passing
        the coefficients of detuning strength dapp_x, dapp_y, dapp_xy
        (scaled to the segment length. NOT normalized to Beam.p0 yet).
        Note that beta_{x,y} are only used to correctly calculate the
        transverse actions J_{x,y}. Although they have an influence on
        the strength of detuning, they have no actual effect on the
        strength of the octupoles (dapp_x, dapp_y, dapp_xy).
        Note: There is no satisfying solution yet to set the number of
        threads self.n_threads when instantiating the segment detuners
        via the DetunerCollection class (which is the usual way to do
        it). """
        self.beta_x = beta_x
        self.beta_y = beta_y

        self.dapp_x = dapp_x
        self.dapp_y = dapp_y
        self.dapp_xy = dapp_xy

        self.n_threads = n_threads

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def detune(self, beam):
        """ Linear amplitude detuning formula, usually used for detuning
        introduced by octupoles. The normalization of dapp_x, dapp_y,
        dapp_xy to the reference momentum is done here (compare
        documentation of AmplitudeDetuning class).
        J_x and J_y resp. denote the horizontal and vertical action of
        a specific particle. """
        cdef double[::1] x = beam.x
        cdef double[::1] y = beam.y
        cdef double[::1] xp = beam.xp
        cdef double[::1] yp = beam.yp
        cdef double p0 = beam.p0

        cdef unsigned int n_particles = x.shape[0]
        cdef double[::1] dQ_x = np.zeros(n_particles, dtype=np.double)
        cdef double[::1] dQ_y = np.zeros(n_particles, dtype=np.double)

        cdef double xp_floquet, yp_floquet, Jx, Jy
        cdef unsigned int i
        for i in prange(n_particles, nogil=True, num_threads=self.n_threads):
            xp_floquet = self.beta_x * xp[i]
            yp_floquet = self.beta_y * yp[i]

            Jx = (x[i]*x[i] + xp_floquet*xp_floquet) / (2.*self.beta_x)
            Jy = (y[i]*y[i] + yp_floquet*yp_floquet) / (2.*self.beta_y)

            dQ_x[i] = (self.dapp_x*Jx + self.dapp_xy*Jy) / p0
            dQ_y[i] = (self.dapp_y*Jy + self.dapp_xy*Jx) / p0

        return dQ_x, dQ_y
