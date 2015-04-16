"""
The PyHEADTAIL.trackers.detuners module implemented in Cython (incl.
support for OpenMP). Collection classes are still written in Python
and correspond to those of the PyHEADTAIL.trackers.detuners module.

NB I. For 1st order chromaticity, the ChromaticitySegment.detune(beam)
cython method performs about the same as the Python implementation
(8ms for 1e6 particles). However, for higher order chromaticity, the
speed-up gets significant:
  2nd order chroma: factor > 2 (single-threaded)
                    10ms (cython) vs. 23ms (python) for 1e6 particles.
  3rd order chroma: factor ~ 3-4 (single-threaded)
                    11ms (cython) vs. 38ms (python) for 1e6 particles.

NB II. Cython does not allow the use of abstract classes to inherit from
which is why there is no ABC Detuner here. When adding a new detuning
element or effect, it must be made sure that the detune(beam) method is
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

TODO
Instead of passing the python object beam to the Cython methods, it could
be better to make a wrapper to the Cython function to pass directly the
arrays beam.x, beam.y, ... for better performance.
The number of threads num_threads in the cython.parallel.prange for loops
is hard-coded to 1 for the moment. Find a good way to make it controllable
by the user.

@author Kevin Li, Michael Schenk, Stefan Hegglin
@date 12. October 2014
@brief Cython implementation of the detuners module to describe the
       elements/effects in an accelerator ring leading to an incoherent
       detuning.
@copyright CERN
"""
from cython.parallel cimport prange
cimport cython

import numpy as np
cimport numpy as np
from scipy.constants import e, c
from libc.math cimport cos, sin
from collections import Iterable

cdef class ChromaticitySegment(object):
    """ Cython implementation of a detuning object for a segment of the
    accelerator ring to describe the incoherent detuning introduced by
    chromaticity effects. """
    cdef double[::1] coeffs_x, coeffs_y
    cdef int order_x, order_y

    def __init__(self, dQp_x, dQp_y):
        """ Return an instance of a ChromaticitySegment. The dQp_{x,y}
        denote numpy arrays containing first, second, third, ... order
        chromaticity coefficients scaled to the segment length. """
        if not (isinstance(dQp_x, Iterable) and isinstance(dQp_y, Iterable)):
            raise TypeError("Scalar values are no longer accepted for dQp_x" +
                            " and dQp_y. They must now be iterables (e.g." +
                            " lists, tuples or numpy arrays) following the" +
                            " correct order [Q', Q'', Q''', ...]. This is" +
                            " true even if the only non-zero chromaticity" +
                            " coefficient is the linear one.")
        self.order_x = dQp_x.shape[0]
        self.order_y = dQp_y.shape[0]
        self.coeffs_x = np.empty(self.order_x, dtype=np.double)
        self.coeffs_y = np.empty(self.order_y, dtype=np.double)

        cdef unsigned int i
        for i in xrange(self.order_x):
            self.coeffs_x[i] = dQp_x[i] / factorial(i+1)
        for i in xrange(self.order_y):
            self.coeffs_y[i] = dQp_y[i] / factorial(i+1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def detune(self, beam):
        """ Calculate for every particle the change in phase advance
        (detuning) dQ_{x,y}  caused by chromaticity effects. """
        cdef double[::1] dp = beam.dp
        cdef unsigned int n_particles = dp.shape[0]
        cdef double[::1] dQ_x = np.empty(n_particles, dtype=np.double)
        cdef double[::1] dQ_y = np.empty(n_particles, dtype=np.double)

        cdef unsigned int i
        if self.order_x > 1 or self.order_y > 1:
            for i in prange(n_particles, nogil=True, num_threads=1):
                dQ_x[i] = eval_poly(self.coeffs_x, self.order_x, dp[i])
                dQ_y[i] = eval_poly(self.coeffs_y, self.order_y, dp[i])
        else:
            for i in prange(n_particles, nogil=True, num_threads=1):
                dQ_x[i] = self.coeffs_x[0] * dp[i]
                dQ_y[i] = self.coeffs_y[0] * dp[i]
        return dQ_x, dQ_y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double factorial(int n):
    """ Calculate factorial of n. """
    cdef double result = 1.
    cdef unsigned int i
    for i in xrange(1, n):
        result *= (i+1)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double eval_poly(double[::1] coeffs, int n, double x) nogil:
    """ Evaluate polynomial of order n with coefficients 'coeffs'
    at value x using Horner's method. Zeroth order coefficient is
    assumed to be always 0, i.e. coeffs[0] is c1, ... """
    cdef double result = 0.
    cdef unsigned int i
    for i in xrange(n, 0, -1):
        result += coeffs[i-1]
        result *= x
    return result


cdef class AmplitudeDetuningSegment(object):
    """ Cython implementation of a detuning object for a segment of
    the accelerator ring to describe detuning with amplitude (e.g.
    introduced by octupoles). """

    cdef double dapp_x, dapp_y, dapp_xy
    cdef double beta_x, beta_y
    cdef double alpha_x, alpha_y

    def __init__(self, dapp_x, dapp_y, dapp_xy, alpha_x, beta_x,
                 alpha_y, beta_y):
        """ Return an instance of an AmplitudeDetuningSegment by passing
        the coefficients of detuning strength dapp_x, dapp_y, dapp_xy
        (scaled to the segment length. NOT normalized to Beam.p0 yet).
        Note that beta_{x,y} are only used to correctly calculate the
        transverse actions J_{x,y}. Although they have an influence on
        the strength of detuning, they have no actual effect on the
        strength of the octupoles (dapp_x, dapp_y, dapp_xy). """
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        self.beta_x = beta_x
        self.beta_y = beta_y

        self.dapp_x = dapp_x
        self.dapp_y = dapp_y
        self.dapp_xy = dapp_xy

    @cython.boundscheck(False)
    @cython.wraparound(False)
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
        cdef double[::1] dQ_x = np.empty(n_particles, dtype=np.double)
        cdef double[::1] dQ_y = np.empty(n_particles, dtype=np.double)

        cdef double alpha_x2 = self.alpha_x * self.alpha_x
        cdef double alpha_y2 = self.alpha_y * self.alpha_y
        cdef double Jx, Jy
        cdef unsigned int i
        for i in prange(n_particles, nogil=True, num_threads=1):
            Jx = 0.5 * ((1 + alpha_x2) / self.beta_x * x[i]*x[i]
                        + 2*self.alpha_x * x[i] * xp[i]
                        + self.beta_x * xp[i]*xp[i]
                       )
            Jy = 0.5 * ((1 + alpha_y2) / self.beta_y * y[i]*y[i]
                        + 2*self.alpha_y * y[i] * yp[i]
                        + self.beta_y * yp[i]*yp[i]
                       )
            dQ_x[i] = (self.dapp_x*Jx + self.dapp_xy*Jy) / p0
            dQ_y[i] = (self.dapp_y*Jy + self.dapp_xy*Jx) / p0

        return dQ_x, dQ_y


''' DetunerCollection classes '''

from abc import abstractmethod, ABCMeta

class DetunerCollection(object):
    """ Abstract base class for a collection of SegmentDetuner objects
    (see above). A detuner collection object defines the detuning for
    one complete turn around the accelerator ring for the given
    detuning element. Hence, the strength of detuning must be specified
    by the user as integrated over one turn.
    The accelerator ring is divided into a number of segments (often
    there is just 1). To apply the detuning segment-wise, a
    SegmentDetuner object is instantiated for each of the accelerator
    segments and the detuning strength is chosen to be proportional to
    the segment_length (normalized to the circumference of the
    accelerator). The instantiation of SegmentDetuner objects is handled
    by the generate_segment_detuner method. This method is called by the
    TransverseSegmentMap object as it contains the information of how
    the segments of the accelerator are defined by the user. The
    SegmentDetuner objects are stored in the segment_detuners list (in
    order of segments along the ring) within the DetunerCollection.

    Since the DetunerCollection is implemented as a sequence, the
    individual SegmentDetuner objects stored by a DetunerCollection can
    be accessed via square brackets [i] where i is the index of the
    segment. """

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_segment_detuner(self, segment_length, **kwargs):
        """ Instantiates a SegmentDetuner of the given type for a
        segment of the accelerator ring. Note that the segment_length
        is given as a relative value, i.e. in units of accelerator
        circumference. It scales the one turn value for the detuning
        strength proportionally to the segment length. The method is
        called by the TransverseMap object which manages the creation
        of a detuner for every defined segment. The kwargs are used to
        e.g. pass the beta functions from the TransverseMap where
        necessary (e.g. for AmplitudeDetuning). """
        pass

    def __len__(self):
        return len(self.segment_detuners)

    def __getitem__(self, key):
        return self.segment_detuners[key]


class AmplitudeDetuning(DetunerCollection):
    """ Collection class to contain/manage the segment-wise defined
    amplitude detuning elements (octupoles). They are stored in the
    self.segment_detuners list. """

    def __init__(self, app_x, app_y, app_xy):
        """ Return an instance of the AmplitudeDetuning
        DetunerCollection class. The coefficients app_x, app_y, app_xy
        are the detuning strengths (one-turn values). Note that the
        convention used here is such that they are NOT normalized to
        the reference momentum Beam.p0. The normalization to Beam.p0
        is done only in the detune(beam) method of the
        AmplitudeDetuningSegment. """
        self.app_x  = app_x
        self.app_y  = app_y
        self.app_xy = app_xy

        self.segment_detuners = []

    @classmethod
    def from_octupole_currents_LHC(cls, i_focusing, i_defocusing):
        """ Calculate the constants of proportionality app_x, app_y and
        app_xy (== app_yx) for the amplitude detuning introduced by the
        LHC octupole magnets (aka. LHC Landau octupoles) from the
        electric currents i_focusing [A] and i_defocusing [A] flowing
        through the magnets. The maximum current is given by
        i_max = +/- 550 [A]. The values app_x, app_y, app_xy  obtained
        from the formulae are proportional to the strength of detuning
        for one complete turn around the accelerator, i.e. one-turn
        values.

        The calculation is based on formulae (3.6) taken from 'The LHC
        transverse coupled-bunch instability' by N. Mounet, EPFL PhD
        Thesis, 2012. Values (hard-coded numbers below) are valid for
        LHC Landau octupoles before LS1. Beta functions in x and y are
        correctly taken into account. Note that here, the values of
        app_x, app_y and app_xy are not normalized to the reference
        momentum p0. This is done only during the calculation of the
        detuning in the corresponding detune method of the
        AmplitudeDetuningSegment.

        More detailed explanations and references on how the formulae
        were obtained are given in the PhD thesis (pg. 85ff) cited
        above. """
        i_max = 550.  # [A]
        E_max = 7000. # [GeV]

        app_x  = E_max * (267065. * i_focusing / i_max -
            7856. * i_defocusing / i_max)
        app_y  = E_max * (9789. * i_focusing / i_max -
            277203. * i_defocusing / i_max)
        app_xy = E_max * (-102261. * i_focusing / i_max +
            93331. * i_defocusing / i_max)

        # Convert to SI units.
        convert_to_SI = e / (1.e-9 * c)
        app_x *= convert_to_SI
        app_y *= convert_to_SI
        app_xy *= convert_to_SI

        return cls(app_x, app_y, app_xy)

    def generate_segment_detuner(self, segment_length, **kwargs):
        """ Instantiates an AmplitudeDetuningSegment for the specified
        segment of the accelerator ring. Note that the segment_length
        is given as a relative value, i.e. in units of accelerator
        circumference. It scales the one-turn values for the detuning
        strength proportionally to the segment length. The method is
        called by the TransverseMap object which manages the creation
        of a detuner for every defined segment. The kwargs are used to
        pass the beta functions from the TransverseMap at the given
        segment. """
        dapp_x = self.app_x * segment_length
        dapp_y = self.app_y * segment_length
        dapp_xy = self.app_xy * segment_length

        detuner = AmplitudeDetuningSegment(dapp_x, dapp_y, dapp_xy, **kwargs)
        self.segment_detuners.append(detuner)


class Chromaticity(DetunerCollection):
    """ Collection class to contain/manage the segment-wise defined
    elements that introduce detuning as a result of chromaticity
    effects.  They are stored in the self.segment_detuners list. """

    def __init__(self, Qp_x, Qp_y):
        """ Return an instance of a Chromaticity DetunerCollection
        class. The Qp_{x,y} are lists containing first, second, third,
        ... order chromaticity coefficients (one-turn values), aka.
        Q'_{x,y}, Q''_{x,y} (Q-prime, Q-double-prime), .... """
        if not (isinstance(Qp_x, Iterable) and isinstance(Qp_y, Iterable)):
            raise TypeError("Scalar values are no longer accepted for Qp_x" +
                            " and Qp_y. They must now be iterables (e.g." +
                            " lists, tuples or numpy arrays) following the" +
                            " correct order [Q', Q'', Q''', ...]. This is" +
                            " true even if the only non-zero chromaticity" +
                            " coefficient is the linear one.")
        self.Qp_x = Qp_x
        self.Qp_y = Qp_y

        self.segment_detuners = []

    def generate_segment_detuner(self, segment_length, **kwargs):
        """ Instantiates a ChromaticitySegment for the specified
        segment of the accelerator ring. Note that the segment_length
        is given as a relative value, i.e. in units of accelerator
        circumference. It scales the one-turn values for the detuning
        strength proportionally to the segment length. The method is
        called by the TransverseMap object which manages the creation
        of a detuner for every defined segment. """
        dQp_x = np.array([ Qp * segment_length for Qp in self.Qp_x ])
        dQp_y = np.array([ Qp * segment_length for Qp in self.Qp_y ])

        detuner = ChromaticitySegment(dQp_x, dQp_y)
        self.segment_detuners.append(detuner)
