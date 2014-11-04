"""
The PyHEADTAIL.trackers.detuners module implemented in Cython (incl.
support for OpenMP). Collection classes are imported from the Python
PyHEADTAIL.trackers.detuners module.

NB. Cython does not allow the use of abstract classes to inherit from
which is why there is no ABC Detuner here. When adding a new detuning
element or effect, it must be made sure that the detune(beam) method is
implemented!

TODO
Instead of passing the python object beam to the Cython methods, it could
be better to make a wrapper to the Cython function to pass directly the
arrays beam.x, beam.y, ... for better performance.
The number of threads num_threads in the cython.parallel.prange for loops
is hard-coded to 1 for the moment. Find a good way to make it controllable
by the user.

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
from cython.parallel cimport prange
cimport cython

import numpy as np
cimport numpy as np
from scipy.constants import e, c
from libc.math cimport cos, sin


cdef class ChromaticitySegment(object):
    """ Cython implementation of a detuning object for a segment of the
    accelerator ring to describe the incoherent detuning introduced by
    chromaticity effects.
    TODO Implement second and third order chromaticity effects. """

    cdef double dQp_x, dQp_y

    def __init__(self, dQp_x, dQp_y):
        """ Return an instance of a ChromaticitySegment. The dQp_{x,y}
        denote the first order chromaticity coefficients scaled to the
        segment length. """
        self.dQp_x = dQp_x
        self.dQp_y = dQp_y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def detune(self, beam):
        """ Method to calculate for every particle the change in phase
        advance (detuning) dQ_{x,y} caused by first order chromaticity
        effects. """
        cdef double[::1] dp = beam.dp
        cdef unsigned int n_particles = dp.shape[0]
        cdef double[::1] dQ_x = np.empty(n_particles, dtype=np.double)
        cdef double[::1] dQ_y = np.empty(n_particles, dtype=np.double)

        cdef unsigned int i
        for i in prange(n_particles, nogil=True, num_threads=1):
            dQ_x[i] = self.dQp_x * dp[i]
            dQ_y[i] = self.dQp_y * dp[i]

        return dQ_x, dQ_y


cdef class AmplitudeDetuningSegment(object):
    """ Cython implementation of a detuning object for a segment of
    the accelerator ring to describe detuning with amplitude (e.g.
    introduced by octupoles). """

    cdef double dapp_x, dapp_y, dapp_xy
    cdef double beta_x, beta_y

    def __init__(self, dapp_x, dapp_y, dapp_xy, beta_x, beta_y):
        """ Return an instance of an AmplitudeDetuningSegment by passing
        the coefficients of detuning strength dapp_x, dapp_y, dapp_xy
        (scaled to the segment length. NOT normalized to Beam.p0 yet).
        Note that beta_{x,y} are only used to correctly calculate the
        transverse actions J_{x,y}. Although they have an influence on
        the strength of detuning, they have no actual effect on the
        strength of the octupoles (dapp_x, dapp_y, dapp_xy). """
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

        cdef double xp_floquet, yp_floquet, Jx, Jy
        cdef unsigned int i
        for i in prange(n_particles, nogil=True, num_threads=1):
            xp_floquet = self.beta_x * xp[i]
            yp_floquet = self.beta_y * yp[i]

            Jx = (x[i]*x[i] + xp_floquet*xp_floquet) / (2.*self.beta_x)
            Jy = (y[i]*y[i] + yp_floquet*yp_floquet) / (2.*self.beta_y)

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
    elements that introduce detuning as a result of first-order
    chromaticity effects.  They are stored in the self.segment_detuners
    list. """

    def __init__(self, Qp_x, Qp_y):
        """ Return an instance of a Chromaticity DetunerCollection
        class. The Qp_{x,y} are the first order chromaticity
        coefficients (one-turn values), aka. Q'_{x,y} (Q-prime). """
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
        dQp_x = self.Qp_x * segment_length
        dQp_y = self.Qp_y * segment_length

        detuner = ChromaticitySegment(dQp_x, dQp_y)
        self.segment_detuners.append(detuner)
