"""
Module to describe devices/effects, such as chromaticity or octupole
magnets, leading to an incoherent detuning of the particles in the beam.
A detuner is (in general) present along the full circumference of the
accelerator and the detuning is applied proportionally along the ring.

The structure of this module is such that there is a DetunerCollection
object for each type of detuning effect present in the accelerator. It
provides a description of the detuning along the full circumference. The
accelerator is divided into segments (1 or more) and the
DetunerCollection can create and store a SegmentDetuner object of the
given type of detuning for each of the segments that are defined by the
user. A SegmentDetuner object has a detune(beam) method that defines how
the tune of each particle in the beam is changed according to the
formula describing the effect.

@author Kevin Li, Michael Schenk, Adrian Oeftiger
@date 23. June 2014
@brief Module to describe elements/effects in an accelerator leading to
       an incoherent detuning.
@copyright CERN
"""
from __future__ import division

import numpy as np
from scipy.constants import e, c

from abc import ABCMeta, abstractmethod


class SegmentDetuner(object):
    """
    Abstract base class for detuning elements and effects defined only
    for a segment of the accelerator ring (NB. The segment can also be
    given by the full circumference).
    Every detuner element/effect inheriting from this class must
    implement the detune(beam) method to describe the change in tune
    for each particle in the beam.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def detune(self, beam):
        pass


class ChromaticitySegment(SegmentDetuner):
    """
    Detuning object for a segment of the accelerator ring to describe
    the detuning introduced by chromaticity effects.

    TO DO:
    Implement second and third order chromaticity effects.
    """
    def __init__(self, dQp_x, dQp_y):
        self.dQp_x = dQp_x
        self.dQp_y = dQp_y

    def detune(self, beam):
        """
        Calculates for every particle the change in phase advance
        (detuning) dQ_x,y  caused by first-order chromaticity effects.
        """
        dQ_x = self.dQp_x * beam.dp
        dQ_y = self.dQp_y * beam.dp

        return dQ_x, dQ_y


class AmplitudeDetuningSegment(SegmentDetuner):
    """
    Detuning object for a segment of the accelerator ring to describe
    amplitude detuning (introduced by octupoles).
    """
    def __init__(self, dapp_x, dapp_y, dapp_xy, beta_x, beta_y):
        """
        Note: Beta_x and beta_y are only used to correctly calculate the
        transverse action J_x and J_y. Although they have an influence
        on the strength of detuning, they have no actual effect on the
        strength of the octupoles (dapp_x, dapp_y, dapp_xy).
        """
        self.beta_x = beta_x
        self.beta_y = beta_y

        self.dapp_x = dapp_x
        self.dapp_y = dapp_y
        self.dapp_xy = dapp_xy

    def detune(self, beam):
        """
        Linear amplitude detuning formula, usually used for detuning
        introduced by octupoles. The normalization of dapp_x, dapp_y,
        dapp_xy to the reference momentum is done here (compare
        documentation of AmplitudeDetuning class).
        J_x and J_y resp. denote the horizontal and vertical action of
        a specific particle.
        """
        Jx = (beam.x**2 + (self.beta_x * beam.xp)** 2) / (2. * self.beta_x)
        Jy = (beam.y**2 + (self.beta_y * beam.yp)** 2) / (2. * self.beta_y)

        dQ_x = (self.dapp_x * Jx + self.dapp_xy * Jy) / beam.p0
        dQ_y = (self.dapp_y * Jy + self.dapp_xy * Jx) / beam.p0

        return dQ_x, dQ_y


class DetunerCollection(object):
    """
    Base class for a collection of SegmentDetuner objects (see above).
    A detuner collection object defines the detuning for one complete
    turn around the accelerator ring for the given detuning element.
    Hence, the strength of detuning must be specified by the user as
    integrated over one turn.
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

    The individual SegmentDetuner objects stored by a DetunerCollection
    can be accessed via square brackets [i] with i the index of the
    segment.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_segment_detuner(self, segment_length, **kwargs):
        """
        Instantiates a SegmentDetuner of the given type for a segment
        of the accelerator ring. Note that the segment_length is given
        as a relative value, i.e. in units of accelerator circumference.
        It scales the one turn value for the detuning strength
        proportionally to the segment length. The method is called by
        the TransverseMap object which manages the creation of a detuner
        for every defined segment. The kwargs are used to e.g. pass the
        beta functions from the TransverseMap where necessary (e.g. for
        AmplitudeDetuning).
        """
        pass

    def __len__(self):
        return len(self.segment_detuners)

    def __getitem__(self, key):
        return self.segment_detuners[key]


class AmplitudeDetuning(DetunerCollection):
    """
    Collection class to contain/manage the segment-wise defined
    amplitude detuning elements (octupoles).
    """
    def __init__(self, app_x, app_y, app_xy):
        """
        Note that the convention used here is such that the coefficients
        app_x, app_y and app_xy are NOT normalized to the reference
        momentum beam.p0. The normalization to beam.p0 is done only in
        the detune(beam) method of the AmplitudeDetuningSegment.
        """
        self.app_x  = app_x
        self.app_y  = app_y
        self.app_xy = app_xy

        self.segment_detuners = []

    @classmethod
    def from_octupole_currents_LHC(cls, i_focusing, i_defocusing):
        """
        Calculate the constants of proportionality app_x, app_y and
        app_xy (== app_yx) for the amplitude detuning introduced by the
        LHC octupole magnets (aka. LHC Landau octupoles) from the
        electric currents i_focusing [A] and i_defocusing [A] flowing
        through the magnets. The maximum current is given by
        i_max = +/- 550 [A]. The values app_x, app_y, app_xy  obtained
        from the formulae are proportional to the strength of detuning
        for one complete turn around the accelerator. Furthermore, they
        must be converted to SI units.

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
        above.
        """
        i_max = 550.  # [A]
        E_max = 7000. # [GeV]

        app_x  = E_max * (267065. * i_focusing / i_max -
            7856. * i_defocusing / i_max)
        app_y  = E_max * (9789. * i_focusing / i_max -
            277203. * i_defocusing / i_max)
        app_xy = E_max * (-102261. * i_focusing / i_max +
            93331. * i_defocusing / i_max)

        convert_to_SI = e / (1.e-9 * c)
        app_x *= convert_to_SI
        app_y *= convert_to_SI
        app_xy *= convert_to_SI

        return cls(app_x, app_y, app_xy)

    def generate_segment_detuner(self, segment_length, **kwargs):
        dapp_x = self.app_x * segment_length
        dapp_y = self.app_y * segment_length
        dapp_xy = self.app_xy * segment_length

        detuner = AmplitudeDetuningSegment(dapp_x, dapp_y, dapp_xy, **kwargs)
        self.segment_detuners.append(detuner)


class Chromaticity(DetunerCollection):
    """
    Collection class to contain/manage the segment-wise defined elements
    that introduce detuning as a result of first-order chromaticity
    effects.
    """
    def __init__(self, Qp_x, Qp_y):
        self.Qp_x = Qp_x
        self.Qp_y = Qp_y

        self.segment_detuners = []

    def generate_segment_detuner(self, segment_length, **kwargs):
        dQp_x = self.Qp_x * segment_length
        dQp_y = self.Qp_y * segment_length

        detuner = ChromaticitySegment(dQp_x, dQp_y)
        self.segment_detuners.append(detuner)
