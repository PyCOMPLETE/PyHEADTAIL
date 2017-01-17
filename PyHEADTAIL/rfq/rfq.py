"""
This module contains the Python implementation of a pillbox-cavity RF
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

@author Michael Schenk, Adrian Oeftiger
@date July, 10th 2014
@brief Python implementation of a pillbox cavity RF quadrupole for
       Landau damping.
@copyright CERN
"""
from __future__ import division

from abc import ABCMeta, abstractmethod
from scipy.constants import c, e
import numpy as np
from ..trackers.detuners import DetunerCollection
import PyHEADTAIL.general.pmath as pm


class RFQTransverseDetuner(DetunerCollection):
    """Collection class to contain/manage the segment-wise defined
    RFQ elements RFQTransverseDetunerSegment acting on the
    betatron tunes (detuner model of the RFQ). This is a pure
    Python class and it derives from the DetunerCollection class
    defined in the module PyHEADTAIL.trackers.detuners.
    """

    def __init__(self, v_2, omega, phi_0, beta_x_RFQ, beta_y_RFQ):
        """An RFQ element is fully characterized by the parameters
          v_2:   quadrupolar expansion coefficient of the accelerating
                 voltage (~strength of the RFQ), in [V/m^2]. One-turn
                 value.
          omega: Angular frequency of the RF wave, in [rad/s].
          phi_0: Constant phase offset wrt. bunch center (z=0), in
                 [rad].

        beta_x_RFQ and beta_y_RFQ are the beta functions at the
        position of the RFQ, although in the detuner model of the RFQ,
        the RFQ should not actually be understood as being localized.
        """
        self.v_2 = v_2
        self.omega = omega
        self.phi_0 = phi_0
        self.beta_x_RFQ = beta_x_RFQ
        self.beta_y_RFQ = beta_y_RFQ
        self.segment_detuners = []

    def generate_segment_detuner(self, dmu_x, dmu_y, **kwargs):
        """Instantiate a RFQTransverseSegmentDetuner for the
        specified segment of the accelerator ring.
        Note that the bare betatron
        phase advances over the current segment, dmu_x and dmu_y, are
        given as relative values, i.e. in units of the overall phase
        advance around the whole accelerator (the betatron tune).
        The method is called by the TransverseMap object which manages
        the creation of a detuner for every defined segment.
        """
        dapp_xz = self.beta_x_RFQ * self.v_2 * e / (2.*np.pi*self.omega)
        dapp_yz = -self.beta_y_RFQ * self.v_2 * e / (2.*np.pi*self.omega)
        dapp_xz *= dmu_x
        dapp_yz *= dmu_y

        detuner = RFQTransverseDetunerSegment(
            dapp_xz, dapp_yz, self.omega, self.phi_0)
        self.segment_detuners.append(detuner)


class RFQTransverseDetunerSegment(object):
    """Python implementation of the RFQ element acting directly on the
    particles' betatron tunes (i.e. RFQ detuner model).
    """

    def __init__(self, dapp_xz, dapp_yz, omega, phi_0):
        """Creates an instance of the RFQTransverseDetunerSegment
        class. The RFQ is characterized by
          omega:   Angular frequency of the RF wave, in [rad/s].
          phi_0:   Constant phase offset wrt. bunch center (z=0), in
                   [rad].
          dapp_xz: Strength of detuning in the horizontal plane, scaled
                   to the relative bare betatron phase advance in x.
          dapp_yz: Strength of detuning in the vertical plane, scaled
                   to the relative bare betatron phase advance in y.
        """
        self.dapp_xz = dapp_xz
        self.dapp_yz = dapp_yz
        self.omega = omega
        self.phi_0 = phi_0

    def detune(self, beam):
        """ Calculates for each particle its betatron detuning
        dQ_x, dQ_y according to formulae taken from [1] (see
        above).
            dQ_x = dapp_xz / p * \cos(omega / (beta c) z + phi_0)
            dQ_y = dapp_yz / p * \cos(omega / (beta c) z + phi_0)
        with
            dapp_xz = beta_x_RFQ  * v_2 * e / (2 Pi * omega)
            dapp_yz = -beta_y_RFQ  * v_2 * e / (2 Pi * omega)
        and p the particle momentum p = (1 + dp) p0.
        (Probably, it would make sense to approximate p by p0 for better
        performance). """
        p = (1. + beam.dp) * beam.p0
        cos_term = pm.cos(self.omega / (beam.beta * c) * beam.z + self.phi_0) / p
        dQ_x = self.dapp_xz * cos_term
        dQ_y = self.dapp_yz * cos_term

        return dQ_x, dQ_y


class RFQKick(object):
    """Python base class to describe the RFQ element in the
    localized kick model for both the transverse and the
    longitudinal coordinates.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def track(self, beam):
        pass


class RFQTransverseKick(RFQKick):
    """Python implementation of the RFQ element acting on the
    particles' transverse coordinates (i.e. localized kick
    model).
    """

    def __init__(self, v_2, omega, phi_0):
        """An RFQ element is fully characterized by the parameters
          v_2:   quadrupolar expansion coefficient of the
                 accelerating voltage (~strength of the RFQ), in
                 [V/m^2].
          omega: Angular frequency of the RF wave, in [rad/s].
          phi_0: Constant phase offset wrt. bunch center (z=0), in
                 [rad].
        """
        self.v_2 = v_2
        self.omega = omega
        self.phi_0 = phi_0

    def track(self, beam):
        """The formula that describes the transverse kick experienced
        by an ultra-relativistic particle traversing the RFQ
        longitudinally is based on the thin-lens approximation
            \Delta p_x = -x*(2 e v_2 / omega) *
                cos(omega z / (beta c) + phi_0),
            \Delta p_y =  y*(2 e v_2 / omega) *
                cos(omega z / (beta c) + phi_0).
        """
        cos_term = (2. * e * self.v_2 / self.omega *
            pm.cos(self.omega / (beam.beta * c) * beam.z + self.phi_0))

        beam.xp += -beam.x * cos_term / beam.p0
        beam.yp += beam.y * cos_term / beam.p0


class RFQLongitudinalKick(RFQKick):
    """Python implementation of the RFQ element acting on the
    particles' longitudinal coordinate dp."""

    def __init__(self, v_2, omega, phi_0):
        """An RFQ element is fully characterized by the parameters
          v_2:   quadrupolar expansion coefficient of the
                 accelerating voltage (~strength of the RFQ), in
                 [V/m^2].
          omega: Angular frequency of the RF wave, in [rad/s].
          phi_0: Constant phase offset wrt. bunch center (z=0), in
                 [rad].
        """
        self.v_2 = v_2
        self.omega = omega
        self.phi_0 = phi_0

    def track(self, beam):
        """The formula used to describe the longitudinal kick is given
        by
            \Delta p_z = -(x^2 - y^2) (e v_2 / (beta c)) *
                sin(omega z / (beta c) + phi_0).
        """
        sin_term = (e * self.v_2 / (beam.beta * c) *
            pm.sin(self.omega / (beam.beta * c) * beam.z + self.phi_0))
        beam.dp += -(beam.x*beam.x - beam.y*beam.y) * sin_term / beam.p0
