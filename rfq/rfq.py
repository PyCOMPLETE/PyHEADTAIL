'''
@author Michael Schenk
@date July, 10th 2014
@brief Implementation of RF quadrupole for Landau damping.
@copyright CERN
'''
from __future__ import division
from abc import ABCMeta, abstractmethod
from scipy.constants import e, c
import numpy as np

cos = np.cos
sin = np.sin


'''
DETUNER MODEL.
'''
"""
Different implementations/parameterizations of the RF Quadrupole as a detuner.
The formulae are based on 'Radio frequency quadrupole for Landau damping in
accelerators', A. Grudiev, 2014.
"""
class RFQTransverseSegmentGeneral(object):
    """
    General RFQ detuner using exact cosine dependence and allowing for phase shift.
    """
    def __init__(self, dapp_xz, dapp_yz, omega_RF, phi_0_RF):

        self.omega_RF = omega_RF
        self.phi_0_RF = phi_0_RF
        self.dapp_xz  = dapp_xz
        self.dapp_yz  = dapp_yz


    def detune(self, beam):

        cos_dependence = cos(self.omega_RF * beam.z / (beam.beta * c) + self.phi_0_RF)
        cos_dependence /= ((1. + beam.dp) * beam.p0)

        dphi_x = self.dapp_xz * cos_dependence
        dphi_y = self.dapp_yz * cos_dependence

        return dphi_x, dphi_y


class RFQTransverseSegmentOnCrest(object):
    """
    RFQ detuner with beam entering on crest (pure cosine dependence).
    Using approximation cos(x) \approx 1-x^2/2.
    """
    def __init__(self, dapp_xz, dapp_yz, omega_RF):

        self.omega_RF = omega_RF
        self.dapp_xz  = dapp_xz
        self.dapp_yz  = dapp_yz


    def detune(self, beam):

        approximate_cos_dependence = 1. - 0.5 * (beam.z * self.omega_RF / (beam.beta * c)) ** 2
        approximate_cos_dependence /= ((1. + beam.dp) * beam.p0)

        dphi_x = self.dapp_xz * approximate_cos_dependence
        dphi_y = self.dapp_yz * approximate_cos_dependence

        return dphi_x, dphi_y



class RFQTransverseSegmentOffCrest(object):
    """
    RFQ detuner with beam entering off crest (pure sine dependence).
    Using approximation sin(x) \approx 1+x.
    """
    def __init__(self, dapp_xz, dapp_yz, omega_RF):

        self.omega_RF = omega_RF
        self.dapp_xz  = dapp_xz
        self.dapp_yz  = dapp_yz


    def detune(self, beam):

        approximate_sin_dependence  = beam.z * self.omega_RF / (beam.beta * c)
        approximate_sin_dependence /= ((1. + beam.dp) * beam.p0)

        dphi_x = self.dapp_xz * approximate_sin_dependence
        dphi_y = self.dapp_yz * approximate_sin_dependence

        return dphi_x, dphi_y


"""
Collection class for the RFQ detuner. This is the class instantiated explicitly by the user.
It uses 1-turn integrated values as input and instantiates an RFQ detuner for each segment
in 's' with a detuning proportional to the segment length.
"""
class RFQTransverse(object):

    def __init__(self, beta_x, beta_y, v2_RF, omega_RF, **kwargs):

        self.beta_x = beta_x
        self.beta_y = beta_y

        self.omega_RF = omega_RF
        self.v2_RF    = v2_RF

        for key, value in kwargs.iteritems():
            setattr(self, key, value)

        self.segment_detuners = []


    @classmethod
    def as_general(cls, beta_x, beta_y, v2_RF, omega_RF, phi_0_RF):

        self = cls(beta_x, beta_y, v2_RF, omega_RF, phi_0_RF=phi_0_RF)
        self.generate_segment_detuner = self.generate_segment_detuner_general

        return self

    def generate_segment_detuner_general(self, relative_segment_length):

        dapp_xz =  self.beta_x * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length
        dapp_yz = -self.beta_y * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length

        self.segment_detuners.append(RFQTransverseSegmentGeneral(dapp_xz, dapp_yz, self.omega_RF,
                                                                 self.phi_0_RF))

    @classmethod
    def as_on_crest(cls, beta_x, beta_y, v2_RF, omega_RF):

        self = cls(beta_x, beta_y, v2_RF, omega_RF)
        self.generate_segment_detuner = self.generate_segment_detuner_on_crest

        return self

    def generate_segment_detuner_on_crest(self, relative_segment_length):

        dapp_xz =  self.beta_x * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length
        dapp_yz = -self.beta_y * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length

        self.segment_detuners.append(RFQTransverseSegmentOnCrest(dapp_xz, dapp_yz, self.omega_RF))


    @classmethod
    def as_off_crest(cls, beta_x, beta_y, v2_RF, omega_RF):

        self = cls(beta_x, beta_y, v2_RF, omega_RF)
        self.generate_segment_detuner = self.generate_segment_detuner_off_crest

        return self


    def generate_segment_detuner_off_crest(self, relative_segment_length):

        dapp_xz =  self.beta_x * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length
        dapp_yz = -self.beta_y * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length

        self.segment_detuners.append(RFQTransverseSegmentOffCrest(dapp_xz, dapp_yz, self.omega_RF))


    def __len__(self):

        return len(self.segment_detuners)


    def __getitem__(self, key):

        return self.segment_detuners[key]


class RFQTransverseKick(object):

    def __init__(self, v2_RF, omega_RF, phi_0_RF):

        self.v2_RF    = v2_RF
        self.omega_RF = omega_RF
        self.phi_0_RF = phi_0_RF


    def track(self, beam):

        cos_dependence = cos(self.omega_RF * beam.z / (beam.beta * c) + self.phi_0_RF)
        k = 2. * e * self.v2_RF / self.omega_RF

        delta_p_x = beam.x * k * cos_dependence
        delta_p_y = -beam.y * k * cos_dependence

        beam.xp += delta_p_x / beam.p0
        beam.yp += delta_p_y / beam.p0


'''
Longitudinal effect of RFQ. Always acting as a localized kick (applied once per turn).
'''
class RFQLongitudinalKick(object):

    def __init__(self, v2_RF, omega_RF, phi_0_RF):

        self.v2_RF    = v2_RF
        self.omega_RF = omega_RF
        self.phi_0_RF = phi_0_RF


    def track(self, beam):

        delta_p = - e * self.v2_RF / (beam.beta * c) * (beam.x ** 2 - beam.y ** 2) * \
                    sin(self.omega_RF * beam.z / (beam.beta * c) + self.phi_0_RF)

        beam.dp = beam.dp + delta_p / beam.p0
