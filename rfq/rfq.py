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

import trackers.detuners

'''
Detuner-framework.
'''

"""
Different implementations/parameterizations of the RF Quadrupole as a detuner. The formulae are mainly based on
'Radio frequency quadrupole for Landau damping in accelerators', A. Grudiev, 2014.
"""
class RFQSegmentGeneral(Detuner):
    """
    Most general RFQ detuner.
    """
    def __init__(self, dapp_xz, dapp_yz, omega_RF, phi_0_RF):

        self.omega_RF = omega_RF
        self.phi_0_RF = phi_0_RF
        self.dapp_xz  = dapp_xz
        self.dapp_yz  = dapp_yz
        

    def detune(self, beam):
        """
        Warning: relativistic beta is assumed to be exactly the same for all particles.
        """
        cos_dependence = cos(self.omega_RF * beam.z / (beam.beta * c) + self.phi_0_RF) / ((1 + beam.dp) * beam.p0)
        dphi_x = self.dapp_xz * cos_dependence 
        dphi_y = self.dapp_yz * cos_dependence

        return dphi_x, dphi_y
        

class RFQSegmentOnCrest(Detuner):
    """
    RFQ detuner with beam entering on crest (pure cosine dependence). Using approximation cos(x) \approx 1-x^2/2.
    """
    def __init__(self, dapp_xz, dapp_yz, omega_RF):

        self.omega_RF = omega_RF
        self.dapp_xz  = dapp_xz
        self.dapp_yz  = dapp_yz
        

    def detune(self, beam):
        """
        Warning: relativistic beta is assumed to be exactly the same for all particles.
        """
        approximate_cos_dependence = (1. - 0.5 * (beam.z * self.omega_RF / (beam.beta * c))**2) /  ((1 + beam.dp) * beam.p0)
        dphi_x = self.dapp_xz * approximate_cos_dependence 
        dphi_y = self.dapp_yz * approximate_cos_dependence

        return dphi_x, dphi_y


class RFQSegmentOffCrest(Detuner):
    """
    RFQ detuner with beam entering off crest (pure sine dependence). Using approximation sin(x) \approx 1+x.
    """
    def __init__(self, dapp_xz, dapp_yz, omega_RF):

        self.omega_RF = omega_RF
        self.dapp_xz  = dapp_xz
        self.dapp_yz  = dapp_yz
        

    def detune(self, beam):
        """
        Warning: relativistic beta is assumed to be exactly the same for all particles.
        """
        approximate_sin_dependence = (beam.z * self.omega_RF / (beam.beta * c)) /  ((1 + beam.dp) * beam.p0)
        dphi_x = self.dapp_xz * approximate_sin_dependence 
        dphi_y = self.dapp_yz * approximate_sin_dependence
        
        return dphi_x, dphi_y


"""
Collection class for the RFQ detuner. This is the class instantiated explicitly by the user. It uses 1-turn integrated
values as input and instantiates an RFQ detuner for each segment in 's' with a detuning proportional to the segment length.
"""
class RFQ(object):
     
    def __init__(self, s, beta_x, beta_y, voltage_RF, omega_RF, **kwargs):

        self.s = s
        self.omega_RF   = omega_RF
        self.voltage_RF = voltage_RF

        for key, value in kwargs.iteritems():
            setattr(self, key, value)

        scale_to_segment = np.diff(s) / s[-1]
        app_xz = -beta_x * voltage_RF * e / (omega_RF * 2. * np.pi)
        app_yz =  beta_y * voltage_RF * e / (omega_RF * 2. * np.pi)
        self.dapp_xz = scale_to_segment * app_xz
        self.dapp_yz = scale_to_segment * app_yz
            

    @classmethod
    def as_general(cls, s, beta_x, beta_y, voltage_RF, omega_RF, phi_0_RF):

        rfq_collection = cls(s, beta_x, beta_y, voltage_RF, omega_RF, phi_0_RF=phi_0_RF)
            
        segment_detuners = []
        n_segments = len(s) - 1
        for seg in range(n_segments):
            segment_detuner = RFQSegmentGeneral(rfq_collection.dapp_xz[seg], rfq_collection.dapp_yz[seg], omega_RF, phi_0_RF)
            segment_detuners.append(segment_detuner)

        rfq_collection.segment_detuners = segment_detuners
                        
        return rfq_collection


    @classmethod
    def as_on_crest(cls, s, beta_x, beta_y, voltage_RF, omega_RF):

        rfq_collection = cls(s, beta_x, beta_y, voltage_RF, omega_RF)

        segment_detuners = []
        n_segments = len(s) - 1
        for seg in range(n_segments):
            segment_detuner = RFQSegmentOnCrest(rfq_collection.dapp_xz[seg], rfq_collection.dapp_yz[seg], omega_RF)
            segment_detuners.append(segment_detuner)

        rfq_collection.segment_detuners = segment_detuners
                        
        return rfq_collection


    @classmethod
    def as_off_crest(cls, s, beta_x, beta_y, voltage_RF, omega_RF):

        rfq_collection = cls(s, beta_x, beta_y, voltage_RF, omega_RF)

        segment_detuners = []
        n_segments = len(s) - 1
        for seg in range(n_segments):
            segment_detuner = RFQSegmentOffCrest(rfq_collection.dapp_xz[seg], rfq_collection.dapp_yz[seg], omega_RF)
            segment_detuners.append(segment_detuner)

        rfq_collection.segment_detuners = segment_detuners
                        
        return rfq_collection


    def __len__(self):

        return len(self.segment_detuners)

    
    def __getitem__(self, key):

        return self.segment_detuners[key]



'''
Longitudinal effect of RFQ. Always acting as kicker (only once per turn).
'''
