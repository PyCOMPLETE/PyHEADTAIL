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

import cython_rfq as cyrfq


# cos = np.cos
# sin = np.sin


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
        

    def detune(self, beam, dphi_x, dphi_y):

        cyrfq.rfq_detune(dphi_x, dphi_y, self.dapp_xz, self.dapp_yz,
                         self.omega_RF, self.phi_0_RF, beam.z, beam.dp,
                         beam.p0, beam.beta)
        

## class RFQTransverseSegmentOnCrest(object):
##     """
##     RFQ detuner with beam entering on crest (pure cosine dependence).
##     Using approximation cos(x) \approx 1-x^2/2.
##     """
##     def __init__(self, dapp_xz, dapp_yz, omega_RF):

##         self.omega_RF = omega_RF
##         self.dapp_xz  = dapp_xz
##         self.dapp_yz  = dapp_yz
        

##     def detune(self, beam, dphi_x, dphi_y):
##         pass
##         # cytrack



## class RFQTransverseSegmentOffCrest(object):
##     """
##     RFQ detuner with beam entering off crest (pure sine dependence).
##     Using approximation sin(x) \approx 1+x.
##     """
##     def __init__(self, dapp_xz, dapp_yz, omega_RF):

##         self.omega_RF = omega_RF
##         self.dapp_xz  = dapp_xz
##         self.dapp_yz  = dapp_yz
        

##     def detune(self, beam):

##         approximate_sin_dependence  = beam.z * self.omega_RF / (beam.beta * c)
##         approximate_sin_dependence /= ((1. + beam.dp) * beam.p0)

##         dphi_x = self.dapp_xz * approximate_sin_dependence 
##         dphi_y = self.dapp_yz * approximate_sin_dependence
        
##         return dphi_x, dphi_y


"""
Collection class for the RFQ detuner. This is the class instantiated explicitly by the user.
It uses 1-turn integrated values as input and instantiates an RFQ detuner for each segment 
in 's' with a detuning proportional to the segment length.
"""
class RFQTransverse(object):
     
    def __init__(self, v2_RF, omega_RF, **kwargs):

        self.omega_RF = omega_RF
        self.v2_RF    = v2_RF

        for key, value in kwargs.iteritems():
            setattr(self, key, value)

        self.segment_detuners = []

    
    @classmethod
    def as_general(cls, v2_RF, omega_RF, phi_0_RF):

        self = cls(v2_RF, omega_RF, phi_0_RF=phi_0_RF)
        self.generate_segment_detuner = self.generate_segment_detuner_general
                                
        return self

    def generate_segment_detuner_general(self, relative_segment_length, beta_x, beta_y):
        
        dapp_xz =  beta_x * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length
        dapp_yz = -beta_y * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length
                        
        self.segment_detuners.append(RFQTransverseSegmentGeneral(dapp_xz, dapp_yz, self.omega_RF, 
                                                                 self.phi_0_RF))

    ## @classmethod
    ## def as_on_crest(cls, v2_RF, omega_RF):

    ##     self = cls(v2_RF, omega_RF)
    ##     self.generate_segment_detuner = self.generate_segment_detuner_on_crest

    ##     return self
        
    ## def generate_segment_detuner_on_crest(self, relative_segment_length, beta_x, beta_y):

    ##     dapp_xz =  beta_x * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length
    ##     dapp_yz = -beta_y * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length

    ##     self.segment_detuners.append(RFQTransverseSegmentOnCrest(dapp_xz, dapp_yz, self.omega_RF))


    ## @classmethod
    ## def as_off_crest(cls, v2_RF, omega_RF):

    ##     self = cls(v2_RF, omega_RF)
    ##     self.generate_segment_detuner = self.generate_segment_detuner_off_crest
                
    ##     return self


    ## def generate_segment_detuner_off_crest(self, relative_segment_length, beta_x, beta_y):

    ##     dapp_xz =  beta_x * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length
    ##     dapp_yz = -beta_y * self.v2_RF * e / (self.omega_RF * 2. * np.pi) * relative_segment_length

    ##     self.segment_detuners.append(RFQTransverseSegmentOffCrest(dapp_xz, dapp_yz, self.omega_RF))


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

        cyrfq.rfq_transverse_kicks(beam.x, beam.xp, beam.y, beam.yp, beam.z, beam.p0, beam.beta,
                                   self.omega_RF, self.v2_RF, self.phi_0_RF)


'''
Longitudinal effect of RFQ. Always acting as a localized kick (applied once per turn).
'''
class RFQLongitudinal(object):

    def __init__(self, v2_RF, omega_RF, phi_0_RF):

        self.v2_RF    = v2_RF
        self.omega_RF = omega_RF
        self.phi_0_RF = phi_0_RF

        
    def track(self, beam):

        cyrfq.rfq_longitudinal_kick(beam.x, beam.y, beam.z, beam.dp, beam.beta,
                                    beam.p0, self.omega_RF, self.phi_0_RF, self.v2_RF)
