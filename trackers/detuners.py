'''
@author Michael Schenk
@date June, 23rd 2014
@brief Factory of detuners
@copyright CERN
'''
from __future__ import division
from abc import ABCMeta, abstractmethod
from scipy.constants import e, c
import numpy as np


cos = np.cos

class Detuner(object):
    """
    
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def detune(self, beam):
        """
        Calculates the detune caused by the corresponding detuner element.
        """
        pass


"""
Some commonly used detuner elements. To be extended.
"""
class SextupoleSegment(Detuner):

    def __init__(self, dQp_x, dQp_y):

        self.dQp_x = dQp_x
        self.dQp_y = dQp_y


    def detune(self, beam):

        # W/o factor 2 np.pi. See TransverseSegmentMap.track().
        dphi_x = self.dQp_x * beam.dp
        dphi_y = self.dQp_y * beam.dp

        return dphi_x, dphi_y
    
            
class OctupoleSegment(Detuner):

    def __init__(self, beta_x, beta_y, dapp_x, dapp_y, dapp_xy):

        # For octupole magnets, dapp_xy == dapp_yx.
        self.beta_x  = beta_x
        self.beta_y  = beta_y
        self.dapp_x  = dapp_x
        self.dapp_y  = dapp_y
        self.dapp_xy = dapp_xy

        
    def detune(self, beam):

        Jx = (beam.x ** 2 + (self.beta_x * beam.xp) ** 2) / (2. * self.beta_x)
        Jy = (beam.y ** 2 + (self.beta_y * beam.yp) ** 2) / (2. * self.beta_y)

        # W/o factor 2 np.pi. See TransverseSegmentMap.track().
        dphi_x = self.dapp_x/beam.p0 * Jx + self.dapp_xy/beam.p0 * Jy
        dphi_y = self.dapp_y/beam.p0 * Jy + self.dapp_xy/beam.p0 * Jx

        return dphi_x, dphi_y

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
Collection classes for each class of detuner. These are the classes instantiated explicitly by the user.
They use 1-turn integrated values as input and instantiate detuners for each segment in 's' with a
detuning proportional to the segment length.
"""
class Octupole(object):

    def __init__(self, s, beta_x, beta_y, app_x, app_y, app_xy):

        self.s = s
        self.beta_x = beta_x
        self.beta_y = beta_y

        scale_to_segment = np.diff(s) / s[-1]
        self.dapp_x  = app_x * scale_to_segment
        self.dapp_y  = app_y * scale_to_segment
        self.dapp_xy = app_xy * scale_to_segment          # For octupole magnets, app_xy == app_yx.

        self._generate_segment_detuners()
        
        
    @classmethod
    def from_currents_LHC(cls, s, beta_x, beta_y, i_focusing, i_defocusing):
        """
        Calculate app_x, app_y, app_xy == app_yx on the basis of formulae (3.6) in
        'THE LHC TRANSVERSE COUPLED-BUNCH INSTABILITY' (EPFL PhD Thesis), N. Mounet, 2012
        from LHC octupole currents i_focusing, i_defocusing [A].

        Measurement values (hard-coded numbers) were obtained before LS1.
        """
        i_max = 550.  # [A]
        E_max = 7000. # [GeV]
        
        app_x  = E_max * (267065. * i_focusing / i_max - 7856. * i_defocusing / i_max)
        app_y  = E_max * (9789. * i_focusing / i_max - 277203. * i_defocusing / i_max)
        app_xy = E_max * (-102261. * i_focusing / i_max + 93331. * i_defocusing / i_max)

        convert_to_SI_units = e/(1e-9*c)
        app_x  *= convert_to_SI_units
        app_y  *= convert_to_SI_units
        app_xy *= convert_to_SI_units
    
        return cls(s, beta_x, beta_y, app_x, app_y, app_xy)

                
    def _generate_segment_detuners(self):

        segment_detuners = []

        n_segments = len(self.s) - 1
        for seg in range(n_segments):
            segment_detuner = OctupoleSegment(self.beta_x[seg], self.beta_y[seg],
                                              self.dapp_x[seg], self.dapp_y[seg], self.dapp_xy[seg])
            segment_detuners.append(segment_detuner)

        self.segment_detuners = segment_detuners

        
    def __len__(self):

        return len(self.segment_detuners)

    
    def __getitem__(self, key):

        return self.segment_detuners[key]

    
class Sextupole(object):

    def __init__(self, s, Qp_x, Qp_y):

        self.s = s

        scale_to_segment = np.diff(s) / s[-1]
        self.dQp_x = Qp_x * scale_to_segment
        self.dQp_y = Qp_y * scale_to_segment

        self._generate_segment_detuners()
        
        
    def _generate_segment_detuners(self):

        segment_detuners = []

        n_segments = len(self.s) - 1
        for seg in range(n_segments):
            segment_detuner = SextupoleSegment(self.dQp_x[seg], self.dQp_y[seg])
            segment_detuners.append(segment_detuner)

        self.segment_detuners = segment_detuners

        
    def __len__(self):

        return len(self.segment_detuners)

    
    def __getitem__(self, key):

        return self.segment_detuners[key]


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


    ## def _generate_segment_detuners(self):
    ##     segment_detuners = []
    ##     n_segments = len(self.s) - 1
    ##     for seg in range(n_segments):
    ##         segment_detuner = RFQSegment(self.dapp_xz[seg], self.dapp_yz[seg], self.omega_RF)
    ##         segment_detuners.append(segment_detuner)

    ##     self.segment_detuners = segment_detuners
        
    def __len__(self):

        return len(self.segment_detuners)

    
    def __getitem__(self, key):

        return self.segment_detuners[key]

