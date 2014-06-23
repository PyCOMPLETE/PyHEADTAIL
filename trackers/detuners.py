'''
@author Michael Schenk
@date June 2014
@brief Factory of detuners
@copyright CERN
'''
from __future__ import division
from abc import ABCMeta, abstractmethod

from scipy.constants import e, c
import numpy as np


class Detuner(object):
    """
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def detune(self, beam):
        """Calculates the detune of the corresponding detuner element.
        """
        pass


class Sextupole(Detuner):

    def __init__(self, dQp_x, dQp_y):
        self.dQp_x = dQp_x
        self.dQp_y = dQp_y


    def detune(self, beam):

        # W/o factor 2 np.pi. See TransverseSegmentMap.track().
        dphi_x = self.dQp_x * beam.dp
        dphi_y = self.dQp_y * beam.dp

        return dphi_x, dphi_y
    
            
class Octupole(Detuner):

    def __init__(self, beta_x, beta_y, dapp_x, dapp_y, dapp_xy):

        # dapp_xy == dapp_yx.
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

        
class RFQ(Detuner):
    pass



# CONVENIENCE FUNCTIONS
def get_LHC_octupole_parameters_from_currents(i_f, i_d):
    # Calculate app_x, app_y, app_xy = app_yx on the basis of formulae (3.6) in
    # 'THE LHC TRANSVERSE COUPLED-BUNCH INSTABILITY', N. Mounet, 2012 from
    # LHC octupole currents i_f, i_d [A].
    app_x  = 7000.*(267065.*i_f/550. - 7856.*i_d/550.)
    app_y  = 7000.*(9789.*i_f/550. - 277203.*i_d/550.)
    app_xy = 7000.*(-102261.*i_f/550. + 93331.*i_d/550.)

    convert_to_SI_units = e/(1e-9*c)
    app_x  *= convert_to_SI_units
    app_y  *= convert_to_SI_units
    app_xy *= convert_to_SI_units

    app_yx = app_xy

    return app_x, app_xy, app_y, app_yx
