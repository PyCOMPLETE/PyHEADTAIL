"""
@author Andrea Passarelli
@date 23. February 2016
@brief Synchrotron radiation damping effect in transverse and longitudinal planes.
@copyright CERN
"""

from __future__ import division
import numpy as np
from scipy.constants import c


class Sync_rad_transverse(object):
    def __init__(self, eq_emit_x, eq_emit_y, synchdamp_x, synchdamp_y, beta_x, beta_y):
        '''
        We are assuming no alpha, no dispersion, etc.
        '''
        #TRANSVERSE
        self.tau_x  = synchdamp_x #Damping time [turns]
        self.tau_y  = synchdamp_y #Damping time [turns]
        self.epsn_x = eq_emit_x   #Equilibrium emittance [m.rad]
        self.epsn_y = eq_emit_y   #Equilibrium emittance [m.rad]
        self.beta_x = beta_x      #Beta average
        self.beta_y = beta_y      #Beta average
        
    def track(self, bunch):
        
        #TRANSVERSE
        sigma_xp = np.sqrt(self.epsn_x/self.beta_x/bunch.beta/bunch.gamma)
        sigma_yp = np.sqrt(self.epsn_y/self.beta_y/bunch.beta/bunch.gamma)
        bunch.xp -= 2*bunch.xp/self.tau_x
        bunch.xp += 2*sigma_xp*np.sqrt(1/self.tau_x)*np.random.normal(size=len(bunch.xp))
        bunch.yp -= 2*bunch.yp/self.tau_y
        bunch.yp += 2*sigma_yp*np.sqrt(1/self.tau_y)*np.random.normal(size=len(bunch.yp))
        
class Sync_rad_longitudinal(object):
    def __init__(self, eq_sig_dp, synchdamp_z, E_loss):
        '''
        We are assuming no alpha, no dispersion, etc.
        '''
        
        #LONGITUDINAL
        self.tau_z  = synchdamp_z    #Damping time [turns]
        self.E_loss = E_loss         #Energy loss [eV]
        self.sigma_dpp0  = eq_sig_dp #Equilibrium momentum spread 
        
    def track(self, bunch):
        
        #LONGITUDINAL
        bunch.dp -= 2*bunch.dp/self.tau_z
        bunch.dp += 2*self.sigma_dpp0*np.sqrt(1/self.tau_z)*np.random.normal(size=len(bunch.dp))
        bunch.dp -= (self.E_loss*np.abs(bunch.charge))/(bunch.mass*c*c*bunch.gamma)/bunch.beta**2
