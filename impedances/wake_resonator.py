from __future__ import division
'''
@class Wakefields
@author Kevin Li & Hannes Bartosik
@date March 2014
@Class for creation and management of wakefields from impedance sources
@copyright CERN
'''

import numpy as np
from configuration import *
            	
class Wakefields(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass        


    def constant_wake_factor(self, bunch):
		n_particles = len(bunch.x)
		q = bunch.intensity / n_particles
		return (bunch.charge * e) ** 2 / (bunch.mass * bunch.gamma * c ** 2) * q
    
    
    def transverse_wakefield_kicks(self, plane):   	
    	assert(plane in ('x', 'y'))
    	@profile
    	def compute_apply_kicks(bunch):
			
            if plane == 'x':                
                slice_position = bunch.slices.mean_x
                dipole_wake = self.dipole_wake_x
                quadrupole_wake = self.quadrupolar_wake_x
                particle_position = bunch.x
                position_prime = bunch.xp
            if plane == 'y':
                slice_position = bunch.slices.mean_y
                dipole_wake = self.dipole_wake_y
                quadrupole_wake = self.quadrupolar_wake_y
                particle_position = bunch.y
                position_prime = bunch.yp
			
            # matrix with distances to target slice
            dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] - np.transpose([bunch.slices.dz_centers[1:-2]])

            # dipole kicks
            self.dipole_kick = np.zeros(bunch.n_slices()+3)
            self.dipole_kick[1:-2] = np.dot(bunch.slices.charge[1:-2] * slice_position[1:-2], dipole_wake(dz_to_target_slice)) * self.constant_wake_factor(bunch)
            
            # quadrupole kicks
            self.quadrupolar_wake_sum = np.zeros(bunch.n_slices()+3)
            self.quadrupolar_wake_sum[1:-2] = np.dot(bunch.slices.charge[1:-2], quadrupole_wake(dz_to_target_slice)) * self.constant_wake_factor(bunch)
            
            # apply kicks
            position_prime += self.dipole_kick[bunch.slices.in_slice] + self.quadrupolar_wake_sum[bunch.slices.in_slice] * particle_position[:]            
                 
        return compute_apply_kicks


def BB_Resonator_Circular(R_shunt, frequency, Q):
	return BB_Resonator_transverse(R_shunt, frequency, Q, Yokoya_X1=1, Yokoya_Y1=1, Yokoya_X2=0, Yokoya_Y2=0)


def BB_Resonator_ParallelPlates(R_shunt, frequency, Q):
	return BB_Resonator_transverse(R_shunt, frequency, Q, Yokoya_X1=np.pi**2/24, Yokoya_Y1=np.pi**2/12, Yokoya_X2=-np.pi**2/24, Yokoya_Y2=np.pi**2/24)


class BB_Resonator_transverse(Wakefields):
    '''
    classdocs
    '''
    def __init__(self, R_shunt, frequency, Q, Yokoya_X1=1, Yokoya_Y1=1, Yokoya_X2=0, Yokoya_Y2=0):
        '''
        Constructor
        '''
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q
        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y2 = Yokoya_Y2
        
        # Taken from Alex Chao's resonator model (2.82)
        self.omega = 2 * np.pi * frequency
        self.alpha = self.omega / (2 * Q)
        self.omegabar = np.sqrt(self.omega ** 2 - self.alpha ** 2)
        
        
    def wake_transverse(self, z):

        # Taken from definition in HEADTAIL (but the relativistic beta factor is still missing !!!)
        if self.Q != 0.5:
			wake = self.R_shunt * self.omega ** 2 / self.Q / self.omegabar * np.exp(self.alpha * z.clip(max=0) / c) * np.sin(self.omegabar * z.clip(max=0) / c)
        else:
			wake = self.R_shunt * self.omega ** 2 / self.Q * z.clip(max=0) / c * np.exp(self.alpha * z.clip(max=0) / c)			
        
        return wake
        
    
    def dipole_wake_x(self, z):
        return self.Yokoya_X1 * self.wake_transverse(z)

    def dipole_wake_y(self, z):     
        return self.Yokoya_Y1 * self.wake_transverse(z)
        
    def quadrupolar_wake_x(self, z):
        return self.Yokoya_X2 * self.wake_transverse(z)

    def quadrupolar_wake_y(self, z):     
        return self.Yokoya_Y2 * self.wake_transverse(z)
        
    def track(self, bunch):
        wakefield_kicks_x = self.transverse_wakefield_kicks('x')
        wakefield_kicks_x(bunch)
        wakefield_kicks_y = self.transverse_wakefield_kicks('y')
        wakefield_kicks_y(bunch)
