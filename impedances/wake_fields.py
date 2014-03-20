from __future__ import division
'''
@class Wakefields
@author Kevin Li & Hannes Bartosik
@date March 2014
@Class for creation and management of wakefields from impedance sources
@copyright CERN
'''


import numpy as np


import pylab as plt
from scipy.constants import c, e
from scipy.interpolate import interp1d
from scipy import interpolate

            	
class Wakefields(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass        

    def wake_factor(self, bunch):
		n_particles = len(bunch.x)
		q = bunch.intensity / n_particles
		return -(bunch.charge * e) ** 2 / (bunch.mass * bunch.gamma * (bunch.beta * c) ** 2) * q

    #~ @profile    
    def transverse_wakefield_kicks(self, plane):   	
    	assert(plane in ('x', 'y'))
    	#~ @profile
    	def compute_apply_kicks(bunch):
			
            if plane == 'x':                
                slice_position = bunch.slices.mean_x
                dipole_wake = self.dipole_wake_x
                quadrupole_wake = self.quadrupole_wake_x
                particle_position = bunch.x
                position_prime = bunch.xp
            if plane == 'y':
                slice_position = bunch.slices.mean_y
                dipole_wake = self.dipole_wake_y
                quadrupole_wake = self.quadrupole_wake_y
                particle_position = bunch.y
                position_prime = bunch.yp
            if plane == 'xy':
                slice_position = bunch.slices.mean_x
                dipole_wake = self.dipole_wake_xy
                quadrupole_wake = self.quadrupole_wake_xy
                particle_position = bunch.x
                position_prime = bunch.yp     
            if plane == 'yx':
                slice_position = bunch.slices.mean_y
                dipole_wake = self.dipole_wake_yx
                quadrupole_wake = self.quadrupole_wake_yx
                particle_position = bunch.y
                position_prime = bunch.xp                
			
            # matrix with distances to target slice
            dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] - np.transpose([bunch.slices.dz_centers[1:-2]])

            # dipole kicks
            self.dipole_kick = np.zeros(bunch.slices.n_slices+4)
            self.dipole_kick[1:-3] = np.dot(bunch.slices.charge[1:-3] * slice_position[1:-3], dipole_wake(bunch, dz_to_target_slice)) * self.wake_factor(bunch)

            # quadrupole kicks
            self.quadrupolar_wake_sum = np.zeros(bunch.slices.n_slices+4)
            self.quadrupolar_wake_sum[1:-3] = np.dot(bunch.slices.charge[1:-3], quadrupole_wake(bunch, dz_to_target_slice)) * self.wake_factor(bunch)

            # apply kicks
            position_prime += self.dipole_kick[bunch.in_slice] + self.quadrupolar_wake_sum[bunch.in_slice] * particle_position[:]            
                 
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
        self.omegabar = np.sqrt(np.abs(self.omega ** 2 - self.alpha ** 2))
           
    def wake_transverse(self, bunch, z):

        Rs = self.R_shunt
        omega = self.omega
        Q = self.Q
        omegabar = self.omegabar
        alpha = self.alpha
        beta_r = bunch.beta
        
        # Taken from definition in HEADTAIL
        if self.Q > 0.5:
			wake =  Rs * omega ** 2 / Q / omegabar * np.exp(alpha * z.clip(max=0) / c / beta_r) * \
                    np.sin(omegabar * z.clip(max=0) / c / beta_r)
        elif self.Q == 0.5:			
			wake =  Rs * omega ** 2 / Q * z.clip(max=0) / c / beta_r * \
                    np.exp(alpha * z.clip(max=0) / c / beta_r)
        else:			
			wake =  Rs * omega ** 2 / Q / omegabar * np.exp(alpha * z.clip(max=0) / c / beta_r) * \
                    np.sinh(omegabar * z.clip(max=0) / c / beta_r)                           
        return wake
        
    
    def dipole_wake_x(self, bunch, z):
        return self.Yokoya_X1 * self.wake_transverse(bunch, z)

    def dipole_wake_y(self, bunch, z):     
        return self.Yokoya_Y1 * self.wake_transverse(bunch, z)
        
    def quadrupole_wake_x(self, bunch, z):
        return self.Yokoya_X2 * self.wake_transverse(bunch, z)

    def quadrupole_wake_y(self, bunch, z):     
        return self.Yokoya_Y2 * self.wake_transverse(bunch, z)
        
    def track(self, bunch): 
        bunch.compute_statistics()    
        wakefield_kicks_x = self.transverse_wakefield_kicks('x')
        wakefield_kicks_x(bunch)
        wakefield_kicks_y = self.transverse_wakefield_kicks('y')
        wakefield_kicks_y(bunch)


class Transverse_wake_from_table(Wakefields):
    '''
    classdocs
    '''
    def __init__(self, wake_file):       
        self.wake_table = np.loadtxt(wake_file, delimiter="\t")
        # insert zeros at origin if wake functions at (or below) zero not provided
        if self.wake_table[0,0] > 0:
            self.wake_table = np.vstack((np.zeros(self.wake_table[0].shape), self.wake_table))
        if self.wake_table[0].shape == 5:
            print 'Imported dipole and quadrupole wakes (x and y)'
        elif self.wake_table[0].shape == 7:
            print 'Imported dipole and quadrupole wakes (x, y and xy/yx)'
    
    #~ @profile
    def wake_transverse(self, n_column_wake_table, bunch, z):
        delta_t = 1.e-9 * self.wake_table[:,0]
        wake = self.wake_table[:,n_column_wake_table] * 1.e15
        wake_interpolated = interp1d(delta_t, wake, kind='linear', bounds_error=False, fill_value=0) 
        return wake_interpolated(-z / c / bunch.beta)   # the minus is important as z will be negative along the bunch!!       
                                    
    def dipole_wake_x(self, bunch, z):
        return self.wake_transverse(1, bunch, z)
        
    def dipole_wake_y(self, bunch, z):
        return self.wake_transverse(2, bunch, z)

    def quadrupole_wake_x(self, bunch, z):
        return self.wake_transverse(3, bunch, z)

    def quadrupole_wake_y(self, bunch, z):
        return self.wake_transverse(4, bunch, z) 
                       
    def dipole_wake_xy(self, bunch, z):
        return self.wake_transverse(5, bunch, z)
        
    def dipole_wake_yx(self, bunch, z):
        return self.wake_transverse(5, bunch, z)

    def quadrupole_wake_xy(self, bunch, z):
        return self.wake_transverse(6, bunch, z)

    def quadrupole_wake_yx(self, bunch, z):
        return self.wake_transverse(6, bunch, z)

    def track(self, bunch):
        bunch.compute_statistics()    
        wakefield_kicks_x = self.transverse_wakefield_kicks('x')
        wakefield_kicks_x(bunch)
        wakefield_kicks_y = self.transverse_wakefield_kicks('y')
        wakefield_kicks_y(bunch)            
        if self.wake_table[0].shape == 7:
            wakefield_kicks_xy = self.transverse_wakefield_kicks('xy')
            wakefield_kicks_xy(bunch)
            wakefield_kicks_yx = self.transverse_wakefield_kicks('yx')
            wakefield_kicks_yx(bunch)        
