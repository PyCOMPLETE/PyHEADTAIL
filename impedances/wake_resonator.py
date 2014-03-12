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
			
            #~ dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] * bunch.n_slices() - np.transpose([bunch.slices.dz_centers[1:-2]])
 #~ 
            #~ self.dipole_kick = np.zeros(bunch.n_slices()+3)
            #~ self.dipole_kick[1:-2] = sum(dipole_wake(dz_to_target_slice) * np.transpose(bunch.slices.charge[1:-2] * [slice_position[1:-2]])) * self.constant_wake_factor(bunch)
            #~ # print 'original          ', self.dipole_kick[1:-2]
            #~ self.quadrupolar_wake_sum = np.zeros(bunch.n_slices()+3)
            #~ self.quadrupolar_wake_sum[1:-2] = sum(quadrupole_wake(dz_to_target_slice) * np.transpose([bunch.slices.charge[1:-2]])) * self.constant_wake_factor(bunch)
            #~ # print 'original          ', self.quadrupolar_wake_sum[1:-2]
            
            # even faster ...
            dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] - np.transpose([bunch.slices.dz_centers[1:-2]])

            self.dipole_kick = np.zeros(bunch.n_slices()+3)
            self.dipole_kick[1:-2] = np.dot(bunch.slices.charge[1:-2] * slice_position[1:-2], dipole_wake(dz_to_target_slice)) * self.constant_wake_factor(bunch)
            # print 'new implementation', self.dipole_kick[1:-2]
            self.quadrupolar_wake_sum = np.zeros(bunch.n_slices()+3)
            self.quadrupolar_wake_sum[1:-2] = np.dot(bunch.slices.charge[1:-2], quadrupole_wake(dz_to_target_slice)) * self.constant_wake_factor(bunch)
            # print 'new implementation', self.quadrupolar_wake_sum[1:-2]
            
            #~ plt.plot(self.dipole_kick)
            #~ plt.plot(self.quadrupolar_wake_sum)
            
            ####################
            #~ # this is really slow:
            #~ for i in xrange(bunch.n_slices(), 0, -1):
                #~ position_prime[bunch.slices.index(i)] += self.dipole_kick[i] + self.quadrupolar_wake_sum[i] * particle_position[bunch.slices.index(i)]            
            ####################
            # faster version using new implementation in slices:
            position_prime += self.dipole_kick[bunch.slices.in_slice] + self.quadrupolar_wake_sum[bunch.slices.in_slice] * particle_position[:]            
                 
        return compute_apply_kicks
        
    
    def convolve_dipolar_wake(self, plane):   	
    	assert(plane in ('x', 'y'))

    	def convolve_dipolar(bunch):
			
            if plane == 'x':                
                slice_position = bunch.slices.mean_x
                wake = self.dipole_wake_x
                position_prime = bunch.xp
            if plane == 'y':
                slice_position = bunch.slices.mean_y
                wake = self.dipole_wake_y
                position_prime = bunch.yp
			
            '''						
            ######################
            # first implementation (seems to work, but slow)
            # initialize kicks with zeros
            self.kick = np.zeros(bunch.n_slices()+3)
            
            # Target
            for i in xrange(bunch.n_slices(), 0, -1):
                # Sources
                for j in xrange(bunch.n_slices(), i, -1):
                    zj = bunch.slices.dz_centers[i] - bunch.slices.dz_centers[j]
                    self.kick[i] += bunch.slices.charge[j] * slice_position[j] * wake(zj)
                
                # apply kick
                index = bunch.slices.index
                self.kick[i] *= self.constant_wake_factor(bunch)
                position_prime[index(i)] += self.kick[i]   
            #~ print self.kick
            ######################
            '''
            
            # new implementation (faster)
            dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] * bunch.n_slices() - np.transpose([bunch.slices.dz_centers[1:-2]])
            self.kick2 = np.zeros(bunch.n_slices()+3)
            self.kick2[1:-2] = sum(wake(dz_to_target_slice) * np.transpose(bunch.slices.charge[1:-2] * [slice_position[1:-2]])) * self.constant_wake_factor(bunch)
            
            for i in xrange(bunch.n_slices(), 0, -1):
                position_prime[bunch.slices.index(i)] += self.kick2[i]
            
            #~ # for comparing the results of the two implementations!
            #~ plt.plot(self.kick - self.kick2)
            #~ plt.plot(self.kick )
            #~ plt.plot(self.kick2)            
                   
        return convolve_dipolar

									
    def convolve_quadrupolar_wake(self, plane):   	
    	assert(plane in ('x', 'y'))

    	def convolve_quadrupolar(bunch):
			
            if plane == 'x':                
                wake = self.quadrupolar_wake_x
                particle_position = bunch.x
                position_prime = bunch.xp                
            if plane == 'y':
                wake = self.quadrupolar_wake_y
                particle_position = bunch.y                
                position_prime = bunch.yp

            
            ######################
            # first implementation (seems to work, but slow)            	
            # initialize wake_sum with zeros
            self.quadrupolar_wake_sum = np.zeros(bunch.n_slices()+3)
            # Target
            for i in xrange(bunch.n_slices(), 0, -1):
                # Sources
                for j in xrange(bunch.n_slices(), i, -1):
                    zj = bunch.slices.dz_centers[i] - bunch.slices.dz_centers[j]
                    self.quadrupolar_wake_sum[i] += bunch.slices.charge[j] * wake(zj)

                # calculate and apply kick
                index = bunch.slices.index
                position_prime[index(i)] += self.constant_wake_factor(bunch) * self.quadrupolar_wake_sum[i] * particle_position[index(i)]
            ######################
            
            
            # new implementation (faster)
            dz_to_target_slice = [bunch.slices.dz_centers[1:-2]] * bunch.n_slices() - np.transpose([bunch.slices.dz_centers[1:-2]])
            self.quadrupolar_wake_sum2 = np.zeros(bunch.n_slices()+3)
            self.quadrupolar_wake_sum2[1:-2] = sum(wake(dz_to_target_slice) * np.transpose([bunch.slices.charge[1:-2]])) * self.constant_wake_factor(bunch)
            
            for i in xrange(bunch.n_slices(), 0, -1):
                position_prime[bunch.slices.index(i)] += self.quadrupolar_wake_sum2[i] * particle_position[bunch.slices.index(i)]

            #~ # for comparing the results of the two implementations!                
            #~ plt.plot(self.quadrupolar_wake_sum * self.constant_wake_factor(bunch) - self.quadrupolar_wake_sum2)
            #~ plt.plot(self.quadrupolar_wake_sum * self.constant_wake_factor(bunch))
            #~ plt.plot(self.quadrupolar_wake_sum2)

        return convolve_quadrupolar


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
        #~ if self.Yokoya_X1:
            #~ convolve_dipolar_x = self.convolve_dipolar_wake('x')
            #~ convolve_dipolar_x(bunch)
        #~ if self.Yokoya_Y1:		
            #~ convolve_dipolar_y = self.convolve_dipolar_wake('y')
            #~ convolve_dipolar_y(bunch)       
        #~ if self.Yokoya_X2:
            #~ convolve_quadrupolar_x = self.convolve_quadrupolar_wake('x')
            #~ convolve_quadrupolar_x(bunch)        
        #~ if self.Yokoya_Y2:
            #~ convolve_quadrupolar_y = self.convolve_quadrupolar_wake('y')
            #~ convolve_quadrupolar_y(bunch)       
        wakefield_kicks_x = self.transverse_wakefield_kicks('x')
        wakefield_kicks_x(bunch)
        wakefield_kicks_y = self.transverse_wakefield_kicks('y')
        wakefield_kicks_y(bunch)






'''

np.digitize(bunch.dz,bunch.slices.dz_bins[1:-1])  


'''			





''' to be developed ...

    #~ def wake_resistive(self, z):
#~ 
        #~ # Taken from Alex Chao's resisitve wall (2.53)
        #~ sigma = 5.4e17 # copper conductivity in CGS [1/s]
        #~ piper = 2e-3
        #~ length = 6911
#~ 
        #~ wake = -1 / (2 * np.pi * eps0) / (np.pi * piper ** 3) * np.sqrt(c / sigma) * 1 / np.sqrt(-z) * length
#~ 
        #~ return wake
        
        
        
            def wake_longitudinal(self, z):

        # Taken from Alex Chao's resonator model (2.82)
        wake = 2 * self.alpha * self.R_shunt * np.exp(self.alpha * z / c) * (np.cos(self.omegabar * z / c) + self.alpha / self.omegabar * sin(self.omegabar * z / c))

        return wake


               # Target
        for i in xrange(n_slices, 0, -1):
			
            # Beam loading of self-slice
            if self.plane == 'z':
                ni = bunch.slices.charge[i]
                #~ self.kick[i] = self.constantfactor(bunch) * np_i * alpha * R_shunt
                self.kick[i] = self.constantfactor(bunch) * ni * alpha * R_shunt
''' 
	
