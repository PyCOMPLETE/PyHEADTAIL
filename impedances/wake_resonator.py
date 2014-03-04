from __future__ import division
'''
@class Wakefields
@author Kevin Li
@date March 2014
@brief Class for creation and management of wakefields from impedance sources
@copyright CERN
'''


import numpy as np


from configuration import *
import pylab as plt

def test(R_shunt, frequency, Q, plane='x'):
	return WakeResonator(R_shunt, frequency, Q, plane='x')
	
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
									
            if not hasattr(self, 'kick'):
                self.kick = np.zeros(bunch.n_slices()+3)
            else:
                #~ plt.clf()
                #~ plt.plot(self.kick)
                #~ plt.gca().set_ylim(-2e-4, 2e-4)
                #~ plt.draw()
                self.kick = np.zeros(bunch.n_slices()+3)

            # Target
            for i in xrange(bunch.n_slices(), 0, -1):
                # Sources
                for j in xrange(bunch.n_slices(), i, -1):
                    zj = bunch.slices.dz_centers[i] - bunch.slices.dz_centers[j]
                    self.kick[i] += bunch.slices.charge[j] * slice_position[j] * wake(zj)
                
                # apply kick
                self.kick[i] *= self.constant_wake_factor(bunch)
                index = bunch.slices.index(i)
                position_prime[bunch.slices.index(i)] += self.kick[i]
                
            #~ print self.kick
                
        return convolve_dipolar
									
    def convolve_quadrupolar_wake(self, plane):   	
    	assert(plane in ('x', 'y'))


		#~ from headtail: kick_x += wakefac*npr[mmain]*0.41/1.23*(xs[mmain]-xpr[ind])*wake_func(zi);
    	def convolve_quadrupolar(bunch):
			
            if plane == 'x':                
                wake = self.quadrupole_wake_x
                position_prime = bunch.xp
            if plane == 'y':
                wake = self.quadrupole_wake_y
                position_prime = bunch.yp
									
            self.kick = np.zeros(bunch.n_slices()+3)

            # Target
            for i in xrange(bunch.n_slices(), 0, -1):
                # Sources
                for j in xrange(bunch.n_slices(), i, -1):
                    zj = bunch.slices.dz_centers[i] - bunch.slices.dz_centers[j]
                    #~ self.kick[i] += bunch.slices.charge[j] * slice_position[j] * wake(zj)
                #~ 
                #~ # apply kick
                #~ self.kick[i] *= self.constant_wake_factor(bunch)
                #~ index = bunch.slices.index(i)
                #~ position_prime[bunch.slices.index(i)] += self.kick[i]
 
        return convolve_quadrupolar
    '''
    def convolve(self, bunch):

              

            
            # Sources
            for j in xrange(n_slices, i, -1):
                nj = bunch.slices.charge[j]

                # bunch.slice(n_slices, nsigmaz=None, mode='cspace')
                #~ pdf, bins, patches = plt.hist(bunch.dz, n_slices)
                #~ for i, ch in enumerate(pdf):
                    #~ print ch, bunch.slices.charge[i+1]
                #~ print len(pdf), len(bunch.slices.charge), sum(bunch.slices.charge)
                #~ print len(bunch.slices.dz_bins), len(bunch.slices.dz_centers), len(bunch.slices.charge)
                #~ plt.stem(bunch.slices.dz_centers[:-1], bunch.slices.charge[:-1], linefmt='g', markerfmt='go')
                #~ [plt.axvline(i, c='y') for i in bunch.slices.dz_bins]
                #~ plt.show()

                # zj = 1 / 2. * (bunch.zbins[i] - bunch.zbins[j] + bunch.zbins[i + 1] - bunch.zbins[j + 1])
                zj = bunch.slices.dz_centers[i] - bunch.slices.dz_centers[j]

                if self.plane == 'x':
                    self.kick[i] += nj * bunch.slices.mean_x[j] * self.wake_transverse(zj)
                elif self.plane == 'y':
                    self.kick[i] += nj * bunch.slices.mean_y[j] * self.wake_transverse(zj)
                elif self.plane == 'z':
                    self.kick[i] += nj * self.wake_longitudinal(zj);

		
    def apply_kick(self, bunch):

        self.kick *= self.constantfactor(bunch)
        n_slices = bunch.n_slices()
        
        for i in xrange(n_slices, 0, -1):
            index = bunch.slices.index(i)
            if self.plane == 'x':
                bunch.xp[index] += self.kick[i]
            elif self.plane == 'y':
                bunch.yp[index] += self.kick[i]
            elif self.plane == 'z':
                bunch.dp[index] += self.kick[i]

    def track(self, bunch):

        self.convolve(bunch)
        self.apply_kick(bunch)
    '''    


def BB_Resonator_Circular(R_shunt, frequency, Q):
	return BB_Resonator_transverse(R_shunt, frequency, Q, YokoyaX1=1, YokoyaY1=1, YokoyaX2=0, YokoyaY2=0)


class BB_Resonator_transverse(Wakefields):
    '''
    classdocs
    '''
    def __init__(self, R_shunt, frequency, Q, YokoyaX1=1, YokoyaY1=1, YokoyaX2=0, YokoyaY2=0):
        '''
        Constructor
        '''
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q
        self.YokoyaX1 = YokoyaX1
        self.YokoyaY1 = YokoyaY1
        self.YokoyaX2 = YokoyaX2
        self.YokoyaY2 = YokoyaY2
        
        # Taken from Alex Chao's resonator model (2.82)
        self.omega = 2 * np.pi * frequency
        self.alpha = self.omega / (2 * Q)
        self.omegabar = np.sqrt(self.omega ** 2 - self.alpha ** 2)
#~ 
        #~ assert(plane in ('x', 'y', 'z'))
        
    def wake_transverse(self, z):

        #~ # Taken from Alex Chao's resonator model (2.82)
        #~ wake = self.omega ** 2 / (self.Q * self.omegabar) * self.R_shunt * np.exp(self.alpha * z / c) * np.sin(self.omegabar * z / c)
        
        # Taken from Alex Chao's resonator model (2.88), HB
        wake = c* self.R_shunt * self.omega / (self.Q * self.omegabar) * np.exp(self.alpha * z / c) * np.sin(self.omegabar * z / c)

        return wake

    def dipole_wake_x(self, z):
        return self.wake_transverse(z)

    def dipole_wake_y(self, z):     
        return self.wake_transverse(z)
        
    def quadrupole_wake_x(self, z):
        return self.wake_transverse(z)

    def quadrupole_wake_y(self, z):     
        return self.wake_transverse(z)
        
    def track(self, bunch):
        if self.YokoyaX1:
            convolve_dipolar_x = self.convolve_dipolar_wake('x')
            convolve_dipolar_x(bunch)
        #~ if self.YokoyaY1:		
            #~ convolve_dipolar_y = self.convolve_dipolar_wake('y')
            #~ convolve_dipolar_y(bunch)       
        #~ if self.YokoyaX2:
            #~ self.convolve_quadrupolar_wake(bunch, 'x')        
        #~ if self.YokoyaY2:
            #~ self.convolve_quadrupolar_wake(bunch, 'y')       


'''
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
	
