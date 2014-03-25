'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


from beams.slices import *
from beams.matching import match_transverse, match_longitudinal, unmatched_inbucket
from scipy.constants import c, e
import sys
import h5py


def bunch_matched_and_sliced(n_macroparticles, charge, energy, intensity, mass,
                             epsn_x, epsn_y, ltm, bunch_length, bucket, matching,
                             n_slices, nsigmaz, slicemode='cspace'):

    # bunch = Bunch.from_empty(1e3, charge, energy, intensity, mass)
    # x, xp, y, yp, dz, dp = random.gsl_quasirandom(bunch)  
    bunch = Bunch.from_gaussian(n_macroparticles, charge, energy, intensity, mass)
    bunch.match_transverse(epsn_x, epsn_y, ltm)
    bunch.match_longitudinal(bunch_length, bucket, matching)
    bunch.set_slices(Slices(n_slices, nsigmaz, slicemode))
    bunch.update_slices()

    return bunch

def bunch_unmatched_inbucket_sliced(n_macroparticles, charge, energy, intensity, mass,
                             epsn_x, epsn_y, ltm, sigma_dz, sigma_dp, bucket,
                             n_slices, nsigmaz, slicemode='cspace'):
    bunch = Bunch.from_gaussian(n_macroparticles, charge, energy, intensity, mass)
    bunch.match_transverse(epsn_x, epsn_y, ltm)
    bunch.unmatched_inbucket(sigma_dz, sigma_dp, bucket)
    bunch.set_slices(Slices(n_slices, nsigmaz, slicemode))
    bunch.update_slices()
    
    return bunch

def bunch_from_file(filename, step, charge, energy, intensity, mass,
                             n_slices, nsigmaz, slicemode='cspace'):

    bunch = Bunch.from_h5file(filename, step, charge, energy, intensity, mass)
    bunch.set_slices(Slices(n_slices, nsigmaz, slicemode))
    bunch.update_slices()

    return bunch

class Bunch(object):
    '''
    Fundamental entity for collective beam dynamics simulations
    '''

    def __init__(self, x, xp, y, yp, dz, dp):
        '''
        Most minimalistic constructor - pure python name binding
        '''
        assert(len(x) == len(xp) == len(y) == len(yp) == len(dz) == len(dp))

        self.x = x
        self.xp = xp
        self.y = y
        self.yp = yp
        self.dz = dz
        self.dp = dp

    @classmethod
    def from_copy(cls, x, xp, y, yp, dz, dp, identity):

        x = np.copy(x)
        xp = np.copy(xp)
        y = np.copy(y)
        yp = np.copy(yp)
        dz = np.copy(dz)
        dp = np.copy(dp)
        
        self = cls(x, xp, y, yp, dz, dp)
        
        self.n_macroparticles = len(x)
        self.n_macroparticles_lost = 0
        self.identity = identity

        return self

    @classmethod
    def from_empty(cls, n_macroparticles, charge, energy, intensity, mass):

        x = np.zeros(n_macroparticles)
        xp = np.zeros(n_macroparticles)
        y = np.zeros(n_macroparticles)
        yp = np.zeros(n_macroparticles)
        dz = np.zeros(n_macroparticles)
        dp = np.zeros(n_macroparticles)

        self = cls(x, xp, y, yp, dz, dp)

        self.set_scalar_quantities(charge, energy, intensity, mass)

        return self

    @classmethod
    def from_h5file(cls, filename, step, charge, energy, intensity, mass):
		# TO DO
        particles = h5py.File(filename + '.h5part', 'r')
		
        x = np.array(particles['Step#' + str(step)]['x']).astype(np.double)
        xp = np.array(particles['Step#' + str(step)]['xp']).astype(np.double)
        y = np.array(particles['Step#' + str(step)]['y']).astype(np.double)
        yp = np.array(particles['Step#' + str(step)]['yp']).astype(np.double)
        dz = np.array(particles['Step#' + str(step)]['dz']).astype(np.double)
        dp = np.array(particles['Step#' + str(step)]['dp']).astype(np.double)
        
        
        self = cls(x, xp, y, yp, dz, dp)
		
        self.n_macroparticles = len(x)
        self.n_macroparticles_lost = 0
        self.identity = np.array(particles['Step#' + str(step)]['identity'])
        self.set_scalar_quantities(charge, energy, intensity, mass)
        
        return self
        
    @classmethod
    def from_gaussian(cls, n_macroparticles, charge, energy, intensity, mass):

        x = np.random.randn(n_macroparticles)
        xp = np.random.randn(n_macroparticles)
        y = np.random.randn(n_macroparticles)
        yp = np.random.randn(n_macroparticles)
        dz = np.random.randn(n_macroparticles)
        dp = np.random.randn(n_macroparticles)

        self = cls(x, xp, y, yp, dz, dp)
		
        self.n_macroparticles = len(x)
        self.n_macroparticles_lost = 0
        self.identity = np.arange(n_macroparticles) + 1
        self.set_scalar_quantities(charge, energy, intensity, mass)

        return self

    @classmethod
    def from_uniform(cls, n_macroparticles, charge, energy, intensity, mass):

        x = np.random.rand(n_macroparticles) * 2 - 1
        xp = np.random.rand(n_macroparticles) * 2 - 1
        y = np.random.rand(n_macroparticles) * 2 - 1
        yp = np.random.rand(n_macroparticles) * 2 - 1
        dz = np.random.rand(n_macroparticles) * 2 - 1
        dp = np.random.rand(n_macroparticles) * 2 - 1

        self = cls(x, xp, y, yp, dz, dp)

        self.n_macroparticles = len(x)
        self.n_macroparticles_lost = 0
        self.identity = np.arange(n_macroparticles) + 1
        self.set_scalar_quantities(charge, energy, intensity, mass)

        return self

    def set_scalar_quantities(self, charge, energy, intensity, mass):

        self.charge = charge
        self.gamma = energy * 1e9 * charge * e / (mass * c ** 2) # + 1 only for below PS
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.intensity = intensity
        self.mass = mass
        self.p0 = mass * self.gamma * self.beta * c

    def match_transverse(self, epsn_x, epsn_y, ltm):

        match_transverse(epsn_x, epsn_y, ltm)(self)

    def match_longitudinal(self, length, bucket=None, matching=None):

        match_longitudinal(length, bucket, matching)(self)

    def unmatched_inbucket(self, sigma_dz, sigma_dp, bucket=None):

        unmatched_inbucket(self, sigma_dz, sigma_dp, bucket)

    #~ @profile
    def compute_statistics(self):

        if not hasattr(self, 'slices'):
            print "*** WARNING: bunch not yet sliced! Aborting..."
            sys.exit(-1)
     
        # determine the start and end indices of each slices 
        i1 = np.append(np.cumsum(self.slices.n_macroparticles[:-2]), np.cumsum(self.slices.n_macroparticles[-2:]))
        i0 = np.zeros(len(i1), dtype='int')
        i0[1:] = i1[:-1]
        i0[-2] = 0
 
        for i in xrange(self.slices.n_slices + 4):
			x = self.x[i0[i]:i1[i]]
			xp = self.xp[i0[i]:i1[i]]
			y = self.y[i0[i]:i1[i]]
			yp = self.yp[i0[i]:i1[i]]
			dz = self.dz[i0[i]:i1[i]]
			dp = self.dp[i0[i]:i1[i]]
			
			self.slices.mean_x[i] = cp.mean(x)
			self.slices.mean_xp[i] = cp.mean(xp)
			self.slices.mean_y[i] = cp.mean(y)
			self.slices.mean_yp[i] = cp.mean(yp)
			self.slices.mean_dz[i] = cp.mean(dz)
			self.slices.mean_dp[i] = cp.mean(dp)
			 
			self.slices.sigma_x[i] = cp.std(x)   
			self.slices.sigma_y[i] = cp.std(y)
			self.slices.sigma_dz[i] = cp.std(dz)
			self.slices.sigma_dp[i] = cp.std(dp)
			
			self.slices.epsn_x[i] = cp.emittance(x, xp) * self.gamma * self.beta * 1e6
			self.slices.epsn_y[i] = cp.emittance(y, yp) * self.gamma * self.beta * 1e6
			self.slices.epsn_z[i] = 4 * np.pi \
								  * self.slices.sigma_dz[i] * self.slices.sigma_dp[i] \
								  * self.mass * self.gamma * self.beta * c / e
								  							  
    def set_slices(self, slices):

        self.slices = slices

    def update_slices(self):

        assert(hasattr(self, 'slices'))

        if self.slices.slicemode == 'ccharge':
            self.slices.slice_constant_charge(self, self.slices.nsigmaz)
        elif self.slices.slicemode == 'cspace':
            self.slices.slice_constant_space(self, self.slices.nsigmaz)


    #~ @profile
    def sort_particles(self):
		# update the number of lost particles
        self.n_macroparticles_lost = (self.n_macroparticles - np.count_nonzero(self.identity))
        
        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)        
        if self.n_macroparticles_lost:
            dz_argsorted = np.lexsort((self.dz, -np.sign(self.identity))) # place lost particles at the end of the array
        else:
            dz_argsorted = np.argsort(self.dz)    

        self.x = self.x[dz_argsorted]
        self.xp = self.xp[dz_argsorted]
        self.y = self.y[dz_argsorted]
        self.yp = self.yp[dz_argsorted]
        self.dz = self.dz[dz_argsorted]
        self.dp = self.dp[dz_argsorted]
        self.identity = self.identity[dz_argsorted]


    def set_in_slice(self, index_after_bin_edges):           
        self.in_slice = (self.slices.n_slices + 3) * np.ones(self.n_macroparticles, dtype=np.int)
        for i in xrange(self.slices.n_slices + 2):
            self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i        
