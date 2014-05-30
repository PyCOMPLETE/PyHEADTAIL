'''
Created on 06.01.2014

@author: Kevin Li, Adrian Oeftiger
'''


import numpy as np
import match

import copy, h5py, sys
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi

# from slices import *
# from solvers.poissonfft import *


# re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
# rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p


class PDF(object):
    '''
    Particle distribution functions (probability density functions); 
    this class provides a standard set of normalized 6d phase space distributions.
    '''

    def __init__(self, n_macroparticles):
        '''
        Allocate the memory to store the 6d phase space 
        (x, x', y, y', z, \Delta p) where \Delta p = p - p0
        for n_macroparticles macroparticles.
        '''
        self.x  = np.zeros(n_macroparticles)
        self.xp = np.zeros(n_macroparticles)
        self.y  = np.zeros(n_macroparticles)
        self.yp = np.zeros(n_macroparticles)
        self.z  = np.zeros(n_macroparticles)
        self.Deltap = np.zeros(n_macroparticles)

    def _create_uniform(self, n_macroparticles):
        '''
        Create a normalized uniform 6d phase space distribution for 
        n_macroparticles macroparticles from -1 to +1 in all dimensions:
        (x, x', y, y', z, \Delta p) where \Delta p = p - p0
        '''
        self.x = 2 * np.random.rand(n_macroparticles) - 1
        self.xp = 2 * np.random.rand(n_macroparticles) - 1
        self.y = 2 * np.random.rand(n_macroparticles) - 1
        self.yp = 2 * np.random.rand(n_macroparticles) - 1
        self.z = 2 * np.random.rand(n_macroparticles) - 1
        self.Deltap = 2 * np.random.rand(n_macroparticles) - 1

    @property
    def n_macroparticles(self):
        return len(self.x)


class Bunch(PDF):
    """Single bunch with 6D coordinates."""

    def __init__(self, n_macroparticles, charge, gamma, intensity, mass, 
                    match_longitudinal, match_transverse):
        """Initiates the bunch with the given Match instances for both the 
        transverse and longitudinal plane."""
        super(Bunch, self).__init__(n_macroparticles)
        self.charge     = charge
        self.gamma      = gamma
        self.intensity  = intensity
        self.mass       = mass

        match_longitudinal.match(self)
        match_transverse.match(self)

        self.id = np.arange(1, n_macroparticles + 1, dtype=int)

    @classmethod
    def asMatchedGaussian(cls, n_macroparticles, charge, gamma, intensity, mass, 
            alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, beta_z, epsn_z):
        """Initialises a Gaussian bunch from the given optics functions."""
        betagamma = np.sqrt(gamma ** 2 - 1)
        p0 = betagamma * mass * c
        match_z  = match.LongitudinalGaussian.fromOptics(beta_z, epsn_z, p0)
        match_xy = match.TransverseGaussian.fromOptics(alpha_x, beta_x, 
                                    epsn_x, alpha_y, beta_y, epsn_y, betagamma)
        return cls(n_macroparticles, charge, gamma, intensity, mass, 
                                                            match_z, match_xy)

    @property
    def beta(self):
        return np.sqrt(1 - 1. / self.gamma ** 2)
    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1 - value ** 2)

    @property
    def betagamma(self):
        return np.sqrt(self.gamma ** 2 - 1)
    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value ** 2 + 1)

    @property
    def p0(self):
        return self.mass * self.gamma * self.beta * c
    @p0.setter
    def p0(self, value):
        self.gamma = value / (self.mass * self.beta * c)

    @property
    def dp(self):
        return self.Deltap / self.p0
    @dp.setter
    def dp(self, value):
        self.Deltap = value * self.p0
    

    # #~ @profile
    def sort_particles(self):
        # update the number of lost particles
        self.n_macroparticles_lost = (self.n_macroparticles - np.count_nonzero(self.id))

        # sort particles according to z (this is needed for correct functioning of bunch.compute_statistics)
        if self.n_macroparticles_lost:
            z_argsorted = np.lexsort((self.z, -np.sign(self.id))) # place lost particles at the end of the array
        else:
            z_argsorted = np.argsort(self.z)

        self.x  = self.x. take(z_argsorted)
        self.xp = self.xp.take(z_argsorted)
        self.y  = self.y. take(z_argsorted)
        self.yp = self.yp.take(z_argsorted)
        self.z  = self.z. take(z_argsorted)
        self.dp = self.dp.take(z_argsorted)
        self.id = self.id.take(z_argsorted)


class Cloud(PDF):

    def __init__(self, n_macroparticles, density, extent_x, extent_y, extent_z):

        self.charge = e
        self.gamma = 1
        self.intensity = density * extent_x * extent_y * extent_z
        self.mass = m_e

        self._create_uniform(n_macroparticles)
        self._match_uniform(extent_x, extent_y, extent_z)

        # Initial distribution
        self.x0 = self.x.copy()
        self.xp0 = self.xp.copy()
        self.y0 = self.y.copy()
        self.yp0 = self.yp.copy()
        self.z0 = self.z.copy()
        self.dp0 = self.dp.copy()

    def _match_uniform(self, extent_x, extent_y, extent_z):

        self.x *= extent_x
        self.xp *= 0
        self.y *= extent_y
        self.yp *= 0
        self.z *= extent_z
        self.dp *= 0

    def reinit(self):

        np.copyto(self.x, self.x0)
        np.copyto(self.xp, self.xp0)
        np.copyto(self.y, self.y0)
        np.copyto(self.yp, self.yp0)
        np.copyto(self.z, self.z0)
        np.copyto(self.dp, self.dp0)


class Ghost(PDF):
    '''
    The ghost class represents a particle ensemble that moves along with its parent ensemble, 
    interacting with all fields excited by the parent ensemble but without exciting any fields itself. 
    It is used for diagnostics purposes to uniformly sample the region around the beam to measure fields, 
    detuning etc. The constructor would most likely have to take the parent ensemble as argument; 
    there would be a similar relationship as with the Slices class. The parent ensemble has no knowledge 
    about the ghost, but the ghost knows about its parent ensemble.
    '''
    pass
