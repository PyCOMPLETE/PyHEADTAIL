'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


from beams.slices import *
from configuration import *
from trackers.longitudinal_tracker import *
import cobra_functions.cobra_functions as cp


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
    def from_copy(cls, x, xp, y, yp, dz, dp):

        x = np.copy(x)
        xp = np.copy(xp)
        y = np.copy(y)
        yp = np.copy(yp)
        dz = np.copy(dz)
        dp = np.copy(dp)

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @classmethod
    def from_file(cls):
        pass

    @classmethod
    def from_empty(cls, n_particles):

        x = np.zeros(n_particles)
        xp = np.zeros(n_particles)
        y = np.zeros(n_particles)
        yp = np.zeros(n_particles)
        dz = np.zeros(n_particles)
        dp = np.zeros(n_particles)

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @classmethod
    def from_gaussian(cls, n_particles):

        x = np.random.randn(n_particles)
        xp = np.random.randn(n_particles)
        y = np.random.randn(n_particles)
        yp = np.random.randn(n_particles)
        dz = np.random.randn(n_particles)
        dp = np.random.randn(n_particles)

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @classmethod
    def from_uniform(cls, n_particles):

        x = np.random.rand(n_particles) * 2 - 1
        xp = np.random.rand(n_particles) * 2 - 1
        y = np.random.rand(n_particles) * 2 - 1
        yp = np.random.rand(n_particles) * 2 - 1
        dz = np.random.rand(n_particles) * 2 - 1
        dp = np.random.rand(n_particles) * 2 - 1

        self = cls(x, xp, y, yp, dz, dp)

        return self

    # TODO: perhaps decorate with matchings...
    @classmethod
    def from_matching(cls, n_particles, charge, energy, intensity, mass,
                      match_transverse=None, match_longitudinal=None, slices=None):

        self = cls.from_gaussian(n_particles)

        self.set_scalar_quantities(charge, energy, intensity, mass)

        match_transverse(self)
        match_longitudinal(self)

        self.slices = slices

        return self

    def set_scalar_quantities(self, charge, energy, intensity, mass):

        self.charge = charge
        self.gamma = energy * 1e9 * charge * e / (mass * c ** 2) + 1
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.intensity = intensity
        self.mass = mass
        self.p0 = mass * self.gamma * self.beta * c

    # @profile
    def compute_statistics(self):

        if not hasattr(self, 'slices'):
            print "*** WARNING: bunch not yet sliced! Aborting..."
            sys.exit(-1)
        else:
            n_particles = len(self.x)

            indices = [self.slices.index(i) for i in range(n_slices + 2)]
            indices.append(range(n_particles))

        for i in xrange(n_slices + 3):
            n = len(indices[i])
            if n:
                x = self.x[indices[i]]
                xp = self.xp[indices[i]]
                y = self.y[indices[i]]
                yp = self.yp[indices[i]]
                dz = self.dz[indices[i]]
                dp = self.dp[indices[i]]

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




    @classmethod
    def from_parameters(cls, n_particles, charge, energy, intensity, mass,
                        epsn_x, beta_x, epsn_y, beta_y, epsn_z, length, cavity=None, matching='simple'):

        x = np.random.randn(n_particles)
        xp = np.random.randn(n_particles)
        y = np.random.randn(n_particles)
        yp = np.random.randn(n_particles)
        dz = np.random.randn(n_particles)
        dp = np.random.randn(n_particles)

        self = cls(x, xp, y, yp, dz, dp)

        # self.charge = charge
        # self.gamma = energy * 1e9 * charge * e / (mass * c ** 2) + 1
        # self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        # self.mass = mass
        # p0 = mass * self.gamma * self.beta * c / e

        # matching.match_transverse(self, ...)
        # matching.match_simple(self, cavity)
        # matching.match_full(self, cavity)
        self.match_distribution(charge, energy, intensity, mass,
                                epsn_x, beta_x, epsn_y, beta_y, epsn_z, length)
        if cavity:
            if matching == 'simple':
                match_simple(self, cavity)
            elif matching == 'full':
                match_full(self, cavity)
            else:
                raise ValueError("Unknown matching " + matching)
        else:
            pass

        return self

    # TODO: perhaps throw to matching/matcher and mark transverse
    def match_distribution(self, charge, energy,  intensity, mass,
                           epsn_x, beta_x, epsn_y, beta_y, epsn_z, length):

        self.charge = charge
        self.gamma = energy * 1e9 * charge * e / (mass * c ** 2) + 1
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.intensity = intensity
        self.mass = mass
        p0 = mass * self.gamma * self.beta * c / e

        sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (self.gamma * self.beta))
        sigma_xp = sigma_x / beta_x
        sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (self.gamma * self.beta))
        sigma_yp = sigma_y / beta_y
        sigma_dz = length
        sigma_dp = epsn_z / (4 * np.pi * sigma_dz) / p0

        self.x *= sigma_x
        self.xp *= sigma_xp
        self.y *= sigma_y
        self.yp *= sigma_yp
        self.dz *= sigma_dz
        self.dp *= sigma_dp

    def update_slices(self):

        # if not hasattr(self, 'slices'):
        #     self.slices = Slices(n_slices)

        assert(hasattr(self, 'slices'))

        if self.slices.slicemode == 'ccharge':
            self.slices.slice_constant_charge(self, self.slices.nsigmaz)
        elif self.slices.slicemode == 'cspace':
            self.slices.slice_constant_space(self, self.slices.nsigmaz)
