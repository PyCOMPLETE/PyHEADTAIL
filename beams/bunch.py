'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


import copy, h5py, sys
from abc import ABCMeta, abstractmethod
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi

from beams.slices import *
from beams.matching import match_transverse, match_longitudinal, unmatched_inbucket
from solvers.poissonfft import *


re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p


class Ensemble(object):

    __metaclass__ = ABCMeta

    def __init__(self, x, xp, y, yp, dz, dp):

        assert(len(x) == len(xp) == len(y) == len(yp) == len(dz) == len(dp))

        self.x = x
        self.xp = xp
        self.y = y
        self.yp = yp
        self.dz = dz
        self.dp = dp

    @classmethod
    def from_empty(cls, n_macroparticles):

        x = np.zeros(n_macroparticles)
        xp = np.zeros(n_macroparticles)
        y = np.zeros(n_macroparticles)
        yp = np.zeros(n_macroparticles)
        dz = np.zeros(n_macroparticles)
        dp = np.zeros(n_macroparticles)

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @classmethod
    def from_gauss(cls, n_macroparticles):

        x = np.random.randn(n_macroparticles)
        xp = np.random.randn(n_macroparticles)
        y = np.random.randn(n_macroparticles)
        yp = np.random.randn(n_macroparticles)
        dz = np.random.randn(n_macroparticles)
        dp = np.random.randn(n_macroparticles)

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @classmethod
    def from_uniform(cls, n_macroparticles):

        x = 2 * np.random.rand(n_macroparticles) - 1
        xp = 2 * np.random.rand(n_macroparticles) - 1
        y = 2 * np.random.rand(n_macroparticles) - 1
        yp = 2 * np.random.rand(n_macroparticles) - 1
        dz = 2 * np.random.rand(n_macroparticles) - 1
        dp = 2 * np.random.rand(n_macroparticles) - 1

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @abstractmethod
    def set_beam_physics(self): return None

    @abstractmethod
    def set_beam_numerics(self): return None


class Beam(Ensemble):

    pass


class Cloud(Ensemble):

    # TODO: rather go for charge, intensity, unitcharge
    # instead of n_macroparticles, n_particles, charge
    # or macrocharge, charge, unitcharge
    @classmethod
    def from_file(self): pass

    @classmethod
    def from_parameters(cls, n_macroparticles, density, extent_x, extent_y, extent_z):

        self = cls.from_uniform(n_macroparticles)

        self.x *= extent_x
        self.xp *= 0
        self.y *= extent_y
        self.yp *= 0
        self.dz *= 0
        self.dp *= 0

        self.set_beam_physics(density, extent_x, extent_y, extent_z)
        self.set_beam_numerics()

        self.x0 = self.x
        self.xp0 = self.xp
        self.y0 = self.yp
        self.yp0 = self.yp

        return self

    def set_beam_physics(self, density, extent_x, extent_y, extent_z):

        self.n_particles = density * extent_x * extent_y * extent_z
        self.charge = e
        self.gamma = 1
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.mass = m_e
        self.p0 = self.mass * self.gamma * self.beta * c

    def set_beam_numerics(self):

        self.n_macroparticles = len(self.x)

        self.id = np.arange(1, len(self.x) + 1)
        self.np = np.ones(self.n_macroparticles) * self.n_particles / self.n_macroparticles

    def add_poisson(self, extent_x, extent_y, nx, ny, other=None):

        self.poisson_self = PoissonFFT(extent_x, extent_y, nx, ny)
        self.kx = np.zeros(self.n_macroparticles)
        self.ky = np.zeros(self.n_macroparticles)

        if other:
            other.poisson_other = copy.deepcopy(self.poisson_self)
            other.kx = np.zeros(other.n_macroparticles)
            other.ky = np.zeros(other.n_macroparticles)

    # # def add_poisson(self, poisson):

    #     # self.poisson_self = poisson
    #     # self.poisson_other = copy.copy(poisson)

    # def copy_poisson(self, poisson):

    #     self.poisson = copy.copy(poisson)

    def reinitialize(self):

        self.x = self.x0
        self.xp = self.xp0
        self.y = self.y0
        self.yp = self.yp0

    def push(self, bunch, i_slice):

        # Normalization factors to speed up computations
        dz = bunch.slice_dz[i_slice + 1] - bunch.slice_dz[i_slice]
        dt = dz / (bunch.beta * c)
        c_e = -2 * c * re * dz * 1 / (1 * bunch.beta)
        #   = -2 * c ** 2 * re  * Ex / dz * dt * 1 / gamma
        c_p = -2 * 1 * rp * dL * 1 / (bunch.gamma * bunch.beta ** 2)
        #   = -2 * c ** 2 * rp  * Ex / dL * dL * 1 / gamma / (beta * c) ** 2

        # Line charge density
        lambda_e = self.density / self.n_macroparticles * (max_x - min_x) * (max_y - min_y)
        lambda_p = bunch.n_particles / bunch.n_macroparticles / dz;

        # Push bunch
        indices = np.s_[::2]
        bunch.xp[indices] += c_p * bunch.kx[indices]
        bunch.yp[indices] += c_p * bunch.ky[indices]

        # Push cloud
        self.xp += c_e * self.kx
        self.yp += c_e * self.ky
        self.x += self.xp * dt
        self.y += self.yp * dt

    def track(self, bunch):

        # self.reinitialize()
        # self.poisson.initialize<Bunch&, Cloud&>(bunch, *this)

        for i in xrange(bunch.slices.n_slices):
            dz = 1
            poisson = self.poisson_self
            lambda_ = self.n_particles / self.n_macroparticles * self.charge / dz
            poisson.fastgather(self.x, self.y, lambda_)
            poisson.compute_potential(poisson)

            dz = 1
            poisson = bunch.poisson_other
            lambda_ = bunch.n_particles / bunch.n_macroparticles * bunch.charge / dz
            poisson.fastgather(bunch.x, bunch.y, lambda_)
            poisson.compute_potential(poisson)

            # poisson.fastgather<Bunch&>(bunch, i)
            # poisson.computePotential<Bunch&>(bunch, i)
            # poisson.compute_field<Bunch&>(bunch, i)

            # poisson.fastgather<Cloud&>(*this, i)
            # poisson.computePotential<Cloud&>(*this, i)
            # poisson.compute_field<Cloud&>(*this, i)

            # poisson.parallelscatter<Bunch&, Cloud&>(bunch, *this, i)

            # self.push(bunch, i)


class Ghost(Ensemble):

    def set_beam_physics(self): pass

    def set_beam_numerics(self): pass


def bunch_matched_and_sliced(n_macroparticles, n_particles, charge, energy, mass,
                             epsn_x, epsn_y, ltm, bunch_length, bucket, matching,
                             n_slices, nsigmaz, slicemode='cspace'):

    # bunch = Bunch.from_empty(1e3, n_particles, charge, energy, mass)
    # x, xp, y, yp, dz, dp = random.gsl_quasirandom(bunch)
    bunch = Bunch.from_gaussian(n_macroparticles, n_particles, charge, energy, mass)
    bunch.match_transverse(epsn_x, epsn_y, ltm)
    bunch.match_longitudinal(bunch_length, bucket, matching)
    bunch.set_slices(Slices(n_slices, nsigmaz, slicemode))
    bunch.update_slices()

    return bunch

def bunch_unmatched_inbucket_sliced(n_macroparticles, n_particles, charge, energy, mass,
                                    epsn_x, epsn_y, ltm, sigma_dz, sigma_dp, bucket,
                                    n_slices, nsigmaz, slicemode='cspace'):
    bunch = Bunch.from_gaussian(n_macroparticles, n_particles, charge, energy, mass)
    bunch.match_transverse(epsn_x, epsn_y, ltm)
    bunch.unmatched_inbucket(sigma_dz, sigma_dp, bucket)
    bunch.set_slices(Slices(n_slices, nsigmaz, slicemode))
    bunch.update_slices()

    return bunch

def bunch_from_file(filename, step, n_particles, charge, energy, mass,
                    n_slices, nsigmaz, slicemode='cspace'):

    bunch = Bunch.from_h5file(filename, step, n_particles, charge, energy, mass)
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
    def from_empty(cls, n_macroparticles, n_particles, charge, energy, mass):

        x = np.zeros(n_macroparticles)
        xp = np.zeros(n_macroparticles)
        y = np.zeros(n_macroparticles)
        yp = np.zeros(n_macroparticles)
        dz = np.zeros(n_macroparticles)
        dp = np.zeros(n_macroparticles)

        self = cls(x, xp, y, yp, dz, dp)

        self.set_beam_physics(n_particles, charge, energy, mass)
        self.set_beam_numerics()

        return self

    @classmethod
    def from_h5file(cls, filename, step, n_particles, charge, energy, mass):
        # TODO
        particles = h5py.File(filename + '.h5part', 'r')

        x = np.array(particles['Step#' + str(step)]['x'], dtype=np.double)
        xp = np.array(particles['Step#' + str(step)]['xp'], dtype=np.double)
        y = np.array(particles['Step#' + str(step)]['y'], dtype=np.double)
        yp = np.array(particles['Step#' + str(step)]['yp'], dtype=np.double)
        dz = np.array(particles['Step#' + str(step)]['dz'], dtype=np.double)
        dp = np.array(particles['Step#' + str(step)]['dp'], dtype=np.double)

        self = cls(x, xp, y, yp, dz, dp)

        self.set_beam_physics(n_particles, charge, energy, mass)
        self.set_beam_numerics()

        self.id = np.array(particles['Step#' + str(step)]['id'])

        return self

    @classmethod
    def from_gaussian(cls, n_macroparticles, n_particles, charge, energy, mass):

        x = np.random.randn(n_macroparticles)
        xp = np.random.randn(n_macroparticles)
        y = np.random.randn(n_macroparticles)
        yp = np.random.randn(n_macroparticles)
        dz = np.random.randn(n_macroparticles)
        dp = np.random.randn(n_macroparticles)

        self = cls(x, xp, y, yp, dz, dp)

        self.set_beam_physics(n_particles, charge, energy, mass)
        self.set_beam_numerics()

        return self

    @classmethod
    def from_uniform(cls, n_macroparticles, n_particles, charge, energy, mass):

        x = np.random.rand(n_macroparticles) * 2 - 1
        xp = np.random.rand(n_macroparticles) * 2 - 1
        y = np.random.rand(n_macroparticles) * 2 - 1
        yp = np.random.rand(n_macroparticles) * 2 - 1
        dz = np.random.rand(n_macroparticles) * 2 - 1
        dp = np.random.rand(n_macroparticles) * 2 - 1

        self = cls(x, xp, y, yp, dz, dp)

        self.set_beam_physics(n_particles, charge, energy, mass)
        self.set_beam_numerics()

        return self

    def set_beam_physics(self, n_particles, charge, energy, mass):
        '''
        Set the physical quantities of the beam
        '''
        self.n_particles = n_particles
        self.charge = charge
        self.gamma = energy * e / (mass * c ** 2)
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.mass = mass
        self.p0 = mass * self.gamma * self.beta * c

    def set_beam_numerics(self):
        '''
        Set the numerical quantities of the beam
        '''
        self.n_macroparticles = len(self.x)
        self.n_macroparticles_lost = 0

        self.id = np.arange(1, len(self.x) + 1, dtype=np.int)
        self.np = np.ones(self.n_macroparticles) * self.n_particles / self.n_macroparticles

    # def set_scalar_quantities(self, charge, energy, n_particles, mass):

    #     self.charge = charge
    #     self.gamma = energy * charge * e / (mass * c ** 2)
    #     self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
    #     self.n_particles = n_particles
    #     self.mass = mass
    #     self.p0 = mass * self.gamma * self.beta * c

    def match_transverse(self, epsn_x, epsn_y, ltm):

        match_transverse(epsn_x, epsn_y, ltm)(self)

    def match_longitudinal(self, length, bucket=None, matching=None):

        match_longitudinal(length, bucket, matching)(self)

    def unmatched_inbucket(self, sigma_dz, sigma_dp, bucket=None):

        # TODO: can we be consistant in that matching returns callables?
        unmatched_inbucket(self, sigma_dz, sigma_dp, bucket)

    # @profile
    def compute_statistics(self):

        if not hasattr(self, 'slices'):
            print "*** WARNING: bunch not yet sliced! Aborting..."
            sys.exit(-1)

        # determine the start and end indices of each slices
        i1 = np.append(np.cumsum(self.slices.n_macroparticles[:-2]), np.cumsum(self.slices.n_macroparticles[-2:]))
        i0 = np.zeros(len(i1), dtype=np.int)
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
        self.n_macroparticles_lost = (self.n_macroparticles - np.count_nonzero(self.id))

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        if self.n_macroparticles_lost:
            dz_argsorted = np.lexsort((self.dz, -np.sign(self.id))) # place lost particles at the end of the array
        else:
            dz_argsorted = np.argsort(self.dz)

        self.x = self.x[dz_argsorted]
        self.xp = self.xp[dz_argsorted]
        self.y = self.y[dz_argsorted]
        self.yp = self.yp[dz_argsorted]
        self.dz = self.dz[dz_argsorted]
        self.dp = self.dp[dz_argsorted]
        self.id = self.id[dz_argsorted]

    def set_in_slice(self, index_after_bin_edges):
        self.in_slice = (self.slices.n_slices + 3) * np.ones(self.n_macroparticles, dtype=np.int)
        for i in xrange(self.slices.n_slices + 2):
            self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i

    # def add_poisson(self, poisson):

    #     self.poisson_self = poisson
    #     # self.poisson_other = copy.copy(poisson)

    # def copy_poisson(self, poisson):

    #     self.poisson = copy.copy(poisson)
