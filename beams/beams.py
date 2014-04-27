'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


import copy, h5py, sys
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi

from beams.slices import *
from beams.matching import match_transverse, match_longitudinal, unmatched_inbucket
from solvers.poissonfft import *


# re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
# rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p


class Beams(object):

    def __init__(self, macrocharge, totalcharge, unitcharge, gamma, mass,
                 alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, sigma_z, sigma_dp,
                 distribution='gauss'):

        if distribution == 'empty':
            _create_empty(macrocharge)
        elif distribution == 'gauss':
            _creat_gauss(macrocharge)
        elif distribution == "uniform":
            _create_uniform(macrocharge)

        _set_beam_physics(totalcharge, unitcharge, gamma, mass)
        _set_beam_geometry(alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, sigma_z, sigma_dp)

        self.x0 = self.x
        self.xp0 = self.xp
        self.y0 = self.y
        self.yp0 = self.yp
        self.z0 = self.z
        self.dp0 = self.dp

    def _create_empty(self, macrocharge):

        self.x = np.zeros(macrocharge)
        self.xp = np.zeros(macrocharge)
        self.y = np.zeros(macrocharge)
        self.yp = np.zeros(macrocharge)
        self.z = np.zeros(macrocharge)
        self.dp = np.zeros(macrocharge)

    def _create_gauss(self, macrocharge):

        self.x = np.random.randn(macrocharge)
        self.xp = np.random.randn(macrocharge)
        self.y = np.random.randn(macrocharge)
        self.yp = np.random.randn(macrocharge)
        self.z = np.random.randn(macrocharge)
        self.dp = np.random.randn(macrocharge)

    def _create_uniform(self, macrocharge):

        self.x = 2 * np.random.rand(macrocharge) - 1
        self.xp = 2 * np.random.rand(macrocharge) - 1
        self.y = 2 * np.random.rand(macrocharge) - 1
        self.yp = 2 * np.random.rand(macrocharge) - 1
        self.z = 2 * np.random.rand(macrocharge) - 1
        self.dp = 2 * np.random.rand(macrocharge) - 1

    def _set_beam_physics(self, totalcharge, unitcharge, gamma, mass):

        self.totalcharge = totalcharge
        self.unitcharge = unitcharge
        self.gamma = gamma
        self.mass = mass

    def _set_beam_geometry(self, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, sigma_z, sigma_dp): pass

    @property:
    def macrocharge(self):

        return len(self.x)

    def reinit():

        self.x = self.x0
        self.xp = self.xp0
        self.y = self.y0
        self.yp = self.yp0
        self.z = self.z0
        self.sp = self.sp0

class Cloud(Ensemble):

    # TODO: rather go for charge, intensity, unitcharge
    # instead of n_macroparticles, n_particles, charge
    # or macrocharge, charge, unitcharge
    # or macrocharge, totalcharge, unitcharge, gamma, mass; particles: q

    @classmethod
    def from_parameters(cls, n_macroparticles, density, extent_x, extent_y, extent_z):

        self = cls.from_uniform(n_macroparticles)

        self.x *= extent_x
        self.xp *= 0
        self.y *= extent_y
        self.yp *= 0
        self.dz *= 0
        self.dp *= 0

        self._set_beam_physics(density, extent_x, extent_y, extent_z)
        self._set_beam_numerics()

        self.x0 = self.x
        self.xp0 = self.xp
        self.y0 = self.yp
        self.yp0 = self.yp

        return self

    def _set_beam_physics(self, density, extent_x, extent_y, extent_z):

        self.n_particles = density * extent_x * extent_y * extent_z
        self.charge = e
        self.gamma = 1
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.mass = m_e
        self.p0 = self.mass * self.gamma * self.beta * c

    def _set_beam_numerics(self):

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

    def push(self, bunch, ix):

        # Normalization factors to speed up computations
        dz = bunch.slices.dz_centers[5] - bunch.slices.dz_centers[4]
        dt = dz / (bunch.beta * c)

        qe = self.n_particles / self.n_macroparticles
        qp = bunch.n_particles / bunch.n_macroparticles
        c_e = -2 * c * re * 1 / bunch.beta * qp
        #   = -2 * c ** 2 * re  * ex / dz * dt * 1 / gamma
        c_p = -2 * 1 * rp * 1 / (bunch.gamma * bunch.beta ** 2) * qe
        #   = -2 * c ** 2 * rp  * ex / dL * dL * 1 / gamma / (beta * c) ** 2

        # Push bunch
        bunch.xp[ix] += c_p * bunch.kx[ix]
        bunch.yp[ix] += c_p * bunch.ky[ix]

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


    def _set_beam_physics(self, n_particles, charge, energy, mass):
        '''
        Set the physical quantities of the beam
        '''
        self.n_particles = n_particles
        self.charge = charge
        self.gamma = energy * e / (mass * c ** 2)
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.mass = mass
        self.p0 = mass * self.gamma * self.beta * c

    def _set_beam_numerics(self):
        '''
        Set the numerical quantities of the beam
        '''
        self.n_macroparticles = len(self.x)
        self.n_macroparticles_lost = 0

        self.id = np.arange(1, len(self.x) + 1, dtype=np.int)
        self.np = np.ones(self.n_macroparticles) * self.n_particles / self.n_macroparticles


class Slices(object):

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

        self.x = self.x.take(dz_argsorted)
        self.xp = self.xp.take(dz_argsorted)
        self.y = self.y.take(dz_argsorted)
        self.yp = self.yp.take(dz_argsorted)
        self.dz = self.dz.take(dz_argsorted)
        self.dp = self.dp.take(dz_argsorted)
        self.id = self.id.take(dz_argsorted)

    def set_in_slice(self, index_after_bin_edges):
        self.in_slice = (self.slices.n_slices + 3) * np.ones(self.n_macroparticles, dtype=np.int)
        for i in xrange(self.slices.n_slices + 2):
            self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i
