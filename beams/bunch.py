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
    def _set_beam_physics(self): pass

    @abstractmethod
    def _set_beam_numerics(self): pass


class Beam(Ensemble):

    pass


class Cloud(Ensemble):

    # TODO: rather go for charge, intensity, unitcharge
    # instead of n_macroparticles, n_particles, charge
    # or macrocharge, charge, unitcharge
    # or macrocharge, totalcharge, unitcharge, gamma, mass; particles: q
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

        self._set_beam_physics(density, extent_x, extent_y, extent_z)
        self._set_beam_numerics()

        self.x0 = self.x.copy()
        self.xp0 = self.xp.copy()
        self.y0 = self.y.copy()
        self.yp0 = self.yp.copy()

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

    def reinitialize(self):

        np.copyto(self.x, self.x0)
        np.copyto(self.xp, self.xp0)
        np.copyto(self.y, self.y0)
        np.copyto(self.yp, self.yp0)

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

    # @profile
    def track(self, bunch):

        bunch.compute_statistics()
        self.reinitialize()

        # phi1 = plt.zeros((bunch.poisson_other.ny, bunch.poisson_other.nx))
        # phi2 = plt.zeros((bunch.poisson_other.ny, bunch.poisson_other.nx))

        index_after_bin_edges = np.cumsum(bunch.slices.n_macroparticles)[:-3]
        index_after_bin_edges[0] = 0

        for i in xrange(bunch.slices.n_slices):
            ix = np.s_[index_after_bin_edges[i]:index_after_bin_edges[i + 1]]

            # Cloud track
            self.poisson_self.gather_from(self.x, self.y, self.poisson_self.rho)
            self.poisson_self.compute_potential()
            self.poisson_self.compute_fields()
            self.poisson_self.scatter_to(bunch.x[ix], bunch.y[ix], bunch.kx[ix], bunch.ky[ix])

            bunch.poisson_other.gather_from(bunch.x[ix], bunch.y[ix], bunch.poisson_other.rho)
            bunch.poisson_other.compute_potential()
            bunch.poisson_other.compute_fields()
            # bunch.poisson_other.compute_potential_fgreenm2m(bunch.poisson_other.x, bunch.poisson_other.y,
            #                                                 phi1, bunch.poisson_other.rho)
            # bunch.poisson_other.compute_potential_fgreenp2m(bunch.x, bunch.y,
            #                                                 bunch.poisson_other.x, bunch.poisson_other.y,
            #                                                 phi2, bunch.poisson_other.rho)
            bunch.poisson_other.scatter_to(self.x, self.y, self.kx, self.ky)

            self.push(bunch, ix)

            # if i == 0:
            #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
            # [ax.cla() for ax in (ax1, ax2, ax3, ax4)]
            # # [ax.set_aspect('equal') for ax in (ax1, ax2, ax3, ax4)]
            # ax1.contour(bunch.poisson_other.fgreen.T, 100)
            # ax2.plot(bunch.poisson_other.phi[bunch.poisson_other.ny / 2,:], '-g')
            # # ax2.plot(phi1[bunch.poisson_other.ny / 2,:], '-r')
            # # ax2.plot(phi2[bunch.poisson_other.ny / 2,:], '-', c='orange')
            # # ax3.contourf(self.poisson_self.x, self.poisson_self.y, 10 * plt.log10(self.poisson_self.rho), 100)
            # print 10 * plt.log10(self.poisson_self.rho)
            # ax3.imshow(10 * plt.log10(self.poisson_self.rho), origin='lower', aspect='auto', vmin=10, vmax=1e2,
            #            extent=(self.poisson_self.x[0,0], self.poisson_self.x[0,-1], self.poisson_self.y[0,0], self.poisson_self.y[-1,0]))
            # # ax3.scatter(self.x[::20], self.y[::20], c='b', marker='.')
            # # ax3.quiver(self.x[::50], self.y[::50], self.kx[::50], self.ky[::50], color='g')
            # # ax3.contour(p.x, p.y, p.phi, 100, lw=2)
            # # ax3.scatter(bunch.x[ix], bunch.y[ix], c='y', marker='.', alpha=0.8)
            # ax4.imshow(plt.sqrt(bunch.poisson_other.ex ** 2 + bunch.poisson_other.ey ** 2), origin='lower', aspect='auto',
            #            extent=(bunch.poisson_other.x[0,0], bunch.poisson_other.x[0,-1], bunch.poisson_other.y[0,0], bunch.poisson_other.y[-1,0]))

            # plt.draw()


class Ghost(Ensemble):

    def _set_beam_physics(self): pass

    def _set_beam_numerics(self): pass


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

        self._set_beam_physics(n_particles, charge, energy, mass)
        self._set_beam_numerics()

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

        self._set_beam_physics(n_particles, charge, energy, mass)
        self._set_beam_numerics()

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

        self._set_beam_physics(n_particles, charge, energy, mass)
        self._set_beam_numerics()

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

        self._set_beam_physics(n_particles, charge, energy, mass)
        self._set_beam_numerics()

        return self

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

    # def add_poisson(self, poisson):

    #     self.poisson_self = poisson
    #     # self.poisson_other = copy.copy(poisson)

    # def copy_poisson(self, poisson):

    #     self.poisson = copy.copy(poisson)
