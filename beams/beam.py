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


class Beam(object):

    def __init__(self, n_macroparticles, n_particles, charge, gamma, mass,
                 alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, sigma_z, sigma_dp,
                 distribution='gauss'):

        if distribution == 'empty':
            _create_empty(n_macroparticles)
        elif distribution == 'gauss':
            _creat_gauss(n_macroparticles)
        elif distribution == "uniform":
            _create_uniform(n_macroparticles)

        _set_beam_physics(n_particles, charge, gamma, mass)
        _set_beam_geometry(alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, sigma_z, sigma_dp)

        self.x0 = self.x.copy()
        self.xp0 = self.xp.copy()
        self.y0 = self.y.copy()
        self.yp0 = self.yp.copy()
        self.z0 = self.z.copy()
        self.dp0 = self.dp.copy()

    def _create_empty(self, n_macroparticles):

        self.x = np.zeros(n_macroparticles)
        self.xp = np.zeros(n_macroparticles)
        self.y = np.zeros(n_macroparticles)
        self.yp = np.zeros(n_macroparticles)
        self.z = np.zeros(n_macroparticles)
        self.dp = np.zeros(n_macroparticles)

    def _create_gauss(self, n_macroparticles):

        self.x = np.random.randn(n_macroparticles)
        self.xp = np.random.randn(n_macroparticles)
        self.y = np.random.randn(n_macroparticles)
        self.yp = np.random.randn(n_macroparticles)
        self.z = np.random.randn(n_macroparticles)
        self.dp = np.random.randn(n_macroparticles)

    def _create_uniform(self, n_macroparticles):

        self.x = 2 * np.random.rand(n_macroparticles) - 1
        self.xp = 2 * np.random.rand(n_macroparticles) - 1
        self.y = 2 * np.random.rand(n_macroparticles) - 1
        self.yp = 2 * np.random.rand(n_macroparticles) - 1
        self.z = 2 * np.random.rand(n_macroparticles) - 1
        self.dp = 2 * np.random.rand(n_macroparticles) - 1

    def _set_beam_physics(self, n_particles, charge, gamma, mass):

        self.n_particles = n_particles
        self.charge = charge
        self.gamma = gamma
        self.mass = mass

    def _set_beam_geometry(self, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, sigma_z, sigma_dp,
                           distribution='gauss'): pass

    @property
    def n_macroparticles(self):

        return len(self.x)

    @property
    def beta(self):

        return np.sqrt(1 - 1 / self.gamma ** 2)

    @property
    def p0(self):

        return self.mass * self.gamma * self.beta * c

    def reinit():

        np.copyto(self.x, self.x0)
        np.copyto(self.xp, self.xp0)
        np.copyto(self.y, self.y0)
        np.copyto(self.yp, self.yp0)
        np.copyto(self.z, self.z0)
        np.copyto(self.dp, self.dp0)




import cobra_functions.stats as cp


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices, nsigmaz=None, mode='cspace'):
        '''
        Constructor
        '''
        self.nsigmaz = nsigmaz
        self.mode = mode

        self.mean_x = np.zeros(n_slices)
        self.mean_xp = np.zeros(n_slices)
        self.mean_y = np.zeros(n_slices)
        self.mean_yp = np.zeros(n_slices)
        self.mean_dz = np.zeros(n_slices)
        self.mean_dp = np.zeros(n_slices)
        self.sigma_x = np.zeros(n_slices)
        self.sigma_y = np.zeros(n_slices)
        self.sigma_dz = np.zeros(n_slices)
        self.sigma_dp = np.zeros(n_slices)
        self.epsn_x = np.zeros(n_slices)
        self.epsn_y = np.zeros(n_slices)
        self.epsn_z = np.zeros(n_slices)

        self.n_macroparticles = np.zeros(n_slices, dtype=int)
        # self.dz_centers = np.zeros(n_slices)
        # self.dz_bins = np.zeros(n_slices + 1)

    @property
    def n_slices(self):

        return len(self.mean_x)

    def get_longitudinal_cuts(self, bunch):

        if self.nsigmaz == None:
            z_cut_tail = bunch.dz[0]
            z_cut_head = bunch.dz[-1 - bunch.n_macroparticles_lost]
        else:
            mean_z = cp.mean(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            sigma_z = cp.std(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            z_cut_tail = mean_z - self.nsigmaz * sigma_z
            z_cut_head = mean_z + self.nsigmaz * sigma_z

        return z_cut_tail, z_cut_head

    # @profile
    def slice_constant_space(self, bunch):

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        bunch.sort_particles()

        # determine the longitudinal cuts (this allows for the user defined static cuts: self.z_cut_tail, self.z_cut_head)
        try:
            self.z_cut_tail, self.z_cut_head
        except AttributeError:
            self.z_cut_tail, self.z_cut_head = self.get_longitudinal_cuts(bunch)

        # 1. z-bins
        # z_bins = np.zeros(self.n_slices + 3)
        # # TODO: ask Hannes: is this check neccessary?
        # z_bins[0] = np.min([bunch.dz[0], z_cut_tail])
        # z_bins[-1] = np.max([bunch.dz[- 1 - bunch.n_macroparticles_lost], z_cut_head])
        # # Not so nice, dz not explicit
        # Constant space
        # dz = (self.z_cut_head - self.z_cut_tail) / self.n_slices
        # self.z_bins = np.arange(self.z_cut_tail, self.z_cut_head + dz, dz)
        self.z_bins = np.linspace(self.z_cut_tail, self.z_cut_head, self.n_slices + 1) # more robust than arange to reach z_cut_head exactly
        self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

        # 2. n_macroparticles - equivalet to x0 <= x < x1 binning
        z_bins_all = np.hstack((self.z_cut_tail, self.z_bins, self.z_cut_head))
        first_index_in_bin = np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], z_bins_all)
        first_index_in_bin[np.where(z_bins_all == bunch.dz[-1 - bunch.n_macroparticles_lost])] += 1 # treat last bin for x0 <= x <= x1
        self.z_index = first_index_in_bin[1:-2]

        n_macroparticles = np.diff(first_index_in_bin)
        self.n_cut_tail = n_macroparticles[0]
        self.n_cut_head = n_macroparticles[-1]
        self.n_macroparticles[:] = n_macroparticles[1:-1]

        # .in_slice indicates in which slice the particle is (needed for wakefields)
        # bunch.set_in_slice(index_after_bin_edges)

    def slice_constant_charge(self, bunch):

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        bunch.sort_particles()

        # determine the longitudinal cuts (this allows for the user defined static cuts: self.z_cut_tail, self.z_cut_head)
        try:
            self.z_cut_tail, self.z_cut_head
        except AttributeError:
            self.z_cut_tail, self.z_cut_head = self.get_longitudinal_cuts(bunch)

        # 1. n_macroparticles
        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        self.n_cut_tail = np.searchsorted(bunch.dz[:n_macroparticles_alive], self.z_cut_tail)
        self.n_cut_head = n_macroparticles_alive - (np.searchsorted(bunch.dz[:n_macroparticles_alive], self.z_cut_head) + 1) # always throw last index into slices (x0 <= x <= x1)
        # distribute macroparticles uniformly along slices
        q0 = n_macroparticles_alive - self.n_cut_tail - self.n_cut_head
        self.n_macroparticles[:] = q0 // self.n_slices
        self.n_macroparticles[np.random.randint(self.n_slices, size=q0 % self.n_slices)] += 1
        n_macroparticles_all = np.hstack((self.n_cut_tail, self.n_macroparticles, self.n_cut_head))

        # 2. z-bins
        # Get indices of the particles defining the bin edges
        first_index_in_bin = np.append(0, np.cumsum(n_macroparticles_all))
        self.z_index = first_index_in_bin[1:-2]
        self.z_bins = map(lambda i: bunch.dz[self.z_index[i] - 1] + (bunch.dz[self.z_index[i]] - bunch.dz[self.z_index[i] - 1]) / 2,
                          np.arange(1, self.n_slices))
        self.z_bins = np.hstack((self.z_cut_tail, self.z_bins, self.z_cut_head))

        # # 1. z-bins
        # # z_bins = np.zeros(self.n_slices + 3)
        # # # TODO: ask Hannes: is this check neccessary?
        # # z_bins[0] = np.min([bunch.dz[0], z_cut_tail])
        # # z_bins[-1] = np.max([bunch.dz[- 1 - bunch.n_macroparticles_lost], z_cut_head])
        # # # Not so nice, dz not explicit
        # # Constant space
        # # dz = (self.z_cut_head - self.z_cut_tail) / self.n_slices
        # # self.z_bins = np.arange(self.z_cut_tail, self.z_cut_head + dz, dz)
        # self.z_bins = np.linspace(self.z_cut_tail, self.z_cut_head, self.n_slices + 1) # more robust than arange to reach z_cut_head exactly
        # self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

        # # 2. n_macroparticles - equivalet to x0 <= x < x1 binning
        # z_bins_all = np.hstack((self.z_cut_tail, self.z_bins, self.z_cut_head))
        # first_index_in_bin = np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], z_bins_all)
        # first_index_in_bin[np.where(z_bins_all == bunch.dz[-1 - bunch.n_macroparticles_lost])] += 1 # treat last bin for x0 <= x <= x1
        # self.z_index = first_index_in_bin[1:-2]

        # bin centers
        self.z_centers = map((lambda i: cp.mean(bunch.dz[first_index_in_bin[i]:first_index_in_bin[i+1]])), np.arange(self.n_slices))

    def update_slices(self, bunch):

        if self.mode == 'ccharge':
            self.slice_constant_charge(bunch)
        elif self.mode == 'cspace':
            self.slice_constant_space(bunch)

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

    # def set_in_slice(self, index_after_bin_edges):

    #     self.in_slice = (self.slices.n_slices + 3) * np.ones(self.n_macroparticles, dtype=np.int)

    #     for i in xrange(self.slices.n_slices + 2):
    #         self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i
