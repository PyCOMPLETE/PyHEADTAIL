'''
@authors: Hannes Bartosik,
          Kevin Li,
          Michael Schenk
@date:    06/01/2014
'''


import numpy as np


from random import sample
import cobra_functions.stats as cp


class Slicer(object):
    '''
    Slicer class that controls longitudinal discretization of a beam.
    '''
    def __init__(self, n_slices, nsigmaz=None, mode='const_space', z_cuts=None):

        self.n_slices = n_slices
        self.nsigmaz = nsigmaz
        self.mode = mode

        if z_cuts:
            self.z_cut_tail, self.z_cut_head = z_cuts
            self.z_bins = np.linspace(self.z_cut_tail, self.z_cut_head, self.n_slices + 1)
            self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.
            self.slice_width = (self.z_cut_head - self.z_cut_tail) / self.n_slices


    def _set_longitudinal_cuts(self, bunch):

        if self.nsigmaz == None:
            z_cut_tail = np.min(bunch.z[:(bunch.n_macroparticles - bunch.n_macroparticles_lost)])
            z_cut_head = np.max(bunch.z[:(bunch.n_macroparticles - bunch.n_macroparticles_lost)])
        else:
            mean_z  = cp.mean(bunch.z[:(bunch.n_macroparticles - bunch.n_macroparticles_lost)])
            sigma_z = cp.std(bunch.z[:(bunch.n_macroparticles - bunch.n_macroparticles_lost)])

            z_cut_tail = mean_z - self.nsigmaz * sigma_z
            z_cut_head = mean_z + self.nsigmaz * sigma_z

        return z_cut_tail, z_cut_head


    def _slice_constant_space(self, bunch):

        try:
            z_cut_tail, z_cut_head = self.z_cut_tail, self.z_cut_head
            slice_width            = self.slice_width
        except AttributeError:
            z_cut_tail, z_cut_head = self._set_longitudinal_cuts(bunch)
            slice_width            = (z_cut_head - z_cut_tail) / self.n_slices
            # linspace is more robust than arange. To reach z_cut_head exactly.
            self.z_bins    = np.linspace(z_cut_tail, z_cut_head, self.n_slices + 1)
            self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

        self.slice_index_of_particle = np.floor((bunch.z + abs(z_cut_tail)) / slice_width ).astype(np.int)
        self.particles_within_cuts   = np.where((self.slice_index_of_particle > -1) &
                                                (self.slice_index_of_particle < self.n_slices))[0]
        self._count_macroparticles_per_slice()


    def _slice_constant_space_old(self, bunch):
        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        bunch.sort_particles()

        # 1. z-bins
        try:
            z_cut_tail, z_cut_head = self.z_cut_tail, self.z_cut_head
        except AttributeError:
            z_cut_tail, z_cut_head = self._set_longitudinal_cuts(bunch)
            self.z_bins = np.linspace(z_cut_tail, z_cut_head, self.n_slices + 1) # more robust than arange, to reach z_cut_head exactly
            self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        self.n_cut_tail = +np.searchsorted(bunch.z[:n_macroparticles_alive], z_cut_tail)
        self.n_cut_head = -np.searchsorted(bunch.z[:n_macroparticles_alive], z_cut_head) + n_macroparticles_alive

        # 2. n_macroparticles
        z_bins_all = np.hstack((bunch.z[0], self.z_bins, bunch.z[n_macroparticles_alive - 1]))
        first_index_in_bin = np.searchsorted(bunch.z[:n_macroparticles_alive], z_bins_all)
        if (self.z_bins[-1] in bunch.z[:n_macroparticles_alive]): first_index_in_bin[-1] += 1
        self.first_particle_index_in_slice = first_index_in_bin[1:-1]

        self.n_macroparticles = np.diff(first_index_in_bin)[1:-1]

        # .in_slice indicates in which slice the particle is (needed for wakefields)
        self._set_slice_index_of_particle(bunch)


    def _slice_constant_charge(self, bunch):

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        bunch.sort_particles()

        # try:
        #     z_cut_tail, z_cut_head = self.z_cut_tail, self.z_cut_head
        # except AttributeError:
        z_cut_tail, z_cut_head = self._set_longitudinal_cuts(bunch)

        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        self.n_cut_tail = +np.searchsorted(bunch.z[:n_macroparticles_alive], z_cut_tail)
        self.n_cut_head = -np.searchsorted(bunch.z[:n_macroparticles_alive], z_cut_head) + n_macroparticles_alive

        # 1. n_macroparticles - distribute macroparticles uniformly along slices.
        # Must be integer. Distribute remaining particles randomly among slices with indices 'ix'.
        q0 = n_macroparticles_alive - self.n_cut_tail - self.n_cut_head
        ix = sample(range(self.n_slices), q0 % self.n_slices)

        self.n_macroparticles = (q0 // self.n_slices) * np.ones(self.n_slices, dtype=int)
        self.n_macroparticles[ix] += 1

        # 2. z-bins
        # Get indices of the particles defining the bin edges
        n_macroparticles_all = np.hstack((self.n_cut_tail, self.n_macroparticles, self.n_cut_head))
        first_index_in_bin = np.cumsum(n_macroparticles_all)
        self.first_particle_index_in_slice = first_index_in_bin[:-1]
        self.first_particle_index_in_slice = (self.first_particle_index_in_slice).astype(int)

        # print(self.z_index.shape)
        self.z_bins = (bunch.z[self.first_particle_index_in_slice - 1] + bunch.z[self.first_particle_index_in_slice]) / 2.
        self.z_bins[0], self.z_bins[-1] = z_cut_tail, z_cut_head
        self.z_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2.

        self._set_slice_index_of_particle(bunch)
        self.particles_within_cuts = np.arange(self.n_cut_tail, n_macroparticles_alive - self.n_cut_head)

        # # self.z_centers = map((lambda i: cp.mean(bunch.z[first_index_in_bin[i]:first_index_in_bin[i+1]])), np.arange(self.n_slices)


    def _count_macroparticles_per_slice(self):

        try:
            cp.macroparticles_per_slice(self.slice_index_of_particle, self.particles_within_cuts,
                                        self.n_macroparticles)
        except AttributeError:
            self.n_macroparticles = np.zeros(self.n_slices, dtype=np.int)
            cp.count_macroparticles_per_slice(self.slice_index_of_particle, self.particles_within_cuts,
                                              self.n_macroparticles)


    def _set_slice_index_of_particle(self, bunch):

        try:
            self.slice_index_of_particle
        except AttributeError:
            self.slice_index_of_particle = np.zeros(bunch.n_macroparticles, dtype=np.int)

        for i in range(self.n_slices):
            self.slice_index_of_particle[self.first_particle_index_in_slice[i]:self.first_particle_index_in_slice[i+1]] = i


    def update_slices(self, bunch):

        if self.mode == 'const_charge':
            self._slice_constant_charge(bunch)
        elif self.mode == 'const_space':
            self._slice_constant_space(bunch)

        if  bunch.same_size_for_all_MPs:
            self.n_particles = self.n_macroparticles*bunch.n_particles_per_mp
        else:
            self.n_particles = 'Not yet implemented for non uniform set'

    '''
    Stats.
    '''
    def mean_x(self, bunch):
        return self._mean(bunch.x)

    def mean_xp(self, bunch):
        return self._mean(bunch.xp)

    def mean_y(self, bunch):
        return self._mean(bunch.y)

    def mean_yp(self, bunch):
        return self._mean(bunch.yp)

    def mean_z(self, bunch):
        return self._mean(bunch.z)

    def mean_dp(self, bunch):
        return self._mean(bunch.dp)

    def sigma_x(self, bunch):
        return self._sigma(bunch.x)

    def sigma_y(self, bunch):
        return self._sigma(bunch.y)

    def sigma_z(self, bunch):
        return self._sigma(bunch.z)

    def sigma_dp(self, bunch):
        return self._sigma(bunch.dp)

    def epsn_x(self, bunch):
        return self._epsn(bunch.x, bunch.xp, bunch.beta, bunch.gamma)

    def epsn_y(self, bunch):
        return self._epsn(bunch.y, bunch.yp, bunch.beta, bunch.gamma)

    def epsn_z(self, bunch):
        '''
        Approximate epsn_z. Correct for Gaussian bunch.
        '''
        return (4. * np.pi * self.sigma_z(bunch) * self.sigma_dp(bunch) * bunch.p0 / bunch.charge)


    '''
    Stats helper functions.
    '''
    def _mean(self, u):

        mean_u = np.zeros(self.n_slices)
        cp.mean_per_slice(self.slice_index_of_particle, self.particles_within_cuts,
                          self.n_macroparticles, u, mean_u)
        return mean_u


    def _sigma(self, u):

        sigma_u = np.zeros(self.n_slices)
        cp.std_per_slice(self.slice_index_of_particle, self.particles_within_cuts,
                         self.n_macroparticles, u, sigma_u)
        return sigma_u


    def _epsn(self, u, up):

        epsn_u = np.zeros(self.n_slices)
        cp.emittance_per_slice(self.slice_index_of_particle, self.particles_within_cuts,
                               self.n_macroparticles, u, up, epsn_u)
        return epsn_u
