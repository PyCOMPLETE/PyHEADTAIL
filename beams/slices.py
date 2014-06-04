'''
Created on 06.01.2014

@author: Kevin Li, Hannes Bartosik, Michael Schenk
'''


import numpy as np


from random import sample
import cobra_functions.stats as cp


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices, n_macroparticles, nsigmaz=None, mode='const_space', z_cuts=None):
        '''
        Constructor
        '''
        self.nsigmaz = nsigmaz
        self.mode = mode

        self.mean_x = np.zeros(n_slices)
        self.mean_xp = np.zeros(n_slices)
        self.mean_y = np.zeros(n_slices)
        self.mean_yp = np.zeros(n_slices)
        self.mean_z = np.zeros(n_slices)
        self.mean_dp = np.zeros(n_slices)
        self.sigma_x = np.zeros(n_slices)
        self.sigma_y = np.zeros(n_slices)
        self.sigma_z = np.zeros(n_slices)
        self.sigma_dp = np.zeros(n_slices)
        self.epsn_x = np.zeros(n_slices)
        self.epsn_y = np.zeros(n_slices)
        self.epsn_z = np.zeros(n_slices)

        self.slice_index_of_particle = np.zeros(n_macroparticles, dtype=np.int)

        # self.n_slices = n_slices
        # self.n_macroparticles = np.zeros(n_slices, dtype=int)
        # self.z_bins = np.zeros(n_slices + 1)
        # self.static_slices = False

        if z_cuts:
            z_cut_tail, z_cut_head = z_cuts
            self.z_bins = np.linspace(z_cut_tail, z_cut_head, self.n_slices + 1)
            self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.
            self.z_cut_tail, self.z_cut_head = z_cut_tail, z_cut_head

    @property
    def n_slices(self):
        return len(self.mean_x)

    def _set_longitudinal_cuts(self, bunch):

        if self.nsigmaz == None:
            z_cut_tail = bunch.z[0]
            z_cut_head = bunch.z[-1 - bunch.n_macroparticles_lost]
        else:
            mean_z = cp.mean(bunch.z[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            sigma_z = cp.std(bunch.z[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            z_cut_tail = mean_z - self.nsigmaz * sigma_z
            z_cut_head = mean_z + self.nsigmaz * sigma_z

        return z_cut_tail, z_cut_head

    # @profile
    def _slice_constant_space(self, bunch):
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
        self.z_index = first_index_in_bin[1:-1]

        # first_index_in_bin = np.searchsorted(bunch.z[:n_macroparticles_alive], self.z_bins)
        # self.z_index = first_index_in_bin

        # self.n_macroparticles = np.diff(first_index_in_bin)
        # print self.n_macroparticles

        self.n_macroparticles = np.diff(first_index_in_bin)[1:-1]

        # .in_slice indicates in which slice the particle is (needed for wakefields)
        self._set_slice_index_of_particle()
        # bunch.set_in_slice(index_after_bin_edges)
	

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

        self.n_macroparticles = (q0 // self.n_slices)*np.ones(self.n_slices)
        self.n_macroparticles[ix] += 1

        # 2. z-bins
        # Get indices of the particles defining the bin edges
        n_macroparticles_all = np.hstack((self.n_cut_tail, self.n_macroparticles, self.n_cut_head))
        first_index_in_bin = np.cumsum(n_macroparticles_all)
        self.z_index = first_index_in_bin[:-1]
        self.z_index = (self.z_index).astype(int)
        
        # print(self.z_index.shape)
        self.z_bins = (bunch.z[self.z_index - 1] + bunch.z[self.z_index]) / 2.
        self.z_bins[0], self.z_bins[-1] = z_cut_tail, z_cut_head
        self.z_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2.

        self._set_slice_index_of_particle()

        # # self.z_centers = map((lambda i: cp.mean(bunch.z[first_index_in_bin[i]:first_index_in_bin[i+1]])), np.arange(self.n_slices)

    def _set_slice_index_of_particle(self):
        for i in range(self.n_slices):
            self.slice_index_of_particle[self.z_index[i]:self.z_index[i+1]] = i

                    
    def update_slices(self, bunch):
        if self.mode == 'const_charge':
            self._slice_constant_charge(bunch)
        elif self.mode == 'const_space':
            self._slice_constant_space(bunch)

    # @profile
    def compute_statistics(self, bunch):

        index = self.n_cut_tail + np.cumsum(np.append(0, self.n_macroparticles))

        # # determine the start and end indices of each slices
        # i1 = np.append(np.cumsum(self.slices.n_macroparticles[:-2]), np.cumsum(self.slices.n_macroparticles[-2:]))
        # i0 = np.zeros(len(i1), dtype=np.int)
        # i0[1:] = i1[:-1]
        # i0[-2] = 0

        for i in xrange(self.n_slices):
            x  = bunch.x[index[i]:index[i + 1]]
            xp = bunch.xp[index[i]:index[i + 1]]
            y  = bunch.y[index[i]:index[i + 1]]
            yp = bunch.yp[index[i]:index[i + 1]]
            z  = bunch.z[index[i]:index[i + 1]]
            dp = bunch.dp[index[i]:index[i + 1]]

            self.mean_x[i] = cp.mean(x)
            self.mean_xp[i] = cp.mean(xp)
            self.mean_y[i] = cp.mean(y)
            self.mean_yp[i] = cp.mean(yp)
            self.mean_z[i] = cp.mean(z)
            self.mean_dp[i] = cp.mean(dp)

            self.sigma_x[i] = cp.std(x)
            self.sigma_y[i] = cp.std(y)
            self.sigma_z[i] = cp.std(z)
            self.sigma_dp[i] = cp.std(dp)

            self.epsn_x[i] = cp.emittance(x, xp) * bunch.gamma * bunch.beta * 1e6
            self.epsn_y[i] = cp.emittance(y, yp) * bunch.gamma * bunch.beta * 1e6
            self.epsn_z[i] = 4 * np.pi * self.sigma_z[i] * self.sigma_dp[i] * bunch.p0 / bunch.charge

    # def sort_particles(self, bunch):

    #     # update the number of lost particles
    #     bunch.n_macroparticles_lost = (bunch.n_macroparticles - np.count_nonzero(bunch.id))

    #     # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
    #     if bunch.n_macroparticles_lost:
    #         dz_argsorted = np.lexsort((bunch.z, -np.sign(bunch.id))) # place lost particles at the end of the array
    #     else:
    #         dz_argsorted = np.argsort(bunch.z)

    #     bunch.x = bunch.x.take(dz_argsorted)
    #     bunch.xp = bunch.xp.take(dz_argsorted)
    #     bunch.y = bunch.y.take(dz_argsorted)
    #     bunch.yp = bunch.yp.take(dz_argsorted)
    #     bunch.z = bunch.z.take(dz_argsorted)
    #     bunch.dp = bunch.dp.take(dz_argsorted)
    #     bunch.id = bunch.id.take(dz_argsorted)

    # def set_in_slice(self, index_after_bin_edges):

    #     self.in_slice = (self.slices.n_slices + 3) * np.ones(self.n_macroparticles, dtype=np.int)

    #     for i in xrange(self.slices.n_slices + 2):
    #         self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i


# import cobra_functions.stats as cp


# class Slices(object):
#     '''
#     classdocs
#     '''

#     def __init__(self, n_slices, nsigmaz=None, slicemode='cspace'):
#         '''
#         Constructor
#         '''
#         self.mean_x = np.zeros(n_slices + 4)
#         self.mean_xp = np.zeros(n_slices + 4)
#         self.mean_y = np.zeros(n_slices + 4)
#         self.mean_yp = np.zeros(n_slices + 4)
#         self.mean_dz = np.zeros(n_slices + 4)
#         self.mean_dp = np.zeros(n_slices + 4)
#         self.sigma_x = np.zeros(n_slices + 4)
#         self.sigma_y = np.zeros(n_slices + 4)
#         self.sigma_dz = np.zeros(n_slices + 4)
#         self.sigma_dp = np.zeros(n_slices + 4)
#         self.epsn_x = np.zeros(n_slices + 4)
#         self.epsn_y = np.zeros(n_slices + 4)
#         self.epsn_z = np.zeros(n_slices + 4)

#         self.n_macroparticles = np.zeros(n_slices + 4, dtype=int)
#         self.dz_centers = np.zeros(n_slices + 3)
#         self.dz_bins = np.zeros(n_slices + 3)

#         self.nsigmaz = nsigmaz
#         self.slicemode = slicemode
#         self.n_slices = n_slices


#     #~ @profile
#     def slice_constant_space(self, bunch, nsigmaz=None):

#         # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
#         bunch.sort_particles()

#         # determine the longitudinal cuts (this allows for the user defined static cuts: self.dz_cut_tail, self.dz_cut_head)
#         try:
#             dz_cut_tail, dz_cut_head = self.dz_cut_tail, self.dz_cut_head
#         except:
#             dz_cut_tail, dz_cut_head = self.determine_longitudinal_cuts(bunch, nsigmaz)

#         # First bins
#         dz_bins = np.zeros(self.n_slices + 3)
#         dz_bins[0] = np.min([bunch.dz[0], dz_cut_tail])
#         dz_bins[-1] = np.max([bunch.dz[- 1 - bunch.n_macroparticles_lost], dz_cut_head])
#         dz_bins[1:-1] = np.linspace(dz_cut_tail, dz_cut_head, self.n_slices + 1)
#         self.dz_centers[:-1] = dz_bins[:-1] + (dz_bins[1:] - dz_bins[:-1]) / 2.
#         self.dz_centers[-1] = self.mean_dz[-1]
#         index_after_bin_edges = np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], dz_bins)
#         index_after_bin_edges[np.where(dz_bins == bunch.dz[-1 - bunch.n_macroparticles_lost])] += 1
#         # Get n_macroparticles
#         self.n_macroparticles = np.diff(index_after_bin_edges)
#         self.n_macroparticles = np.concatenate((self.n_macroparticles, [bunch.n_macroparticles - bunch.n_macroparticles_lost], [bunch.n_macroparticles_lost]))

#         # .in_slice indicates in which slice the particle is (needed for wakefields)
#         bunch.set_in_slice(index_after_bin_edges)


#     def slice_constant_charge(self, bunch, nsigmaz=None):

#         # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
#         bunch.sort_particles()

#         # determine the longitudinal cuts
#         dz_cut_tail, dz_cut_head = self.determine_longitudinal_cuts(bunch, nsigmaz)

#         # First n_macroparticles
#         particles_in_left_cut = np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], dz_cut_tail)
#         particles_in_right_cut = bunch.n_macroparticles - bunch.n_macroparticles_lost - np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], dz_cut_head)
#         # set number of macro_particles in the slices that are cut (slice 0 and n_slices+1)
#         self.n_macroparticles[0] = particles_in_left_cut
#         self.n_macroparticles[-3] = particles_in_right_cut
#         # determine number of macroparticles used for slicing
#         q0 = bunch.n_macroparticles - bunch.n_macroparticles_lost - self.n_macroparticles[0] - self.n_macroparticles[-3]
#         # distribute macroparticles uniformly along slices
#         self.n_macroparticles[1:-3] = int(q0 / self.n_slices)
#         self.n_macroparticles[1:(q0 % self.n_slices + 1)] += 1
#         # number of macroparticles in full bunch slice and lost particles slice
#         self.n_macroparticles[-2:] =  bunch.n_macroparticles - bunch.n_macroparticles_lost, bunch.n_macroparticles_lost

#         # Get indices of the particles defining the bin edges
#         index_after_bin_edges = np.append(0, np.cumsum(self.n_macroparticles[:-2]))

#         # bin centers
#         self.dz_centers[:-1] = map((lambda i: cp.mean(bunch.dz[index_after_bin_edges[i]:index_after_bin_edges[i+1]])), np.arange(self.n_slices + 2))
#         self.dz_centers[-1] = cp.mean(bunch.dz)

#         # .in_slice indicates in which slice the particle is (needed for wakefields)
#         bunch.set_in_slice(index_after_bin_edges)


#     def determine_longitudinal_cuts(self, bunch, nsigmaz):

#         if nsigmaz == None:
#             dz_cut_tail = bunch.dz[0]
#             dz_cut_head = bunch.dz[-1 - bunch.n_macroparticles_lost]
#         else:
#             sigma_dz = cp.std(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
#             mean_dz = cp.mean(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
#             dz_cut_tail = -nsigmaz * sigma_dz + mean_dz
#             dz_cut_head = nsigmaz * sigma_dz + mean_dz

#         return dz_cut_tail, dz_cut_head
