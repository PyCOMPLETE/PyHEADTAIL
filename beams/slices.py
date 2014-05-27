'''
Created on 06.01.2014

@author: Kevin Li, Hannes Bartosik
'''


import numpy as np


from random import sample
import cobra_functions.stats as cp


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices, nsigmaz=None, mode='const_space', z_cuts=None):
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

