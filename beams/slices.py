'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np
'''http://docs.scipy.org/doc/numpy/reference/routines.html'''


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices, nsigmaz=None, slicemode='cspace'):
        '''
        Constructor
        '''
        self.mean_x = np.zeros(n_slices + 3)
        self.mean_xp = np.zeros(n_slices + 3)
        self.mean_y = np.zeros(n_slices + 3)
        self.mean_yp = np.zeros(n_slices + 3)
        self.mean_dz = np.zeros(n_slices + 3)
        self.mean_dp = np.zeros(n_slices + 3)
        self.sigma_x = np.zeros(n_slices + 3)
        self.sigma_y = np.zeros(n_slices + 3)
        self.sigma_dz = np.zeros(n_slices + 3)
        self.sigma_dp = np.zeros(n_slices + 3)
        self.epsn_x = np.zeros(n_slices + 3)
        self.epsn_y = np.zeros(n_slices + 3)
        self.epsn_z = np.zeros(n_slices + 3)

        self.charge = np.zeros(n_slices + 3, dtype=int)
        self.dz_bins = np.zeros(n_slices + 3)
        self.dz_centers = np.zeros(n_slices + 3)

        self.nsigmaz = nsigmaz
        self.slicemode = slicemode

    def index(self, slice_number):

        i0 = sum(self.charge[:slice_number])
        i1 = i0 + self.charge[slice_number]

        index = self.dz_argsorted[i0:i1]

        return index

    def slice_constant_space(self, bunch, nsigmaz=None):

        n_particles = len(bunch.x)
        n_slices = len(self.mean_x) - 3
        self.dz_argsorted = np.argsort(bunch.dz)

        sigma_dz = np.std(bunch.dz)
        if nsigmaz == None:
            cutleft = np.min(bunch.dz)
            cutright = np.max(bunch.dz)
        else:
            cutleft = -nsigmaz * sigma_dz
            cutright = nsigmaz * sigma_dz

        # First bins
        self.dz_bins[0] = np.min(bunch.dz)
        self.dz_bins[-1] = np.max(bunch.dz)
        dz = (cutright - cutleft) / n_slices
        self.dz_bins[1:-1] = cutleft + np.arange(n_slices + 1) * dz

        self.dz_centers[:-1] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        self.dz_centers[-1] = self.mean_dz[-1]

        # Get charge
        self.charge[0] = len(np.where(bunch.dz < cutleft)[0])
        self.charge[-2] = len(np.where(bunch.dz >= cutright)[0])
        self.charge[1:-2] = [len(np.where(
                                # can be tricky here when
                                # cutright == self.dz_bins[i + 1] == bunch.dz
                                (bunch.dz < self.dz_bins[i + 1])
                              & (bunch.dz >= self.dz_bins[i])
                            )[0]) for i in range(1, n_slices + 1)]
        self.charge[-1] = sum(self.charge[:-1])

    def slice_constant_charge(self, bunch, nsigmaz):

        n_particles = len(bunch.x)
        n_slices = len(self.mean_x) - 3
        self.dz_argsorted = np.argsort(bunch.dz)

        sigma_dz = np.std(bunch.dz)
        if nsigmaz == None:
            cutleft = np.min(bunch.dz)
            cutright = np.max(bunch.dz)
        else:
            cutleft = -nsigmaz * sigma_dz
            cutright = nsigmaz * sigma_dz

        # First charge
        self.charge[0] = len(np.where(bunch.dz < cutleft)[0])
        self.charge[-2] = len(np.where(bunch.dz >= cutright)[0])
        q0 = n_particles - self.charge[0] - self.charge[-2]
        self.charge[1:-2] = int(q0 / n_slices)
        self.charge[1:(q0 % n_slices + 1)] += 1
        self.charge[-1] = sum(self.charge[:-1])

        # Get bins
        self.dz_bins[0] = np.min(bunch.dz)
        self.dz_bins[-1] = np.max(bunch.dz)
        self.dz_bins[1:-1] = [bunch.dz[
                           self.dz_argsorted[
                           sum(self.charge[:(i + 1)])]]
                           for i in np.arange(n_slices + 1)]

        self.dz_centers[:-1] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        self.dz_centers[-1] = self.mean_dz[-1]
