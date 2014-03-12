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
        # bug: self.n_slices
        self.n_slices = n_slices

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


'''
class Slices(object):
    ''
    classdocs
    ''

    def __init__(self, n_slices):
        ''
        Constructor
        ''
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


    def index(self, slice_number):

        i0 = sum(self.charge[:slice_number])
        i1 = i0 + self.charge[slice_number]

        index = self.dz_argsorted[i0:i1]

        return index

    #~ @profile
    def slice_constant_space(self, bunch, nsigmaz=None):

        n_particles = len(bunch.x)
        n_slices = len(self.mean_x) - 3

        if nsigmaz == None:
            cutleft = np.min(bunch.dz)
            cutright = np.max(bunch.dz)
        else:
            sigma_dz = cp.std(bunch.dz)
            cutleft = -nsigmaz * sigma_dz
            cutright = nsigmaz * sigma_dz

        # sort particles according to dz (this is needed for efficiency of bunch.copmutate_statistics)
        dz_argsorted = np.argsort(bunch.dz)
        bunch.x = bunch.x[dz_argsorted]
        bunch.xp = bunch.xp[dz_argsorted]
        bunch.y = bunch.y[dz_argsorted]
        bunch.yp = bunch.yp[dz_argsorted]
        bunch.dz = bunch.dz[dz_argsorted]
        bunch.dp = bunch.dp[dz_argsorted]
        
        dz = (cutright - cutleft) / n_slices
        self.dz_bins[0] = bunch.dz[0]
        self.dz_bins[-1] = bunch.dz[-1]
        self.dz_bins[1:-1] = cutleft + np.arange(n_slices + 1) * dz

        self.dz_centers[:-1] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        self.dz_centers[-1] = self.mean_dz[-1]
        
        index_after_bin_edges = np.searchsorted(bunch.dz,self.dz_bins)
        index_after_bin_edges[-1] += 1   
        
        # .in_slice indicates in which slice the particle is (needed for wakefields)     
        self.in_slice = np.zeros(n_particles, dtype=np.int)
        for i in xrange(n_slices + 2):
            self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i 
                
        self.charge = np.diff(index_after_bin_edges)
        self.charge = np.append(self.charge, sum(self.charge))
        

    #~ @profile
    def slice_constant_charge(self, bunch, nsigmaz):

        n_particles = len(bunch.x)
        n_slices = len(self.mean_x) - 3
        
        # sort particles according to dz (this is needed for efficiency of bunch.copmutate_statistics)
        dz_argsorted = np.argsort(bunch.dz)
        bunch.x = bunch.x[dz_argsorted]
        bunch.xp = bunch.xp[dz_argsorted]
        bunch.y = bunch.y[dz_argsorted]
        bunch.yp = bunch.yp[dz_argsorted]
        bunch.dz = bunch.dz[dz_argsorted]
        bunch.dp = bunch.dp[dz_argsorted]
        
        if nsigmaz == None:
            cutleft = np.min(bunch.dz)
            cutright = np.max(bunch.dz)
        else:
            sigma_dz = cp.std(bunch.dz)
            cutleft = -nsigmaz * sigma_dz
            cutright = nsigmaz * sigma_dz

        self.charge[0] = np.searchsorted(bunch.dz,cutleft)
        self.charge[-2] = n_particles - np.searchsorted(bunch.dz,cutright)
        q0 = n_particles - self.charge[0] - self.charge[-2]
        self.charge[1:-2] = int(q0 / n_slices)
        self.charge[1:(q0 % n_slices + 1)] += 1
        self.charge[-1] = sum(self.charge[:-1])

        index_after_bin_edges = np.append(0, np.cumsum(self.charge[:-1]))
        self.dz_bins[-1] = bunch.dz[-1]
        self.dz_bins[:-1] = bunch.dz[index_after_bin_edges[:-1]]                 
        self.dz_centers[:-1] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        self.dz_centers[-1] = self.mean_dz[-1]
        
        # .in_slice indicates in which slice the particle is (needed for wakefields) 
        self.in_slice = np.zeros(n_particles, dtype=np.int)
        for i in xrange(n_slices + 2):
            self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i 
'''


