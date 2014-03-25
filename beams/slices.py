'''
Created on 06.01.2014

@author: Kevin Li, Hannes Bartosik
'''


import numpy as np
'''http://docs.scipy.org/doc/numpy/reference/routines.html'''
import cobra_functions.cobra_functions as cp


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices, nsigmaz=None, slicemode='cspace'):
        '''
        Constructor
        '''
        self.mean_x = np.zeros(n_slices + 4)
        self.mean_xp = np.zeros(n_slices + 4)
        self.mean_y = np.zeros(n_slices + 4)
        self.mean_yp = np.zeros(n_slices + 4)
        self.mean_dz = np.zeros(n_slices + 4)
        self.mean_dp = np.zeros(n_slices + 4)
        self.sigma_x = np.zeros(n_slices + 4)
        self.sigma_y = np.zeros(n_slices + 4)
        self.sigma_dz = np.zeros(n_slices + 4)
        self.sigma_dp = np.zeros(n_slices + 4)
        self.epsn_x = np.zeros(n_slices + 4)
        self.epsn_y = np.zeros(n_slices + 4)
        self.epsn_z = np.zeros(n_slices + 4)

        self.charge = np.zeros(n_slices + 4, dtype=int)
        self.dz_bins = np.zeros(n_slices + 3)
        self.dz_centers = np.zeros(n_slices + 3)

        self.nsigmaz = nsigmaz
        self.slicemode = slicemode
        self.n_slices = n_slices


    #~ @profile
    def slice_constant_space(self, bunch, nsigmaz=None):

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)        
        bunch.sort_particles() 
        
        # determine the longitudinal cuts        
        cutleft, cutright = self.determine_longitudinal_cuts(bunch, nsigmaz)
            
        # First bins
        self.dz_bins[0] = bunch.dz[0]
        self.dz_bins[-1] = bunch.dz[- 1 - bunch.n_particles_lost]
        dz = (cutright - cutleft) / self.n_slices
        self.dz_bins[1:-1] = cutleft + np.arange(self.n_slices + 1) * dz        
        self.dz_centers[:-1] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        self.dz_centers[-1] = self.mean_dz[-1]
        index_after_bin_edges = np.searchsorted(bunch.dz[:-bunch.n_particles_lost-1],self.dz_bins)
        index_after_bin_edges[-1] += 1  
        
        # Get charge
        self.charge = np.diff(index_after_bin_edges)
        self.charge = np.concatenate((self.charge, [bunch.n_particles - bunch.n_particles_lost], [bunch.n_particles_lost]))
        
        # .in_slice indicates in which slice the particle is (needed for wakefields)     
        bunch.set_in_slice(index_after_bin_edges)


    def slice_constant_charge(self, bunch, nsigmaz=None):
       
        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)        
        bunch.sort_particles() 
        
        # determine the longitudinal cuts        
        cutleft, cutright = self.determine_longitudinal_cuts(bunch, nsigmaz)

        # First charge
        self.charge[0] = np.searchsorted(bunch.dz[:-1 - bunch.n_particles_lost],cutleft)
        self.charge[-3] = bunch.n_particles - bunch.n_particles_lost - np.searchsorted(bunch.dz[:-1 - bunch.n_particles_lost],cutright)
        q0 = bunch.n_particles - bunch.n_particles_lost - self.charge[0] - self.charge[-3]
        self.charge[1:-3] = int(q0 / self.n_slices)
        self.charge[1:(q0 % self.n_slices + 1)] += 1
        self.charge[-2:] =  bunch.n_particles - bunch.n_particles_lost, bunch.n_particles_lost

        # Get bins
        index_after_bin_edges = np.append(0, np.cumsum(self.charge[:-2]))
        self.dz_bins[-1] = bunch.dz[-1 - bunch.n_particles_lost]
        self.dz_bins[:-1] = bunch.dz[index_after_bin_edges[:-1]] 
        self.dz_centers[:-1] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        self.dz_centers[-1] = self.mean_dz[-1]
        
        # .in_slice indicates in which slice the particle is (needed for wakefields)     
        bunch.set_in_slice(index_after_bin_edges)


    def determine_longitudinal_cuts(self, bunch, nsigmaz):
        if nsigmaz == None:
            cutleft = bunch.dz[0]
            cutright = bunch.dz[-1 - bunch.n_particles_lost]
        else:
            sigma_dz = cp.std(bunch.dz[:bunch.n_particles - bunch.n_particles_lost])
            cutleft = -nsigmaz * sigma_dz
            cutright = nsigmaz * sigma_dz
        return cutleft, cutright

