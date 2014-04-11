'''
Created on 06.01.2014

@author: Kevin Li, Hannes Bartosik
'''


import numpy as np


import cobra_functions.stats as cp


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

        self.n_macroparticles = np.zeros(n_slices + 4, dtype=int)
        self.dz_centers = np.zeros(n_slices + 3)

        self.nsigmaz = nsigmaz
        self.slicemode = slicemode
        self.n_slices = n_slices


    #~ @profile
    def slice_constant_space(self, bunch, nsigmaz=None):

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)        
        bunch.sort_particles() 
        
        # determine the longitudinal cuts (this allows for the user defined static cuts: self.dz_cut_tail, self.dz_cut_head)
        try:
            dz_cut_tail, dz_cut_head = self.dz_cut_tail, self.dz_cut_head
        except:
            dz_cut_tail, dz_cut_head = self.determine_longitudinal_cuts(bunch, nsigmaz)

        # First bins
        dz_bins = np.zeros(self.n_slices + 3)
        dz_bins[0] = np.min([bunch.dz[0], dz_cut_tail])
        dz_bins[-1] = np.max([bunch.dz[- 1 - bunch.n_macroparticles_lost], dz_cut_head])
        dz_bins[1:-1] = np.linspace(dz_cut_tail, dz_cut_head, self.n_slices + 1)
        self.dz_centers[:-1] = dz_bins[:-1] + (dz_bins[1:] - dz_bins[:-1]) / 2.
        self.dz_centers[-1] = self.mean_dz[-1]
        index_after_bin_edges = np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], dz_bins)  
        index_after_bin_edges[np.where(dz_bins == bunch.dz[-1 - bunch.n_macroparticles_lost])] += 1
        # Get n_macroparticles
        self.n_macroparticles = np.diff(index_after_bin_edges)
        self.n_macroparticles = np.concatenate((self.n_macroparticles, [bunch.n_macroparticles - bunch.n_macroparticles_lost], [bunch.n_macroparticles_lost]))

        # .in_slice indicates in which slice the particle is (needed for wakefields)     
        bunch.set_in_slice(index_after_bin_edges)


    def slice_constant_charge(self, bunch, nsigmaz=None):
       
        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)        
        bunch.sort_particles() 
        
        # determine the longitudinal cuts        
        dz_cut_tail, dz_cut_head = self.determine_longitudinal_cuts(bunch, nsigmaz)

        # First n_macroparticles
        particles_in_left_cut = np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], dz_cut_tail)
        particles_in_right_cut = bunch.n_macroparticles - bunch.n_macroparticles_lost - np.searchsorted(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost], dz_cut_head)
        # set number of macro_particles in the slices that are cut (slice 0 and n_slices+1)
        self.n_macroparticles[0] = particles_in_left_cut
        self.n_macroparticles[-3] = particles_in_right_cut  
        # determine number of macroparticles used for slicing
        q0 = bunch.n_macroparticles - bunch.n_macroparticles_lost - self.n_macroparticles[0] - self.n_macroparticles[-3]
        # distribute macroparticles uniformly along slices
        self.n_macroparticles[1:-3] = int(q0 / self.n_slices)
        self.n_macroparticles[1:(q0 % self.n_slices + 1)] += 1
        # number of macroparticles in full bunch slice and lost particles slice
        self.n_macroparticles[-2:] =  bunch.n_macroparticles - bunch.n_macroparticles_lost, bunch.n_macroparticles_lost

        # Get indices of the particles defining the bin edges
        index_after_bin_edges = np.append(0, np.cumsum(self.n_macroparticles[:-2]))

        # bin centers
        self.dz_centers[:-1] = map((lambda i: cp.mean(bunch.dz[index_after_bin_edges[i]:index_after_bin_edges[i+1]])), np.arange(self.n_slices + 2))  
        self.dz_centers[-1] = cp.mean(bunch.dz)
        
        # .in_slice indicates in which slice the particle is (needed for wakefields)     
        bunch.set_in_slice(index_after_bin_edges)


    def determine_longitudinal_cuts(self, bunch, nsigmaz):
        if nsigmaz == None:
            dz_cut_tail = bunch.dz[0]
            dz_cut_head = bunch.dz[-1 - bunch.n_macroparticles_lost]
        else:
            sigma_dz = cp.std(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            mean_dz = cp.mean(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            dz_cut_tail = -nsigmaz * sigma_dz + mean_dz
            dz_cut_head = nsigmaz * sigma_dz + mean_dz
        return dz_cut_tail, dz_cut_head

