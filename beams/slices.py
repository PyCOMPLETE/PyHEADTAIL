'''
Created on 06.01.2014

@authors: Kevin Li and Hannes Bartosik
'''


import numpy as np
'''http://docs.scipy.org/doc/numpy/reference/routines.html'''
import cobra_functions.cobra_functions as cp


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices):
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

#     def set_slice(self, bunch, n_slices, nsigmaz, mode):
# 
#         # Compute longitudinal moments
#         if not bunch.slices:
#             self.mean_dz[-1] = np.mean(bunch.dz)
#             self.sigma_dz[-1] = np.std(bunch.dz)
# 
#         # Sorting
#         self.index = np.argsort(bunch.dz)
# 
#         # Slicing
#         if mode == "cspace":
#             self.slice_constant_space(slices, nsigmaz)
#         elif mode == "ccharge":
#             self.slice_constant_charge(slices, nsigmaz)
# 
#         bunch.compute_statistics()

    def index(self, slice_number):

        i0 = sum(self.charge[:slice_number])
        i1 = i0 + self.charge[slice_number]

        index = self.dz_argsorted[i0:i1]

        return index

    # def dz(self, slice_number):

    #     return [0]

    #~ @profile
    def slice_constant_space(self, bunch, nsigmaz=None):

        n_particles = len(bunch.x)
        n_slices = len(self.mean_x) - 3
        #~ self.dz_argsorted = np.argsort(bunch.dz)

        if nsigmaz == None:
            cutleft = np.min(bunch.dz)
            cutright = np.max(bunch.dz)
        else:
            sigma_dz = cp.std(bunch.dz)
            cutleft = -nsigmaz * sigma_dz
            cutright = nsigmaz * sigma_dz

        #~ dz = (cutright - cutleft) / n_slices
        #~ self.dz_bins[0] = np.min(bunch.dz)
        #~ self.dz_bins[-1] = np.max(bunch.dz)
        #~ self.dz_bins[1:-1] = cutleft + np.arange(n_slices + 1) * dz
#~ 
        #~ self.dz_centers[:-1] = self.dz_bins[:-1] \
                          #~ + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        #~ self.dz_centers[-1] = self.mean_dz[-1]
        
        ##############
        #~ self.charge[0] = len(np.where(bunch.dz < cutleft)[0])
        #~ self.charge[-2] = len(np.where(bunch.dz >= cutright)[0])
#~ #         q0 = n_particles - self.charge[0] - self.charge[-2]
#~ #         self.charge[1:-2] = int(q0 / n_slices)
#~ #         self.charge[1:(q0 % n_slices + 1)] += 1
        #~ self.charge[1:-2] = [len(np.where(
                                #~ # can be tricky here when
                                #~ # cutright == self.dz_bins[i + 1] == bunch.dz
                                #~ (bunch.dz < self.dz_bins[i + 1])
                              #~ & (bunch.dz >= self.dz_bins[i])
                            #~ )[0]) for i in range(1, n_slices + 1)]
        #~ self.charge[-1] = sum(self.charge[:-1])
        ##############
        #~ # HB: faster version
        #~ self.in_slice = np.digitize(bunch.dz,self.dz_bins[1:-1],right=False)
        #~ self.charge = np.bincount(self.in_slice, minlength = n_slices+2)
        #~ self.charge = np.append(self.charge, sum(self.charge))
        #~ 
        #~ # test1 = map(lambda i: np.where(self.in_slice == i)[0], xrange(n_slices+2))
        #~ # test2 = np.where(self.in_slice == 1)[0]
        #~ # test3 = np.array(xrange(n_particles))[self.in_slice == 1]
#~ 
        #~ in_slice_argsorted = np.argsort(self.in_slice)      
        #~ # i1 = np.cumsum(self.charge[:-1])
        #~ # i0 = np.append(0, i1[:-1])
        #~ # self._slice_index = map(lambda j: self.in_slice_argsorted[i0[j]:i1[j]], xrange(n_slices+2))
        #~ # self._slice_index.append(np.array(self.in_slice_argsorted))
        #~ 
        #~ self.in_slice = self.in_slice[in_slice_argsorted]
        #~ bunch.x = bunch.x[in_slice_argsorted]
        #~ bunch.xp = bunch.xp[in_slice_argsorted]
        #~ bunch.y = bunch.y[in_slice_argsorted]
        #~ bunch.yp = bunch.yp[in_slice_argsorted]
        #~ bunch.dz = bunch.dz[in_slice_argsorted]
        #~ bunch.dp = bunch.dp[in_slice_argsorted]

        # HB: even faster version ################
        dz_argsorted = np.argsort(bunch.dz, kind='quickksort')
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
        self.in_slice = np.zeros(n_particles, dtype=np.int)
        for i in xrange(n_slices + 2):
            self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i 
                
        self.charge = np.diff(index_after_bin_edges)
        self.charge = np.append(self.charge, sum(self.charge))
        

        #~ self._slice_index = np.array(self._slice_index[:], np.array(1))
        #~ self._slice_index = [[] for i in range(n_slices+2)]
        #~ for j in xrange(n_slices+2):            
            #~ self._slice_index[j] = self.in_slice_argsorted[i0[j]:i1[j]]
        #~ print self._slice_index
        #~ print self._slice_index[1]
        #~ print self._slice_index[-1]
        
        #~ # further development: sort phase space according to dz       
        #~ phase_space = np.zeros(n_particles, dtype={'names':['x', 'y','in_slice'], 'formats':['f8','f8','i4']})
        #~ phase_space['x'] = bunch.x
        #~ phase_space['y'] = bunch.y
        #~ phase_space['in_slice'] = self.in_slice
        #~ print phase_space
        #~ phase_space.sort(axis=0,order='in_slice')
        #~ print phase_space
        #~ xtest = bunch.x[self.in_slice_argsorted]
        #~ for i in xrange(n_slices+2):
            #~ mask = phase_space['in_slice']==i
            #~ sub_phase_space = phase_space[mask]
            #~ x = sub_phase_space['x']
            #~ print x
            #~ print mask
            #~ print len(mask)
            #~ phase_space = phase_space[map(operator.not_, mask)]




    #~ @profile
    def slice_constant_charge(self, bunch, nsigmaz):

        n_particles = len(bunch.x)
        n_slices = len(self.mean_x) - 3
        
        ####################### 
        #~ self.dz_argsorted = np.argsort(bunch.dz)
#~ 
        #~ if nsigmaz == None:
            #~ cutleft = np.min(bunch.dz)
            #~ cutright = np.max(bunch.dz)
        #~ else:
            #~ sigma_dz = np.std(bunch.dz)
            #~ cutleft = -nsigmaz * sigma_dz
            #~ cutright = nsigmaz * sigma_dz
#~ 
        #~ self.charge[0] = len(np.where(bunch.dz < cutleft)[0])
        #~ self.charge[-2] = len(np.where(bunch.dz >= cutright)[0])
        #~ q0 = n_particles - self.charge[0] - self.charge[-2]
        #~ self.charge[1:-2] = int(q0 / n_slices)
        #~ self.charge[1:(q0 % n_slices + 1)] += 1
        #~ self.charge[-1] = sum(self.charge[:-1])
#~ 
        #~ self.dz_bins[0] = np.min(bunch.dz)
        #~ self.dz_bins[-1] = np.max(bunch.dz)
        #~ self.dz_bins[1:-1] = [bunch.dz[
                           #~ self.dz_argsorted[
                           #~ sum(self.charge[:(i + 1)])]]
                           #~ for i in np.arange(n_slices + 1)]
#~ 
        #~ self.dz_centers[:-1] = self.dz_bins[:-1] \
                          #~ + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.
        #~ self.dz_centers[-1] = self.mean_dz[-1]
        #######################
        # HB: faster version
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
        
        self.in_slice = np.zeros(n_particles, dtype=np.int)
        for i in xrange(n_slices + 2):
            self.in_slice[index_after_bin_edges[i]:index_after_bin_edges[i+1]] = i 


#     void slice_constant_charge(std::vector<Slice> slices, int nsigmaz)
#     {
#         int ns = get_nslices();
#         int np = get_nparticles();
#         std::vector<int> q(ns, 0);
#         q[0] = cut_front(slices, nsigmaz);
#         q[ns + 1] = cut_back(slices, nsigmaz);
#         q[ns + 2] = np;
#         int k = np - q[0] - q[ns + 1];
# 
#         for (int i=1; i<ns + 1; i++)
#             q[i] = k / ns;
#         for (int i=1; i<k % ns + 1; i++)
#             q[i] += 1;
# 
#         set_slice_positions(slices, q);
#         set_slice_indices(slices, q);
#     }
# 
#     self.cut_front(self, std::vector<Slice> slices, int nsigmaz)
#     {
#         int k = 0;
#         int ns = get_nslices();
#         int np = get_nparticles();
#         double mean_dz = this->mean_dz[ns + 2];
#         double sigma_dz = this->sigma_dz[ns + 2];
# 
#         while (slices[k].dz - mean_dz < -nsigmaz * sigma_dz)
#         {
#             k++;
#             if (k > np)
#                 std::cerr << "*** WARNING! All particles cut in cut_front()!"
#                           << std::endl;
#         }
# 
#         return k;
#     }
# 
#     int cut_back(std::vector<Slice> slices, int nsigmaz)
#     {
#         int k = 0;
#         int ns = get_nslices();
#         int np = get_nparticles();
#         double mean_dz = this->mean_dz[ns + 2];
#         double sigma_dz = this->sigma_dz[ns + 2];
# 
#         while (slices[k].dz - mean_dz >= nsigmaz * sigma_dz)
#         {
#             k++;
#             if (k > np)
#                 std::cerr << "*** WARNING! All particles cut in cut_back()!"
#                           << std::endl;
#         }
# 
#         return k;
#     }
# 
#     void set_slice_positions(std::vector<Slice> slices, std::vector<int> q)
#     {
#         int k = 0;
#         int ns = get_nslices();
#         int np = get_nparticles();
# 
#         slice_dz[0] = slices[0].dz;
#         for (int i=0; i<ns + 1; i++)
#         {
#             k += q[i];
#             slice_dz[i + 1] = 1 / 2. * (slices[k - 1].dz + slices[k].dz);
#         }
#         slice_dz[ns + 2] = slices[np - 1].dz;
#     }
# 
