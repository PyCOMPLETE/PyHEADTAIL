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
        self.dz_centers = np.zeros(n_slices + 2)

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
# 
#     def get_indices(self, slice_number):
# 
#         i0 = np.sum(slice_charge[:slice_number - 1])
#         i1 = np.sum(slice_charge[:slice_number])
# 
#         return indices[i0:i1]

    def index(self, slice_number):

        i0 = sum(self.charge[:slice_number])
        i1 = i0 + self.charge[slice_number]

        index = self.dz_argsorted[i0:i1]

        return index

    def dz(self, slice_number):

        return [0]

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

        dz = (cutright - cutleft) / n_slices
        self.dz_bins[0] = np.min(bunch.dz)
        self.dz_bins[-1] = np.max(bunch.dz)
        self.dz_bins[1:-1] = cutleft + np.arange(n_slices + 1) * dz

        self.dz_centers[:] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.

        self.charge[0] = len(np.where(bunch.dz < cutleft)[0])
        self.charge[-2] = len(np.where(bunch.dz >= cutright)[0])
#         q0 = n_particles - self.charge[0] - self.charge[-2]
#         self.charge[1:-2] = int(q0 / n_slices)
#         self.charge[1:(q0 % n_slices + 1)] += 1
        self.charge[1:-2] = [len(np.where(
                                # can be tricky here when
                                # cutright == self.dz_bins[i + 1] == bunch.dz
                                (bunch.dz < self.dz_bins[i + 1])
                              & (bunch.dz >= self.dz_bins[i])
                            )[0]) for i in range(1, n_slices + 1)]

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

        self.charge[0] = len(np.where(bunch.dz < cutleft)[0])
        self.charge[-2] = len(np.where(bunch.dz >= cutright)[0])
        q0 = n_particles - self.charge[0] - self.charge[-2]
        self.charge[1:-2] = int(q0 / n_slices)
        self.charge[1:(q0 % n_slices + 1)] += 1

        self.dz_bins[0] = np.min(bunch.dz)
        self.dz_bins[-1] = np.max(bunch.dz)
        self.dz_bins[1:-1] = [bunch.dz[
                           self.dz_argsorted[
                           sum(self.charge[:(i + 1)])]]
                           for i in np.arange(n_slices + 1)]

        self.dz_centers[:] = self.dz_bins[:-1] \
                          + (self.dz_bins[1:] - self.dz_bins[:-1]) / 2.

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
#     void set_slice_indices(std::vector<Slice> slices, std::vector<int> q)
#     {
#         int k = 0;
#         int ns = get_nslices();
#         int np = get_nparticles();
# 
#         for (size_t i=0; i<ns + 2; i++)
#         {
#             slice_index[i].resize(q[i]);
#             for (int j=0; j<q[i]; j++)
#                 slice_index[i][j] = slices[k + j].ix;
#             k += q[i];
#         }
#         slice_index[ns + 2].resize(np);
#         for (size_t i=0; i<np; i++)
#             slice_index[ns + 2][i] = i;
#     }
