'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


from beams.slices import *
from configuration import *
import cobra_functions.cobra_functions as cp


class Bunch(object):
    '''
    classdocs
    '''

    def __init__(self, x, xp, y, yp, dz, dp):
        '''
        Constructor
        '''
        self.x = x
        self.xp = xp
        self.y = y
        self.yp = yp
        self.dz = dz
        self.dp = dp

    @classmethod
    def default(cls, n_particles):

        x = np.zeros(n_particles)
        xp = np.zeros(n_particles)
        y = np.zeros(n_particles)
        yp = np.zeros(n_particles)
        dz = np.zeros(n_particles)
        dp = np.zeros(n_particles)

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @classmethod
    def from_copy(cls, x, xp, y, yp, dz, dp):

        x = np.copy(x)
        xp = np.copy(xp)
        y = np.copy(y)
        yp = np.copy(yp)
        dz = np.copy(dz)
        dp = np.copy(dp)

        self = cls(x, xp, y, yp, dz, dp)

        return self

    @classmethod
    def from_file(cls):
        pass

    @classmethod
    def from_parameters(cls, n_particles, charge, energy, intensity, mass,
                        epsn_x, beta_x, epsn_y, beta_y, epsn_z, length):

        x = np.random.randn(n_particles)
        xp = np.random.randn(n_particles)
        y = np.random.randn(n_particles)
        yp = np.random.randn(n_particles)
        dz = np.random.randn(n_particles)
        dp = np.random.randn(n_particles)

        self = cls(x, xp, y, yp, dz, dp)

        self.match_distribution(charge, energy, mass,
                                epsn_x, beta_x, epsn_y, beta_y, epsn_z, length)

        return self

    def match_distribution(self, charge, energy, mass,
                           epsn_x, beta_x, epsn_y, beta_y, epsn_z, length):

        self.charge = charge
        self.gamma = energy * 1e9 * charge * e / (mass * c ** 2) + 1
        self.beta = np.sqrt(1 - 1 / self.gamma ** 2)
        self.mass = mass
        p0 = mass * self.gamma * self.beta * c / e

        sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (self.gamma * self.beta))
        sigma_xp = sigma_x / beta_x
        sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (self.gamma * self.beta))
        sigma_yp = sigma_y / beta_y
        sigma_dz = length
        sigma_dp = epsn_z / (4 * np.pi * sigma_dz) / p0

        self.x *= sigma_x
        self.xp *= sigma_xp
        self.y *= sigma_y
        self.yp *= sigma_yp
        self.dz *= sigma_dz
        self.dp *= sigma_dp

    # @profile
    def compute_statistics(self):

        if not hasattr(self, 'slices'):
            print "*** WARNING: bunch not yet sliced! Aborting..."
            sys.exit(-1)
        else:
            n_particles = len(self.x)

            indices = [self.slices.index(i) for i in range(n_slices + 2)]
            indices.append(range(n_particles))

#             indices = [np.copy(self.slices.index(i))
#                        for i in range(n_slices + 2)]
#             indices.append(np.arange(n_particles))
#             for i in range(len(indices)):
#                 indices[i].resize(n_particles)
#             indices = np.vstack(indices)
# 
# 
#         self.slices.mean_x = np.mean(self.x[indices], axis=1)
#         self.slices.mean_xp = np.mean(self.xp[indices], axis=1)
#         self.slices.mean_y = np.mean(self.y[indices], axis=1)
#         self.slices.mean_yp = np.mean(self.yp[indices], axis=1)
# #                 self.slices.mean_dz[i] = np.mean(self.dz[k])
# #                 self.slices.mean_dp[i] = np.mean(self.dp[k])
# 
# #                 self.slices.sigma_x[i] = np.std(self.x[k])
# #                 self.slices.sigma_y[i] = np.std(self.y[k])
# #                 self.slices.sigma_dz[i] = np.std(self.dz[k])
# #                 self.slices.sigma_dp[i] = np.std(self.dp[k])
# 
#         stdx2 = np.std(self.x[indices] ** 2, axis=1)
#         stdxp2 = np.std(self.xp[indices] ** 2, axis=1)
#         stdxxp = np.std(self.x[indices] * self.xp[indices], axis=1)
#         stdy2 = np.std(self.y[indices] ** 2, axis=1)
#         stdyp2 = np.std(self.yp[indices] ** 2, axis=1)
#         stdyyp = np.std(self.y[indices] * self.yp[indices], axis=1)
# 
#         self.slices.epsn_x = np.sqrt(stdx2 * stdxp2 - stdxxp ** 2) \
#                            * self.gamma * self.beta * 1e6
#         self.slices.epsn_y = np.sqrt(stdy2 * stdyp2 - stdyyp ** 2) \
#                            * self.gamma * self.beta * 1e6
# #                 self.slices.epsn_z[i] = 4 * np.pi \
# #                         * self.slices.sigma_dz[i] * self.slices.sigma_dp[i] \
# #                         * self.mass * self.gamma * self.beta * c / e


        for i in xrange(n_slices + 3):
            n = len(indices[i])
            if n:
                x = self.x[indices[i]]
                xp = self.xp[indices[i]]
                y = self.y[indices[i]]
                yp = self.yp[indices[i]]
                dz = self.dz[indices[i]]
                dp = self.dp[indices[i]]

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

    def slice(self, n_slices, nsigmaz, mode):

        if not hasattr(self, 'slices'):
            self.slices = Slices(n_slices)

        if mode == 'ccharge':
            self.slices.slice_constant_charge(self, nsigmaz)
        elif mode == 'cspace':
            self.slices.slice_constant_space(self, nsigmaz)
        else:
            print '*** ERROR! Unknown mode '+mode+'! Aborting...'
            sys.exit(-1)

#     def set_slice(self, n_slices):
#         
#         if not self.slices:
#             self.slices = Slices(n_slices)
# 
#         int np = get_nparticles();
#         int ns = get_nslices();
#     
#         // Allocate memory
#         if (ns < n_slices)
#         {
#             create_slices(n_slices);
#             ns = get_nslices();
#         }
#     
#         // Compute longitudinal moments
#         if (!is_sliced)
#         {
#             std::vector<int> index(np);
#             for (int i=0; i<np; i++)
#                 index[i] = i;
#     
#             this->mean_dz[ns + 2] = compute_first_moment(dz, index);
#             this->sigma_dz[ns + 2] = compute_second_moment(dz, mean_dz[ns + 2],
#                                                            index);
#         }
#     
#         // Sorting
#         std::vector<Slice> slices;
#         for (int i=0; i<np; i++)
#             slices.push_back(Slice(i, dz[i]));
#         std::sort(slices.begin(), slices.end());
#     
#         // Slicing
#         if (mode == "cspace")
#             slice_constant_space(slices, nsigmaz);
#         else if (mode == "ccharge")
#             slice_constant_charge(slices, nsigmaz);
#     
#         if (!is_sliced)
#             is_sliced = 1;
#     
#         self.compute_statistics()
