'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


import copy, h5py, sys
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi

# from slices import *
# from solvers.poissonfft import *


# re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
# rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p


class PDF(object):
    '''
    Particle distribution functions (probability density functions); this class provides a standard set of normalized 6d phase space distributions.
    '''

    def _create_empty(self, n_macroparticles):
        '''
        Allocate the memory to store the 6d phase space for n_macroparticles macroparticles.
        '''
        self.x = np.zeros(n_macroparticles)
        self.xp = np.zeros(n_macroparticles)
        self.y = np.zeros(n_macroparticles)
        self.yp = np.zeros(n_macroparticles)
        self.z = np.zeros(n_macroparticles)
        self.dp = np.zeros(n_macroparticles)

    def _create_gaussian(self, n_macroparticles):
        '''
        Create a normalized gaussian 6d phase space distribution for n_macroparticles macroparticles with mean = 0 and sigma = 1 in all dimensions.
        '''
        self.x = np.random.randn(n_macroparticles)
        self.xp = np.random.randn(n_macroparticles)
        self.y = np.random.randn(n_macroparticles)
        self.yp = np.random.randn(n_macroparticles)
        self.z = np.random.randn(n_macroparticles)
        self.dp = np.random.randn(n_macroparticles)

    def _create_uniform(self, n_macroparticles):
        '''
        Create a normalized uniform 6d phase space distribution for n_macroparticles macroparticles from -1 to +1 in all dimensions.
        '''
        self.x = 2 * np.random.rand(n_macroparticles) - 1
        self.xp = 2 * np.random.rand(n_macroparticles) - 1
        self.y = 2 * np.random.rand(n_macroparticles) - 1
        self.yp = 2 * np.random.rand(n_macroparticles) - 1
        self.z = 2 * np.random.rand(n_macroparticles) - 1
        self.dp = 2 * np.random.rand(n_macroparticles) - 1


class Bunch(PDF):

    def __init__(self, n_macroparticles, charge, gamma, intensity, mass,
                 alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, beta_z, sigma_z=None, epsn_z=None, match=None):

        self.charge = charge
        self.gamma = gamma
        self.intensity = intensity
        self.mass = mass

        self._create_gaussian(n_macroparticles)
        self._match_simple_gaussian_transverse(alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y)
        self._match_simple_gaussian_longitudinal(beta_z, sigma_z, epsn_z)

        self.id = np.arange(1, n_macroparticles + 1, dtype=int)

    def _match_simple_gaussian_transverse(self, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y):

        sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (self.gamma * self.beta))
        sigma_xp = sigma_x / beta_x
        sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (self.gamma * self.beta))
        sigma_yp = sigma_y / beta_y

        self.x *= sigma_x
        self.xp *= sigma_xp
        self.y *= sigma_y
        self.yp *= sigma_yp

    def _match_simple_gaussian_longitudinal(self, beta_z, sigma_z=None, epsn_z=None):

        if sigma_z and epsn_z:
            sigma_dp = epsn_z / (4 * np.pi * sigma_z) * e / self.p0
            if sigma_z / sigma_dp != beta_z:
                print '*** WARNING: beam mismatched in bucket. Set synchrotron tune as to obtain beta_z = ', sigma_z / sigma_dp
        elif not sigma_z and epsn_z:
            sigma_z = np.sqrt(beta_z * epsn_z / (4 * np.pi) * e / self.p0)
            sigma_dp = sigma_z / beta_z
        else:
            sigma_dp = sigma_z / beta_z

        self.z *= sigma_z
        self.dp *= sigma_dp

    @property
    def n_macroparticles(self):
        return len(self.x)

    @property
    def beta(self):
        return np.sqrt(1 - 1. / self.gamma ** 2)
    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1 - value ** 2)

    @property
    def p0(self):
        return self.mass * self.gamma * self.beta * c
    @p0.setter
    def p0(self, value):
        self.gamma = value / (self.mass * self.beta * c)

    # #~ @profile
    def sort_particles(self):
        # update the number of lost particles
        self.n_macroparticles_lost = (self.n_macroparticles - np.count_nonzero(self.id))

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        if self.n_macroparticles_lost:
            z_argsorted = np.lexsort((self.z, -np.sign(self.id))) # place lost particles at the end of the array
        else:
            z_argsorted = np.argsort(self.z)

        self.x = self.x.take(z_argsorted)
        self.xp = self.xp.take(z_argsorted)
        self.y = self.y.take(z_argsorted)
        self.yp = self.yp.take(z_argsorted)
        self.z = self.z.take(z_argsorted)
        self.dp = self.dp.take(z_argsorted)
        self.id = self.id.take(z_argsorted)


class Cloud(PDF):

    def __init__(self, n_macroparticles, density, extent_x, extent_y, extent_z):

        self.charge = e
        self.gamma = 1
        self.intensity = density * extent_x * extent_y * extent_z
        self.mass = m_e

        self._create_uniform(n_macroparticles)
        self._match_uniform(extent_x, extent_y, extent_z)

        # Initial distribution
        self.x0 = self.x.copy()
        self.xp0 = self.xp.copy()
        self.y0 = self.y.copy()
        self.yp0 = self.yp.copy()
        self.z0 = self.z.copy()
        self.dp0 = self.dp.copy()

    def _match_uniform(self, extent_x, extent_y, extent_z):

        self.x *= extent_x
        self.xp *= 0
        self.y *= extent_y
        self.yp *= 0
        self.z *= extent_z
        self.dp *= 0

    @property
    def n_macroparticles(self):

        return len(self.x)

    def reinit(self):

        np.copyto(self.x, self.x0)
        np.copyto(self.xp, self.xp0)
        np.copyto(self.y, self.y0)
        np.copyto(self.yp, self.yp0)
        np.copyto(self.z, self.z0)
        np.copyto(self.dp, self.dp0)


class Ghost(PDF):
    '''
    The ghost class represents a particle ensemble that moves along with its parent ensemble,
    interacting with all fields excited by the parent ensemble but without exciting any fields
    itself. It is used for diagnostics purposes to uniformly sample the region around the beam
    to measure fields, detuning etc. The constructor would most likely have to take the parent
    ensemble as argument; there would be a similar relationship as with the Slices class. The
    parent ensemble has no knowledge about the ghost, but the ghost knows about its parent ensemble.
    '''
    pass




# class Beam(object):

#     #     _set_beam_quality(charge, gamma, intensity, mass)
#     #     _set_beam_geometry(alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, beta_z, sigma_z, epsn_z)

#     @classmethod
#     def as_bunch(cls, n_macroparticles, charge, gamma, intensity, mass,
#                  alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, beta_z, sigma_z=None, epsn_z=None):

#         self = cls()

#         self._create_gauss(n_macroparticles)
#         self.id = np.arange(1, n_macroparticles + 1, dtype=int)

#         # General
#         self.charge = charge
#         self.gamma = gamma
#         self.intensity = intensity
#         self.mass = mass

#         # Transverse
#         sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (gamma * self.beta))
#         sigma_xp = sigma_x / beta_x
#         sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (gamma * self.beta))
#         sigma_yp = sigma_y / beta_y

#         self.x *= sigma_x
#         self.xp *= sigma_xp
#         self.y *= sigma_y
#         self.yp *= sigma_yp

#         # Longitudinal
#         # Assuming a gaussian-type stationary distribution: beta_z = eta * circumference / (2 * np.pi * Qs)
#         if sigma_z and epsn_z:
#             sigma_dp = epsn_z / (4 * np.pi * sigma_z) * e / self.p0
#             if sigma_z / sigma_dp != beta_z:
#                 print '*** WARNING: beam mismatched in bucket. Set synchrotron tune to obtain beta_z = ', sigma_z / sigma_dp
#         elif not sigma_z and epsn_z:
#             sigma_z = np.sqrt(beta_z * epsn_z / (4 * np.pi) * e / self.p0)
#             sigma_dp = sigma_z / beta_z
#         else:
#             sigma_dp = sigma_z / beta_z

#         self.z *= sigma_z
#         self.dp *= sigma_dp

#         return self

#     @classmethod
#     def as_cloud(cls, n_macroparticles, density, extent_x, extent_y, extent_z):

#         self = cls()

#         self._create_uniform(n_macroparticles)

#         # General
#         self.charge = e
#         self.gamma = 1
#         self.intensity = density * extent_x * extent_y * extent_z
#         self.mass = m_e

#         # Transverse
#         self.x *= extent_x
#         self.xp *= 0
#         self.y *= extent_y
#         self.yp *= 0
#         self.z *= extent_z
#         self.dp *= 0

#         # Initial distribution
#         self.x0 = self.x.copy()
#         self.xp0 = self.xp.copy()
#         self.y0 = self.y.copy()
#         self.yp0 = self.yp.copy()
#         self.z0 = self.z.copy()
#         self.dp0 = self.dp.copy()

#         return self

#     @classmethod
#     def as_ghost(cls, n_macroparticles):

#         self = cls()

#         self._create_uniform(n_macroparticles)

#         return self

#     def _create_empty(self, n_macroparticles):

#         self.x = np.zeros(n_macroparticles)
#         self.xp = np.zeros(n_macroparticles)
#         self.y = np.zeros(n_macroparticles)
#         self.yp = np.zeros(n_macroparticles)
#         self.z = np.zeros(n_macroparticles)
#         self.dp = np.zeros(n_macroparticles)

#     def _create_gauss(self, n_macroparticles):

#         self.x = np.random.randn(n_macroparticles)
#         self.xp = np.random.randn(n_macroparticles)
#         self.y = np.random.randn(n_macroparticles)
#         self.yp = np.random.randn(n_macroparticles)
#         self.z = np.random.randn(n_macroparticles)
#         self.dp = np.random.randn(n_macroparticles)

#     def _create_uniform(self, n_macroparticles):

#         self.x = 2 * np.random.rand(n_macroparticles) - 1
#         self.xp = 2 * np.random.rand(n_macroparticles) - 1
#         self.y = 2 * np.random.rand(n_macroparticles) - 1
#         self.yp = 2 * np.random.rand(n_macroparticles) - 1
#         self.z = 2 * np.random.rand(n_macroparticles) - 1
#         self.dp = 2 * np.random.rand(n_macroparticles) - 1

#     # def _set_beam_geometry(self, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, beta_z, sigma_z=None, epsn_z=None,
#     #                        distribution='gauss'):

#     #     # Transverse
#     #     if distribution == 'gauss':
#     #         sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (bunch.gamma * bunch.beta))
#     #         sigma_xp = sigma_x / beta_x
#     #         sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (bunch.gamma * bunch.beta))
#     #         sigma_yp = sigma_y / beta_y

#     #         self.x *= sigma_x
#     #         self.xp *= sigma_xp
#     #         self.y *= sigma_y
#     #         self.yp *= sigma_yp
#     #     else:
#     #         raise(ValueError)

#     #     # Longitudinal
#     #     # Assuming a gaussian-type stationary distribution: beta_z = eta * circumference / (2 * np.pi * Qs)
#     #     if sigma_z and epsn_z:
#     #         sigma_dp = epsn_z / (4 * np.pi * sigma_z) * e / self.p0
#     #         if sigma_z / sigma_dp != beta_z:
#     #             print '*** WARNING: beam mismatched in bucket. Set synchrotron tune to obtain beta_z = ', sigma_z / sigma_dp
#     #     elif not sigma_z and epsn_z:
#     #         sigma_z = np.sqrt(beta_z * epsn_z / (4 * np.pi) * e / self.p0)
#     #         sigma_dp = sigma_dz / beta_z
#     #     else:
#     #         sigma_dp = sigma_dz / beta_z

#     #     self.dz *= sigma_dz
#     #     self.dp *= sigma_dp

#     @property
#     def n_macroparticles(self):

#         return len(self.x)

#     @property
#     def beta(self):

#         return np.sqrt(1 - 1 / self.gamma ** 2)

#     @property
#     def p0(self):

#         return self.mass * self.gamma * self.beta * c

#     def reinit(self):

#         np.copyto(self.x, self.x0)
#         np.copyto(self.xp, self.xp0)
#         np.copyto(self.y, self.y0)
#         np.copyto(self.yp, self.yp0)
#         np.copyto(self.z, self.z0)
#         np.copyto(self.dp, self.dp0)

#     # #~ @profile
#     def sort_particles(self):
#         # update the number of lost particles
#         self.n_macroparticles_lost = (self.n_macroparticles - np.count_nonzero(self.id))

#         # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
#         if self.n_macroparticles_lost:
#             z_argsorted = np.lexsort((self.z, -np.sign(self.id))) # place lost particles at the end of the array
#         else:
#             z_argsorted = np.argsort(self.z)

#         self.x = self.x.take(z_argsorted)
#         self.xp = self.xp.take(z_argsorted)
#         self.y = self.y.take(z_argsorted)
#         self.yp = self.yp.take(z_argsorted)
#         self.z = self.z.take(z_argsorted)
#         self.dp = self.dp.take(z_argsorted)
#         self.id = self.id.take(z_argsorted)
