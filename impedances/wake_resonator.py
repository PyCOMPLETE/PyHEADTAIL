from __future__ import division
'''
@class Wakefields
@author Kevin Li
@date March 2013
@brief Class for creation and management of wakefields from impedance sources
@copyright CERN
'''


import numpy as np


from configuration import *
import pylab as plt


class WakeResonator(object):
    '''
    classdocs
    '''

    def __init__(self, R_shunt, frequency, Q, plane='x'):
        '''
        Constructor
        '''
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q
        self.plane = plane

        assert(plane in ('x', 'y', 'z'))
    
        # Taken from Alex Chao's resonator model (2.82)
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(omega ** 2 - alpha ** 2)

    def wake_transverse(self, z):

        # Taken from Alex Chao's resonator model (2.82)
        wake = omega ** 2 / (Q * omegabar) * R_shunt * np.exp(alpha * z / c) * np.sin(omegabar * z / c)

        return wake

    def wake_longitudinal(self, z):

        # Taken from Alex Chao's resonator model (2.82)
        wake = 2 * alpha * R_shunt * np.exp(alpha * z / c) * (np.cos(omegabar * z / c) + alpha / omegabar * sin(omegabar * z / c))

        return wake

    def wake_resistive(self, z):
        
        # Taken from Alex Chao's resisitve wall (2.53)
        sigma = 5.4e17 # copper conductivity in CGS [1/s]
        piper = 2e-3
        length = 6911

        wake = -1 / (2 * np.pi * eps0) / (np.pi * piper ** 3) * np.sqrt(c / sigma) * 1 / np.sqrt(-z) * length

        return wake

    def convolve(self, bunch):

        n_particles = len(bunch.x)
        q = bunch.intensity / n_particles
        cf1 = bunch.charge ** 2 / (bunch.mass * bunch.gamma * c ** 2)

        n_slices = len(bunch.slices.mean_x) - 3
        if not hasattr(self, 'kick'):
            self.kick = np.zeros(n_slices)
        else:
            print self.kick
            self.kick[:] = 0

        # Target
        for i in xrange(n_slices, 0, -1):
            # Beam loading of self-slice
            if self.plane == 'z':
                ni = bunch.slices.charge[i]
                self.kick[i] = cf1 * q * np_i * alpha * R_shunt
            # Sources
            for j in xrange(n_slices, i, -1):
                nj = bunch.slices.charge[j]

                print n_slices
                plt.hist(bunch.dz, n_slices+2)
                plt.stem(bunch.slices.dz_bins, bunch.slices.charge, linefmt='g', markerfmt='go')
                [plt.axvline(i, c='orange') for i in bunch.slices.dz_bins]
                # [plt.axvline(i, c='purple') for i in bunch.slices.dz_centers]
                plt.show()

                zj = 1 / 2. * (bunch.zbins[i] - bunch.zbins[j] + bunch.zbins[i + 1] - bunch.zbins[j + 1])

                if self.plane == 'x':
                    self.kick[i] += cf1 * q * nj * bunch.slices.mean_x[j] * wake_transverse(zj)
                elif self.plane == 'y':
                    self.kick[i] += cf1 * q * nj * bunch.slices.mean_y[j] * wake_transverse(zj)
                elif self.plane == 'z':
                    self.kick[i] += cf1 * q * nj * wake_longitudinal(zj);

                # if (j == n_slices + 1):
                #     bunch.mean_kx[i] = wake_resonator_r(zj);
                #     bunch.mean_ky[i] = kick_y;
                #     bunch.mean_kz[i] = wake_resonator_z(zj);
                    # out << std::scientific
                    #     << j << '\t' << kick_x << '\t'
                    #     << wake_resonator_r(zj) << '\t' << kick_z << std::endl;

    def apply_kick(self, bunch):

        n_slices = len(bunch,slices.mean_x - 3)
        for i in xrange(n_slices, 0, -1):
            index = bunch.slices.index(i)
            if plane == 'x':
                bunch.xp[index] += self.kick[i]
            elif plane == 'y':
                bunch.yp[index] += self.kick[i]
            elif plane == 'z':
                bunch.dp[index] += self.kick[i]

    def track(self, bunch):

        self.convolve(bunch)
        self.apply_kick(bunch)
