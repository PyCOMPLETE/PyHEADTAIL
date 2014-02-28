from __future__ import division
'''
@class Wakefields
@author Kevin Li
@date March 2013
@brief Class for creation and management of wakefields from impedance sources
@copyright CERN
'''


import numpy as np


class WakeResonator(object):
    '''
    classdocs
    '''

    def __init__(self, R_shunt, frequency, Q):
        '''
        Constructor
        '''
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q
    
        # Taken from Alex Chao's resonator model (2.82)
        omega = 2 * pi * frequency
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

    def track(self, bunch):

        n_slices = len(bunch.slices.mean_x) - 3
        if not self.kick:
            self.kick = np.zeros(n_slices)
        else:
            self.kick[:] = 0
            
        n_particles = len(bunch.x)
        q = bunch.intensity / n_particles
        cf1 = bunch.charge ** 2 / (bunch.mass * bunch.gamma * c ** 2)

        # Initialization
        kick_x = 0;
        kick_y = 0;
        kick_z = cf1 * q * np_i * alpha * R_shunt # Beam loading of self-slice

        bunch.get_slice(i, lambda_i, index);
        int np_i = index.size();

        # TODO: write as source and target...
        for i in xrange(n_slices + 1, 0, -1):
            ni = bunch.slices.charge[i]
            for j in xrange(n_slices + 1, i, -1):
                nj = bunch.slices.charge[j]
                index = bunch.slices.indices[i]

                double zj = 1 / 2. * (bunch.slice_dz[j] - bunch.slice_dz[i]
                                    + bunch.slice_dz[j + 1] - bunch.slice_dz[i + 1]);

                # double zj = 1 / 2. * (bunch.zbins[i] - bunch.zbins[j]
                                    # + bunch.zbins[i + 1] - bunch.zbins[j + 1]);
                if self.plane == 'x':
                    kick += cf1 * q * nj * bunch.slices.mean_x[j] * wake_transverse(zj)
                elif self.plane == 'y':
                    kick += cf1 * q * nj * bunch.slices.mean_y[j] * wake_transverse(zj)
                elif self.plane == 'z':
                    kick += cf1 * q * nj * wake_longitudinal(zj);

                if (j == n_slices + 1):
                    bunch.mean_kx[i] = wake_resonator_r(zj);
                    bunch.mean_ky[i] = kick_y;
                    bunch.mean_kz[i] = wake_resonator_z(zj);
                    # out << std::scientific
                    #     << j << '\t' << kick_x << '\t'
                    #     << wake_resonator_r(zj) << '\t' << kick_z << std::endl;
            
        # Apply kicks
        for (int j=0; j<np_i; j++)
        {
            bunch.xp[index[j]] += kick_x;
            bunch.yp[index[j]] += kick_y;
            bunch.dp[index[j]] += kick_z;
        }
