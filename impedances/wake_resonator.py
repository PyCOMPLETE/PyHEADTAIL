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
        
        double lambda_i;
        std::vector<int> index;

        double qp = bunch.intensity / bunch.get_nparticles();
        double scale = -bunch.charge * bunch.charge / (bunch.mass * bunch.gamma * c * c);

        for (size_t i=1; i<bunch.get_nslices() + 1; i++)
        # for (size_t i=bunch.get_nslices(); i>0; i--)
        {
            bunch.get_slice(i, lambda_i, index);
            int np_i = index.size();

            # Initialization
            kick_x = 0;
            kick_y = 0;
            kick_z = scale * qp * np_i * alpha_z * Rs_z; // Beam loading of self-slice

            # Interact with all sources
            for (size_t j=1; j<i; j++)
            # for (size_t j=bunch.get_nslices(); j>i; j--)
            {
                int np_j = bunch.slice_index[j].size();
                double zj = 1 / 2. * (bunch.slice_dz[j] - bunch.slice_dz[i]
                                    + bunch.slice_dz[j + 1] - bunch.slice_dz[i + 1]);

                # double zj = 1 / 2. * (bunch.zbins[i] - bunch.zbins[j]
                                    # + bunch.zbins[i + 1] - bunch.zbins[j + 1]);

                kick_x += scale * qp * np_j * bunch.mean_x[j] * wake_resonator_r(zj);
                kick_y += scale * qp * np_j * bunch.mean_y[j] * wake_resonator_r(zj);
                kick_z += scale * qp * np_j * wake_resonator_z(zj);

                # if (j==bunch.get_nslices())
                if (j==1)
                {
                    bunch.mean_kx[i] = wake_resonator_r(zj);
                    bunch.mean_ky[i] = kick_y;
                    bunch.mean_kz[i] = wake_resonator_z(zj);
                    out << std::scientific
                        << j << '\t' << kick_x << '\t'
                        << wake_resonator_r(zj) << '\t' << kick_z << std::endl;
                }
            }

            # Apply kicks
            for (int j=0; j<np_i; j++)
            {

                bunch.xp[index[j]] += kick_x;
                bunch.yp[index[j]] += kick_y;
                bunch.dp[index[j]] += kick_z;
            }
        }
