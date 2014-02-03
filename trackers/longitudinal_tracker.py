'''
@class Cavity
@author Kevin Li
@date 03.02.2014
@brief Class for creation and management of the synchrotron transport matrices
@copyright CERN
'''


import numpy as np


from configuration import *


class RFCavity(object):
    '''
    classdocs
    '''

    def __init__(self, length, gamma_transition, frequency, voltage, phi)):
        '''
        Constructor
        '''
        self.i_turn = 0
        self.time = 0
        self.length = length
        self. gamma_transition = gamma_transition
        self.frequency = frequency
        self.voltage = voltage
        self.phi = phi

    def match(bunch):
    
         pass

    def track(bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        eta = 1 / (bunch.gamma * bunch.gamma) - 1 / (gamma_transition * gamma_transition)

        tsgn = (eta > 0) - (eta < 0)

        coefficient_omega = 2 * np.pi * frequency / (bunch.beta * c)
        coefficient_voltage = tsgn * e * voltage / (p0 * bunch.beta * c)

        # Length L drift
        bunch.dz += - eta * length * bunch.dp
        # Full turn kick
        bunch.dp += c_voltage * np.sin(c_omega * bunch.dz + phi)

    
class CSCavity(object):
    '''
    classdocs
    '''

    def __init__(self, length, gamma_transition, Qs):
        self.circumference = circumference
        self.gamma_transition = transition
        self.Qs = Qs

    def track(bunch):
            double p0 = bunch.mass * bunch.gamma * bunch.beta * c;
    double eta = 1 / (bunch.gamma * bunch.gamma) - 1 / (gamma_tr * gamma_tr);

    int tsgn = (eta > 0) - (eta < 0);

    double omega_0 = 2 * M_PI * bunch.beta * c / C;
    double omega_s = Qs * omega_0;

    double dQs = 2 * M_PI * Qs;
    double cosdQs = cos(dQs);
    double sindQs = sin(dQs);

    for (size_t i=0; i<bunch.get_nparticles(); i++)
    {
        double dz0 = bunch.dz[i];
        double dp0 = bunch.dp[i];

        bunch.dz[i] = dz0 * cosdQs - eta * c / omega_s * dp0 * sindQs;
        bunch.dp[i] = dp0 * cosdQs + omega_s / eta / c * dz0 * sindQs;
    }
