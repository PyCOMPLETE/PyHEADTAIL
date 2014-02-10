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

    def __init__(self, length, gamma_transition, frequency, voltage, phi):
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

    def match(self, bunch):
    
         pass

    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        eta = 1 / (bunch.gamma * bunch.gamma) - 1 / (self.gamma_transition * self.gamma_transition)

        tsgn = (eta > 0) - (eta < 0)

        c_omega = 2 * np.pi * self.frequency / (bunch.beta * c)
        c_voltage = tsgn * e * self.voltage / (p0 * bunch.beta * c)

        # Length L drift
        bunch.dz += - eta * self.length * bunch.dp
        # Full turn kick
        bunch.dp += c_voltage * np.sin(c_omega * bunch.dz + self.phi)

    
class CSCavity(object):
    '''
    classdocs
    '''

    def __init__(self, circumference, gamma_transition, Qs):

        self.circumference = circumference
        self.gamma_transition = gamma_transition
        self.Qs = Qs

    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        eta = 1 / (bunch.gamma * bunch.gamma) - 1 / (self.gamma_transition * self.gamma_transition)

        tsgn = (eta > 0) - (eta < 0)

        omega_0 = 2 * np.pi * bunch.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = np.cos(dQs)
        sindQs = np.sin(dQs)

        dz0 = bunch.dz
        dp0 = bunch.dp
    
        bunch.dz = dz0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        bunch.dp = dp0 * cosdQs + omega_s / eta / c * dz0 * sindQs
