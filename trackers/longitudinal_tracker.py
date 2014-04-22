from __future__ import division
'''
@class Cavity
@author Kevin Li
@date 03.02.2014
@brief Class for creation and management of the synchrotron transport matrices
@copyright CERN
'''


import numpy as np
import sys


from beams.distributions import stationary_exponential
from scipy.integrate import quad, dblquad
from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e
from libintegr import symple

sin = np.sin
cos = np.cos



class LongitudinalTracker(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def hamiltonian():
        pass

    @abstractmethod
    def separatrix():
        pass

    @abstractmethod
    def isin_separatrix():
        pass

class RFCavity(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, gamma_transition, 
                        frequency, voltage, phi_s, integrator=symple.Euler_Cromer):
        '''
        Constructor
        '''
        self.integrator = integrator

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.h = frequency
        self.voltage = voltage
        self.phi_s = phi_s

    def eta(self, bunch):
        return self.gamma_transition**-2 - bunch.gamma**-2

    def Qs(self, bunch):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           - e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))
        '''
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(bunch)) * self.h / (2 * np.pi * bunch.p0 * bunch.beta * c))

        return Qs
    
    def hamiltonian(self, dz, dp, bunch):

        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)

        phi = self.h / R * dz + self.phi_s

        H = -0.5 * eta * bunch.beta * c * dp ** 2 \
           + e * self.voltage / (bunch.p0 * 2 * np.pi * self.h) \
           * (cos(phi) - cos(self.phi_s) + (phi - self.phi_s) * sin(self.phi_s))

        return H

    def separatrix(self, dz, bunch):
        '''
        Separatrix defined by

        .. math::
        p(dz): (H(dz, dp) == H(zmax, 0))
        '''

        R = self.circumference / (2 * np.pi) 
        eta = self.eta(bunch)
        Qs = self.Qs(bunch)

        phi = self.h / R * dz + self.phi_s 
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        p_sq =  cf1 * (1 + cos(phi) + (phi - np.pi) * sin(self.phi_s))

        return np.sqrt(p_sq)

    def isin_separatrix(self, dz, dp, bunch):

        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) #np.sqrt(e * self.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        zmax = np.pi * R / self.h
        pmax = cf1 * (-1 - cos(phi) + (np.pi - phi) * sin(self.phi_s))

        isin = np.abs(dz) < zmax and dp ** 2 < np.abs(pmax)

        return isin

    # @profile
    def track(self, bunch):

        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
         
        cf1 = self.h / R
        cf2 = np.sign(eta) * e * self.voltage / (bunch.p0 * bunch.beta * c)

        def drift(dp): return -eta * self.length * dp           # Hamiltonian derived by dp
        def kick(dz): return -cf2 * sin(cf1 * dz + self.phi_s)  # Hamiltonian derived by dz

        # vectorised:
        bunch.dz, bunch.dp = self.integrator(
                        bunch.dz, bunch.dp, self.length, drift, kick)

        bunch.update_slices()


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
        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2

        omega_0 = 2 * np.pi * bunch.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = cos(dQs)
        sindQs = sin(dQs)

        dz0 = bunch.dz
        dp0 = bunch.dp
    
        bunch.dz = dz0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        bunch.dp = dp0 * cosdQs + omega_s / eta / c * dz0 * sindQs
        
        bunch.update_slices()
