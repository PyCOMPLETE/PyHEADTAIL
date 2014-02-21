'''
@class Cavity
@author Kevin Li
@date 03.02.2014
@brief Class for creation and management of the synchrotron transport matrices
@copyright CERN
'''


import numpy as np


from abc import ABCMeta, abstractmethod 
from configuration import *


class LongitudinalTracker(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def dpmax():

        return None

    @abstractmethod
    def dzmax():

        return None

    @abstractmethod
    def hamiltonian():

        return None

    @abstractmethod
    def separatrix():

        return None


def match_simple(bunch, cavity):

    n_particles = len(bunch.dz)
    for i in xrange(n_particles):
        if not cavity.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
            while not cavity.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
                bunch.dz[i] = np.random.randn()
                bunch.dp[i] = np.random.randn()

def match_full(bunch, cavity):
    pass

def synchrotron_tune(bunch, cavity):
    '''
    Synchrotron tune derived from the linearized Hamiltonian

    .. math::
    H = -1 / 2 * eta * beta * c * delta ** 2
    - e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))
    '''
    p0 = bunch.mass * bunch.gamma * bunch.beta * c

    R = self.circumference / (2 * np.pi)
    omega_0 = bunch.beta * c / R
    omega_rf = 2 * np.pi * self.frequency
    h = omega_rf / omega_0

    eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2
    phi = R / h * dz + self.phi_s
    phi_s = self.phi_s
    V = self.voltage

    Qs = e * V * eta * h / (2 * pi * bunch.beta * c * p0)

    return Qs

def bunchlength(H,  sigma_dz):

    print 'Iterative evaluation of bunch length...'

    counter = 0
    eps = 1

    zmax = pi * R / h
    Hmax = H(zmax, 0)

    # Initial values
    z0 = sigma_dz
    p0 = z0 * Qs / eta / R            #Matching condition
    H0 = eta * beta * c * p0 ** 2

    z1 = z0
    while abs(eps)>1e-6:
        # Separatrix
        dplim = lambda dz: sqrt(2) * Qs / (eta * h) * sqrt(cos(h / R * dz) - cos(h / R * zmax))
        # Stationary distribution
        psi = lambda dz, dp: exp(H(dz, dp) / H0) - exp(Hmax / H0)

        N = dblquad(lambda dp, dz: psi(dz, dp), -zmax, zmax, lambda dz: -dplim(dz), lambda dz: dplim(dz))
        I = dblquad(lambda dp, dz: dz ** 2 * psi(dz, dp), -zmax, zmax, lambda dz: -dplim(dz), lambda dz: dplim(dz))

        # Second moment
        z2 = sqrt(I[0] / N[0])
        eps = z2-z0

        print z1, z2, eps
        z1 -= eps

        p0 = z1 * Qs / eta / R
        H0 = eta * beta * c * p0 ** 2

        counter += 1
        if counter > 10000:
            print '\n*** WARNING: too many interation steps! Target bunchlength seems to exceed bucket. Aborting...'
            exit(-1)

    return z1


class RFCavity(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, gamma_transition, frequency, voltage, phi_s):
        '''
        Constructor
        '''
        self.i_turn = 0
        self.time = 0
        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.frequency = frequency
        self.voltage = voltage
        self.phi_s = phi_s

    def hamiltonian(dz, dp, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c

        R = self.circumference / (2 * np.pi)
        omega_0 = bunch.beta * c / R
        omega_rf = 2 * np.pi * self.frequency
        h = omega_rf / omega_0

        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2
        phi = R / h * dz + self.phi_s
        phi_s = self.phi_s
        V = self.voltage

        H = -1 / 2 * eta * bunch.beta * c * dp ** 2
          - e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))

        return H

    # def separatrix(dp, bunch):
    #     '''
    #     Separatriox defined by

    #     .. math::
    #     p(dz): (H(dz, dp) == H(zmax, 0))
    #     '''
    #     p0 = bunch.mass * bunch.gamma * bunch.beta * c
    
    #     R = self.circumference / (2 * np.pi)
    #     omega_0 = bunch.beta * c / R
    #     omega_rf = 2 * np.pi * self.frequency
    #     h = omega_rf / omega_0

    #     eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2
    #     phi = R / h * dz + self.phi_s
    #     phi_s = self.phi_s
    #     V = self.voltage

    #     Qs = np.sqrt(e * V * eta * h / (2 * np.pi * p0 * self.beta * c))
    #     cf1 = 2 * Qs ** 2 / (eta * h) ** 2

    #     p =  cf1 * (-1 - np.cos(phi) + (np.pi - phi) * np.sin(phi_s))
    #     p = np.sqrt(p)

    #     return p

    def isin_separatrix(dz, dp, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
    
        R = self.circumference / (2 * np.pi)
        omega_0 = bunch.beta * c / R
        omega_rf = 2 * np.pi * self.frequency
        h = omega_rf / omega_0

        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2
        phi = R / h * dz + self.phi_s
        phi_s = self.phi_s
        V = self.voltage

        Qs = np.sqrt(e * V * eta * h / (2 * np.pi * p0 * self.beta * c))
        cf1 = 2 * Qs ** 2 / (eta * h) ** 2

        zmax = np.pi * R / h
        pmax = cf1 * (-1 - np.cos(phi) + (np.pi - phi) * np.sin(phi_s))

        isin = np.abs(dz) <= cavity.zmax() or dp ** 2 <= pmax

        return isin

    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2

        cf1 = 2 * np.pi * self.frequency / (bunch.beta * c)
        cf2 = np.sign(eta) * e * self.voltage / (p0 * bunch.beta * c)

        # Length L drift
        bunch.dz += - eta * self.length * bunch.dp
        # Full turn kick
        bunch.dp += cf2 * np.sin(cf1 * bunch.dz + self.phi)


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
        cosdQs = np.cos(dQs)
        sindQs = np.sin(dQs)

        dz0 = bunch.dz
        dp0 = bunch.dp
    
        bunch.dz = dz0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        bunch.dp = dp0 * cosdQs + omega_s / eta / c * dz0 * sindQs
