from __future__ import division
'''
@class Cavity
@author Kevin Li
@date 03.02.2014
@brief Class for creation and management of the synchrotron transport matrices
@copyright CERN
'''


import numpy as np


from beams.distributions import stationary_exponential
from scipy.integrate import quad, dblquad
from abc import ABCMeta, abstractmethod 
from configuration import *
import pylab as plt


class LongitudinalTracker(object):

    __metaclass__ = ABCMeta

    # @abstractmethod
    # def dpmax():

    #     return None

    # @abstractmethod
    # def dzmax():

    #     return None

    @abstractmethod
    def hamiltonian():

        return None

    # @abstractmethod
    # def separatrix():

    #     return None


# @profile
def match_simple(bunch, cavity):

    p0 = bunch.mass * bunch.gamma * bunch.beta * c
    sigma_dz = np.std(bunch.dz)
    sigma_dp = np.std(bunch.dp)
    epsn_z = 4 * np.pi * sigma_dz * sigma_dp * p0 / e

    n_particles = len(bunch.dz)
    for i in xrange(n_particles):
        if not cavity.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
            while not cavity.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
                bunch.dz[i] = sigma_dz * np.random.randn()
                bunch.dp[i] = sigma_dp * np.random.randn()

def match_full(bunch, cavity):

    p0 = bunch.mass * bunch.gamma * bunch.beta * c

    R = cavity.circumference / (2 * np.pi)
    omega_0 = bunch.beta * c / R
    omega_rf = 2 * np.pi * cavity.frequency
    h = omega_rf / omega_0

    eta = 1 / cavity.gamma_transition ** 2 - 1 / bunch.gamma ** 2
    Qs = np.sqrt(e * cavity.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))

    zmax = np.pi * R / h
    pmax = 2 * Qs / eta / h
    # pmax = cf1 * (-1 - np.cos(phi) + (np.pi - phi) * np.sin(self.phi_s))
    Hmax1 = cavity.hamiltonian(zmax, 0, bunch)
    Hmax2 = cavity.hamiltonian(0, pmax, bunch)
    assert(Hmax1 == Hmax2)
    Hmax = Hmax1
    epsn_z = np.pi / 2 * zmax * pmax * p0 / e
    print '\nStatistical parameters from RF bucket:'
    print zmax, pmax, epsn_z, '\n'

    # Assuming a gaussian-type stationary distribution
    n_particles = len(bunch.x)
    sigma_dz = np.std(bunch.dz)
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * bunch.beta * c * sigma_dp ** 2
    epsn_z = 4 * np.pi * Qs / eta / R * sigma_dz ** 2 * p0 / e
    print '\nStatistical parameters from initialisation:'
    print sigma_dz, sigma_dp, epsn_z, '\n'
    
    print '\nBunchlength:'
    #~#print bunchlength(H1, zmax)[0] * p0 / e
    sigma_dz = bunchlength(bunch, cavity, sigma_dz)
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * bunch.beta * c * sigma_dp ** 2
    #~exit(-1)

    plt.figure(3)
    zl = plt.linspace(-zmax, zmax, 1000) * 1.5
    pl = plt.linspace(-pmax, pmax, 1000) * 1.5

    #~y = exp(H(zl, 0) / H0) - exp(Hmax / H0)
    #~plot(zl, y)
    #~show()
    #~exit(-1)

    zz, pp = plt.meshgrid(zl, pl)
    HH = (np.exp(cavity.hamiltonian(zz, pp, bunch) / H0) - np.exp(Hmax / H0))
    HHmax = np.amax(HH)
    # plt.contourf(zz, pp, HH, 20)
    print HHmax, Hmax, sigma_dz, sigma_dp, H0

    dz = np.zeros(n_particles)
    dp = np.zeros(n_particles)
    for i in xrange(n_particles):
        while True:
            s = (np.random.rand() - 0.5) * 2 * zmax
            #~pmax = sqrt(2) * Qs / eta / h * sqrt(1 + cos(s * h / R))
            #~pmax = sqrt(e * V) / sqrt(c * h * p0 * pi * beta * eta) * sqrt(1 + cos(s * h / R))
            t = (np.random.rand() - 0.5) * 2 * pmax
            u = (np.random.rand()) * HHmax * 1.01
            C = np.exp(cavity.hamiltonian(s, t, bunch) / H0) - np.exp(Hmax / H0)
            if u < C:
                break
        bunch.dz[i] = s
        bunch.dp[i] = t
        
    epsz = np.sqrt(np.mean(dz * dz) * np.mean(dp *dp) - np.mean(dz * dp) * np.mean(dz * dp))
    print '\nStatistical parameters from distribution:'
    print np.std(dz), np.std(dp), 4*np.pi*np.std(dz)*np.std(dp)*p0/e, 4*np.pi*epsz*p0/e

    # plt.scatter(dz, dp, c='r', marker='.')
    # plt.show()
    # exit(-1)

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

def bunchlength(bunch, cavity, sigma_dz):

    print 'Iterative evaluation of bunch length...'

    counter = 0
    eps = 1

    p0 = bunch.mass * bunch.gamma * bunch.beta * c

    R = cavity.circumference / (2 * np.pi)
    omega_0 = bunch.beta * c / R
    omega_rf = 2 * np.pi * cavity.frequency
    h = omega_rf / omega_0

    eta = 1 / cavity.gamma_transition ** 2 - 1 / bunch.gamma ** 2
    Qs = np.sqrt(e * cavity.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))

    zmax = np.pi * R / h
    Hmax = cavity.hamiltonian(zmax, 0, bunch)

    # Initial values
    z0 = sigma_dz
    p0 = z0 * Qs / eta / R            #Matching condition
    H0 = eta * bunch.beta * c * p0 ** 2

    z1 = z0
    while abs(eps)>1e-6:
        # cf1 = 2 * Qs ** 2 / (eta * h) ** 2
        # dplim = lambda dz: np.sqrt(cf1 * (1 + np.cos(h / R * dz) + (h / R * dz - np.pi) * np.sin(cavity.phi_s)))
        # dplim = lambda dz: np.sqrt(2) * Qs / (eta * h) * np.sqrt(np.cos(h / R * dz) - np.cos(h / R * zmax))
        # Stationary distribution
        # psi = lambda dz, dp: np.exp(cavity.hamiltonian(dz, dp, bunch) / H0) - np.exp(Hmax / H0)

        # zs = zmax / 2.

        psi = stationary_exponential(cavity.hamiltonian, Hmax, H0, bunch)
        N = dblquad(lambda dp, dz: psi(dz, dp), -zmax, zmax,
                    lambda dz: -cavity.separatrix(dz, bunch), lambda dz: cavity.separatrix(dz, bunch))
        I = dblquad(lambda dp, dz: dz ** 2 * psi(dz, dp), -zmax, zmax,
                    lambda dz: -cavity.separatrix(dz, bunch), lambda dz: cavity.separatrix(dz, bunch))

        # Second moment
        z2 = np.sqrt(I[0] / N[0])
        eps = z2 - z0

        print z1, z2, eps
        z1 -= eps

        p0 = z1 * Qs / eta / R
        H0 = eta * bunch.beta * c * p0 ** 2

        counter += 1
        if counter > 100:
            print "\n*** WARNING: too many interation steps! There are several possible reasons for that:"
            print "1. Is the Hamiltonian correct?"
            print "2. Is the stationary distribution function convex around zero?"
            print "3. Is the bunch too long to fit into the bucket?"
            print "4. Is this algorithm not qualified?"
            print "Aborting..."
            sys.exit(-1)

    return z1


class RFCavity(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, gamma_transition, frequency, voltage, phi_s, integrator=None):
        '''
        Constructor
        '''
        self.integrator = integrator

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.frequency = frequency
        self.voltage = voltage
        self.phi_s = phi_s

        # self._p0 = 0
        # self._R = 0
        # self._omega_0 = 0
        # self._omega_rf = 0
        # self._h = 0

        # self._eta = 0
        # self._Qs = 0
        # self._cf1 = 0

    def hamiltonian(self, dz, dp, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c

        R = self.circumference / (2 * np.pi)
        omega_0 = bunch.beta * c / R
        omega_rf = 2 * np.pi * self.frequency
        h = omega_rf / omega_0

        phi = h / R * dz + self.phi_s

        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2

        H = -1 / 2 * eta * bunch.beta * c * dp ** 2 \
           + e * self.voltage / (p0 * 2 * np.pi * h) \
           * (np.cos(phi) - np.cos(self.phi_s) + (phi - self.phi_s) * np.sin(self.phi_s))

        return H

    def separatrix(self, dz, bunch):
        '''
        Separatriox defined by

        .. math::
        p(dz): (H(dz, dp) == H(zmax, 0))
        '''
        p0 = bunch.mass * bunch.gamma * bunch.beta * c
    
        R = self.circumference / (2 * np.pi)
        omega_0 = bunch.beta * c / R
        omega_rf = 2 * np.pi * self.frequency
        h = omega_rf / omega_0

        phi = h / R * dz + self.phi_s
            
        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2
        Qs = np.sqrt(e * self.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))
        cf1 = 2 * Qs ** 2 / (eta * h) ** 2

        p =  cf1 * (1 + np.cos(phi) + (phi - np.pi) * np.sin(self.phi_s))
        p = np.sqrt(p)

        return p

    def isin_separatrix(self, dz, dp, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c

        R = self.circumference / (2 * np.pi)
        omega_0 = bunch.beta * c / R
        omega_rf = 2 * np.pi * self.frequency
        h = omega_rf / omega_0

        phi = h / R * dz + self.phi_s

        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2
        Qs = np.sqrt(e * self.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))
        cf1 = 2 * Qs ** 2 / (eta * h) ** 2

        zmax = np.pi * R / h
        pmax = cf1 * (-1 - np.cos(phi) + (np.pi - phi) * np.sin(self.phi_s))

        isin = np.abs(dz) < zmax and dp ** 2 < np.abs(pmax)

        return isin

    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2

        cf1 = 2 * np.pi * self.frequency / (bunch.beta * c)
        cf2 = np.sign(eta) * e * self.voltage / (p0 * bunch.beta * c)

        # Length L drift
        bunch.dz += - eta * self.length * bunch.dp
        # Full turn kick
        bunch.dp += cf2 * np.sin(cf1 * bunch.dz + self.phi_s)


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
