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


sin = np.sin
cos = np.cos


class LongitudinalTracker(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def hamiltonian(): pass

    @abstractmethod
    def separatrix(): pass

    @abstractmethod
    def is_in_separatrix(): pass


class RFCavity(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, momentum_compaction, frequency, voltage, phi_s, integrator='rk4'):
        '''
        Constructor
        '''
        self.integrator = integrator

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.alpha = momentum_compaction
        self.h = frequency
        self.voltage = voltage
        self.phi_s = phi_s

        # self._p0 = 0
        # self._R = 0
        # self._omega_0 = 0
        # self._omega_rf = 0
        # self._h = 0

        # self._eta = 0
        # self._Qs = 0

    def eta(self, bunch):

        return self.alpha - 1. / bunch.gamma ** 2

    def Qs(self, bunch):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           + e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))
        '''
        return np.sqrt(e * self.voltage * np.abs(self.eta(bunch)) * self.h
                / (2 * np.pi * bunch.p0 * bunch.beta * c))

    def hamiltonian(self, dz, dp, bunch):

        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)

        phi = self.h / R * dz + self.phi_s

        H = -1 / 2 * eta * bunch.beta * c * dp ** 2 \
           + e * self.voltage / (bunch.p0 * 2 * np.pi * self.h) \
           * (np.cos(phi) - np.cos(self.phi_s) + (phi - self.phi_s) * np.sin(self.phi_s))

        Hsep = e * self.voltage * (-1 - cos(self.phi_s)
             + (np.pi - self.phi_s) * sin(self.phi_s)) / (bunch.p0 * 2 * np.pi * self.h)

        return Hsep - H

    def separatrix(self, dz, bunch):
        '''
        Separatriox defined by

        .. math::
        p(dz): (H(dz, dp) == H(zmax, 0))
        '''
        # p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) # np.sqrt(e * self.voltage * np.abs(eta) * self.h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        p =  cf1 * (1 + np.cos(phi) + (phi - np.pi) * np.sin(self.phi_s))
        p = np.sqrt(p)

        return p

    def is_in_separatrix(self, z, dp, bunch):

        return 2 * np.pi * self.h / self.circumference * np.abs(z) < np.pi and self.hamiltonian(z, dp, bunch) < 0

    #~ @profile
    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)

        cf1 = self.h / R
        cf2 = np.sign(eta) * e * self.voltage / (p0 * bunch.beta * c)

        if self.integrator == 'rk4':
            # Initialize
            dz0 = bunch.z
            dp0 = bunch.dp

            # Integration
            k1 = -eta * self.length * dp0
            kp1 = cf2 * sin(cf1 * dz0 + self.phi_s)
            k2 = -eta * self.length * (dp0 + kp1 / 2)
            kp2 = cf2 * sin(cf1 * (dz0 + k1 / 2) + self.phi_s)
            k3 = -eta * self.length * (dp0 + kp2 / 2)
            kp3 = cf2 * sin(cf1 * (dz0 + k2 / 2) + self.phi_s)
            k4 = -eta * self.length * (dp0 + kp3);
            kp4 = cf2 * sin(cf1 * (dz0 + k3) + self.phi_s)

            # Finalize
            bunch.z = dz0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
            bunch.dp = dp0 + kp1 / 6 + kp2 / 3 + kp3 / 3 + kp4 / 6

        elif self.integrator == 'euler-chromer':
            # Length L drift
            bunch.z += - eta * self.length * bunch.dp
            # Full turn kick
            bunch.dp += cf2 * sin(cf1 * bunch.z + self.phi_s)

        elif self.integrator == 'ruth4':
            alpha = 1 - 2 ** (1 / 3)
            d1 = 1 / (2 * (1 + alpha))
            d2 = alpha / (2 * (1 + alpha))
            d3 = alpha / (2 * (1 + alpha))
            d4 = 1 / (2 * (1 + alpha))
            c1 = 1 / (1 + alpha)
            c2 = (alpha - 1) / (1 + alpha)
            c3 = 1 / (1 + alpha)
            # c4 = 0;

            # Initialize
            dz0 = bunch.z
            dp0 = bunch.dp

            dz1 = dz0
            dp1 = dp0
            # Drift
            dz1 += d1 * -eta * self.length * dp1
            # Kick
            dp1 += c1 * cf2 * sin(cf1 * dz1 + self.phi_s)

            dz2 = dz1
            dp2 = dp1
            # Drift
            dz2 += d2 * -eta * self.length * dp2
            # Kick
            dp2 += c2 * cf2 * sin(cf1 * dz2 + self.phi_s)

            dz3 = dz2
            dp3 = dp2
            # Drift
            dz3 += d3 * -eta * self.length * dp3
            # Kick
            dp3 += c3 * cf2 * sin(cf1 * dz3 + self.phi_s)

            dz4 = dz3
            dp4 = dp3
            # Drift
            dz4 += d4 * -eta * self.length * dp4

            # Finalize
            bunch.z = dz4
            bunch.dp = dp4

        # bunch.update_slices()


class LongitudinalMap(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, gamma_transition, harmonics, voltages, phi_s, integrator='rk4'):
        '''
        Constructor
        '''
        self.integrator = integrator

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.h = harmonics
        self.voltages = voltages
        self.phi_s = phi_s

    def eta(self, bunch):

        eta = 1. / self.gamma_transition ** 2 - 1. / bunch.gamma ** 2

        return eta

    def Qs(self, bunch):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           - e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))
        '''
        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        Qs = np.sqrt(e * np.abs(self.eta(bunch)) * np.sum(self.V * self.h) / (2 * np.pi * p0 * bunch.beta * c))

        return Qs

    def hamiltonian(self, dz, dp, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)

        phi = self.h / R * dz + self.phi_s

        T = -1 / 2 * eta * bunch.beta * c * dp ** 2
        V = 0
        for i in range(len(self.voltages)):
            V += e * self.voltage[i] / (p0 * 2 * np.pi * self.h[i]) \
           * (np.cos(phi[i]) - np.cos(self.phi_s[i]) + (phi[i] - self.phi_s[i]) * np.sin(self.phi_s[i]))

        H = T + V

        return H

    def separatrix(self, dz, bunch):
        '''
        Separatriox defined by

        .. math::
        p(dz): (H(dz, dp) == H(zmax, 0))
        '''
        # p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) # np.sqrt(e * self.voltage * np.abs(eta) * self.h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        p =  cf1 * (1 + np.cos(phi) + (phi - np.pi) * np.sin(self.phi_s))
        p = np.sqrt(p)

        return p

    def isin_separatrix(self, dz, dp, bunch):

        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) #np.sqrt(e * self.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        zmax = np.pi * R / self.h
        pmax = cf1 * (-1 - np.cos(phi) + (np.pi - phi) * np.sin(self.phi_s))

        isin = np.abs(dz) < zmax and dp ** 2 < np.abs(pmax)

        return isin

    #~ @profile
    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)

        cf1 = self.h / R
        cf2 = np.sign(eta) * e * self.voltage / (p0 * bunch.beta * c)

        if self.integrator == 'rk4':
            # Initialize
            dz0 = bunch.z
            dp0 = bunch.dp

            # Integration
            k1 = -eta * self.length * dp0
            kp1 = cf2 * sin(cf1 * dz0 + self.phi_s)
            k2 = -eta * self.length * (dp0 + kp1 / 2)
            kp2 = cf2 * sin(cf1 * (dz0 + k1 / 2) + self.phi_s)
            k3 = -eta * self.length * (dp0 + kp2 / 2)
            kp3 = cf2 * sin(cf1 * (dz0 + k2 / 2) + self.phi_s)
            k4 = -eta * self.length * (dp0 + kp3);
            kp4 = cf2 * sin(cf1 * (dz0 + k3) + self.phi_s)

            # Finalize
            bunch.z = dz0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
            bunch.dp = dp0 + kp1 / 6 + kp2 / 3 + kp3 / 3 + kp4 / 6

        elif self.integrator == 'euler-chromer':
            # Length L drift
            bunch.z += - eta * self.length * bunch.dp
            # Full turn kick
            bunch.dp += cf2 * sin(cf1 * bunch.z + self.phi_s)

        elif self.integrator == 'ruth4':
            alpha = 1 - 2 ** (1 / 3)
            d1 = 1 / (2 * (1 + alpha))
            d2 = alpha / (2 * (1 + alpha))
            d3 = alpha / (2 * (1 + alpha))
            d4 = 1 / (2 * (1 + alpha))
            c1 = 1 / (1 + alpha)
            c2 = (alpha - 1) / (1 + alpha)
            c3 = 1 / (1 + alpha)
            # c4 = 0;

            # Initialize
            dz0 = bunch.z
            dp0 = bunch.dp

            dz1 = dz0
            dp1 = dp0
            # Drift
            dz1 += d1 * -eta * self.length * dp1
            # Kick
            dp1 += c1 * cf2 * sin(cf1 * dz1 + self.phi_s)

            dz2 = dz1
            dp2 = dp1
            # Drift
            dz2 += d2 * -eta * self.length * dp2
            # Kick
            dp2 += c2 * cf2 * sin(cf1 * dz2 + self.phi_s)

            dz3 = dz2
            dp3 = dp2
            # Drift
            dz3 += d3 * -eta * self.length * dp3
            # Kick
            dp3 += c3 * cf2 * sin(cf1 * dz3 + self.phi_s)

            dz4 = dz3
            dp4 = dp3
            # Drift
            dz4 += d4 * -eta * self.length * dp4

            # Finalize
            bunch.z = dz4
            bunch.dp = dp4


class LinearMap(object):
    '''
    classdocs
    '''

    def __init__(self, circumference, momentum_compaction, Qs):

        self.circumference = circumference
        self.alpha = momentum_compaction
        self.Qs = Qs

    def track(self, bunch):

        eta = self.alpha - 1 / bunch.gamma ** 2

        omega_0 = 2 * np.pi * bunch.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = cos(dQs)
        sindQs = sin(dQs)

        z0 = bunch.z
        dp0 = bunch.dp

        bunch.z = z0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        bunch.dp = dp0 * cosdQs + omega_s / eta / c * z0 * sindQs
