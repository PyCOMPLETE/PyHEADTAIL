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
from functools import partial


from beams.distributions import stationary_exponential
from scipy.integrate import quad, dblquad
from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e
from libintegrators import symple

sin = np.sin
cos = np.cos



class LongitudinalTracker(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def hamiltonian(self, dz, dp, bunch):
        pass

    @abstractmethod
    def separatrix(self, dz, bunch):
        pass

    @abstractmethod
    def isin_separatrix(self, dz, dp, bunch):
        pass

    @abstractmethod
    def track(self, bunch):
        pass

class RFCavity(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, gamma_transition, 
                        harmonic, voltage, phi_s, integrator=symple.Euler_Cromer):
        '''
        Constructor
        '''
        self.integrator = integrator

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.harmonic = harmonic
        self.voltage = voltage
        self.phi_s = phi_s

    def eta(self, bunch):
        return self.gamma_transition**-2 - bunch.gamma**-2

    def Qs(self, bunch):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           + e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))
        '''
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(bunch)) * self.harmonic \
                / (2 * np.pi * bunch.p0 * bunch.beta * c))

        return Qs

    def potential(self, dz, bunch):
        """the potential part V(dz) of the cavity's separable Hamiltonian"""

        R = self.circumference / (2 * np.pi)
        phi = self.harmonic / R * dz + self.phi_s

        return e * self.voltage / (bunch.p0 * 2 * np.pi * self.harmonic) \
           * (cos(phi) - cos(self.phi_s) + (phi - self.phi_s) * sin(self.phi_s))

    def hamiltonian(self, dz, dp, bunch):
        """the full separable Hamiltonian of the cavity"""

        kinetic = -0.5 * self.eta(bunch) * bunch.beta * c * dp ** 2

        return kinetic + self.potential(dz, bunch)

    def separatrix(self, dz, bunch):
        '''
        returns the separatrix momentum depending on dz.
        Separatrix defined by

        .. math::
        p(dz): (H(dz, dp) == H(zmax, 0))
        '''

        R = self.circumference / (2 * np.pi) 
        eta = self.eta(bunch)
        Qs = self.Qs(bunch)

        phi = self.harmonic / R * dz + self.phi_s 
        cf1 = 2 * Qs ** 2 / (eta * self.harmonic) ** 2

        return np.sqrt( cf1 * (1 + cos(phi) + (phi - np.pi) * sin(self.phi_s)) )

    def isin_separatrix(self, dz, dp, bunch):

        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) #np.sqrt(e * self.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.harmonic / R * dz + self.phi_s
        cf1 = 2 * Qs ** 2 / (eta * self.harmonic) ** 2

        zmax = np.pi * R / self.harmonic
        psqmax = cf1 * (-1 - cos(phi) + (np.pi - phi) * sin(self.phi_s))

        isin = np.abs(dz) < zmax and dp ** 2 < np.abs(psqmax)

        return isin

    # @profile
    def track(self, bunch):

        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
         
        cf1 = self.harmonic / R
        cf2 = np.sign(eta) * e * self.voltage / (bunch.p0 * bunch.beta * c)

        def drift(dp): return -eta * self.length * dp           # Hamiltonian derived by dp
        def kick(dz): return -cf2 * sin(cf1 * dz + self.phi_s)  # Hamiltonian derived by dz

        bunch.dz, bunch.dp = self.integrator(bunch.dz, bunch.dp, 1, drift, kick)

class RFCavityArray(LongitudinalTracker):
    """
        provides an array of RFCavities which is able to accelerate.
        The signature is the same as for RFCavity except that frequencies, voltages
        and phi_s are passed as lists with as many entries as there are RFCavities.
        The length of each RFCavity will be 0 except for the last one
        covering the whole length.
        The acceleration method shall only be applied once per turn!
        For one RFCavityArray per ring layout (with all RFCavities at the 
        same longitudinal position) the longitudinal separatrix function 
        is exact and makes a valid local statement about stability!
    """
    def __init__(self, circumference, length, gamma_transition, 
                harmonic_list, voltage_list, phi_s_list, integrator=symple.Euler_Cromer):

        if not len(harmonic_list) == len(voltage_list) == len(phi_s_list):
            print ("Warning: parameter lists for RFCavityArray do not have the same length!")
        self.cavities = []
        parameters = zip(harmonic_list, voltage_list, phi_s_list)
        # drive-thru from the back
        parameters.reverse()
        l = length
        for harmonic, voltage, phi_s in parameters:
            self.cavities.append( 
                            RFCavity(self, circumference, l, gamma_transition, 
                                                    harmonic, voltage, phi_s, integrator)
                                )
            l = 0
        self.cavities.reverse()

    def track(self, bunch):
        for cavity in self.cavities:
            cavity.track(bunch)

    def hamiltonian(self, dz, dp, bunch):
        def fetch_potential(cavity):
            return cavity.potential(dz, bunch)
        potential_list = map(fetch_potential, self.cavities)
        kinetic = -0.5 * self.eta(bunch) * bunch.beta * c * dp ** 2
        return kinetic + sum(potential_list)

    # def separatrix(self, dz, bunch):
    #     pass

    def isin_separatrix(self, dz, dp, bunch):
        # the proper way to do it:
        #   - define bucket reference interval (as lowest symmetry interval)
        #       by lowest harmonic number of cavities
        #   - search the highest maxima of the same value
        #   - (make sure reference interval of bucket starts at first one)
        #   - if bucket non-accelerated (i.e. right side same value as left side)
        #       -> real bucket is between two consecutive highest maxima
        #   - if bucket accelerated (i.e. right side has offset from left side value)
        #       -> 
        pass

    def accelerate_to(self, bunch, gamma):
        """accelerates the given bunch to the given gamma, i.e.
        - its transverse geometric emittances shrink
        - its gamma becomes the given gamma
        - phi_s of the cavities changes
        for the moment, acceleration only works for Euler-Cromer,
        as the other multi-step integrators have to be adapted to
        change parameters (adapted to respective gammas) during integration!"""

        assert (self.integrator is symple.Euler-Cromer)
        self.phi_s = 
        gamma_old   = bunch.gamma
        beta_old    = bunch.beta
        bunch.gamma = gamma
        bunch.x    *= np.sqrt(gamma_old * beta_old / (bunch.gamma * bunch.beta))
        bunch.xp   *= np.sqrt(gamma_old * beta_old / (bunch.gamma * bunch.beta))
        bunch.y    *= np.sqrt(gamma_old * beta_old / (bunch.gamma * bunch.beta))
        bunch.yp   *= np.sqrt(gamma_old * beta_old / (bunch.gamma * bunch.beta))


class CSCavity(object):
    '''
        Courant-Snyder transportation.
    '''

    def __init__(self, circumference, gamma_transition, Qs):

        self.circumference = circumference
        self.gamma_transition = gamma_transition
        self.Qs = Qs

    def track(self, bunch):

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
