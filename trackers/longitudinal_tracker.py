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

    def __init__(self, circumference, length, gamma_transition, 
                        harmonic, voltage, dphi, integrator=symple.Euler_Cromer):

        self.integrator = integrator

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.harmonic = harmonic
        self.voltage = voltage
        self.dphi = dphi

    def eta(self, bunch):
        return self.gamma_transition**-2 - bunch.gamma**-2

    def Qs(self, bunch):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           + e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(dphi) + (phi - dphi) * np.sin(dphi))
        '''
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(bunch)) * self.harmonic \
                / (2 * np.pi * bunch.p0 * bunch.beta * c))

        return Qs

    def potential(self, dz, bunch):
        """the potential part V(dz) of the cavity's separable Hamiltonian"""

        R = self.circumference / (2 * np.pi)
        phi = self.harmonic / R * dz + self.dphi

        return e * self.voltage / (bunch.p0 * 2 * np.pi * self.harmonic) \
           * (cos(phi) - cos(self.dphi) + (phi - self.dphi) * sin(self.dphi))

    def hamiltonian(self, dz, dp, bunch):
        """the full separable Hamiltonian of the rf cavity"""

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
        Qs = self.Qs(bunch)

        phi = self.harmonic / R * dz + self.dphi 
        cf1 = 2 * Qs ** 2 / (self.eta(bunch) * self.harmonic) ** 2

        return np.sqrt( cf1 * (1 + cos(phi - self.dphi) + (phi - np.pi) * sin(self.dphi)) )

    def isin_separatrix(self, dz, dp, bunch):

        R = self.circumference / (2 * np.pi)
        Qs = self.Qs(bunch) 

        phi = self.harmonic / R * dz + self.dphi
        cf1 = 2 * Qs ** 2 / (self.eta(bunch) * self.harmonic) ** 2

        zmax = np.pi * R / self.harmonic
        psqmax = cf1 * (-1 - cos(phi - self.dphi) + (np.pi - phi) * sin(self.dphi))

        isin = np.abs(dz) < zmax and dp ** 2 < np.abs(psqmax)

        return isin

    def drift(self, bunch, dp):
        """the drift (i.e. Delta z) of the particle's z coordinate is given by
        the (separable) Hamiltonian derived by dp (being (p - p0) / p0).
        It's a function of dp."""
        return -self.eta(bunch) * self.length * dp

    def kick(self, bunch, dz):
        """the kick (i.e. Delta dp) of the particle's dp coordinate is given by
        the (separable) Hamiltonian derived by z, i.e. the negative force.
        (negative because of the symplectic structure of the e.o.m.,
        kills the minus sign of the momentum part in the integrator.)
        It's a function of z.
        It is for a stationary bucket (no net momentum update!)."""
        sgn_eta = np.sign(self.eta(bunch))
        cf1 = 2 * np.pi * self.harmonic / self.circumference
        cf2 = sgn_eta * e * self.voltage / (bunch.p0 * bunch.beta * c)
        force = cf2 * sin(cf1 * dz + self.dphi)
        return -force

    # @profile
    def track(self, bunch):
        """
            It is assumed that this cavity adds fully to the total one-turn kick
            (this is needed for instance in the RFSystems where the cavities
                with zero self.length add to the total kick but have no drift, this way
                we end up with one drift and a sum of kicks (in the same place) 
                by the various cavities while the validity of the separatrix 
                as a local statement of stability is ensured!)
            The self.length of the cavity only tells how much the drift advances.
            (For distributed rf systems around the ring, adaptions have to be made --
            mind that then the separatrix loses it's local meaning as a stability criterion.)
        """
        # we want a "time step" of 1 (turn) since the cavity will contribute to tracking
        # over one turn
        def drift(dp):  return self.drift(bunch, dp)
        def kick(dz):   return self.kick(bunch, dz)
        bunch.dz, bunch.dp = self.integrator(bunch.dz, bunch.dp, 1, drift, kick)

class RFSystems(LongitudinalTracker):
    """
        provides a sequence of (1 or more) RFCavity objects which supports acceleration.
        The signature is the same as for RFCavity except that frequencies, voltages
        and dphi's are passed as lists with as many entries as there are RFCavity objects.
        The first entry of these lists defines the single accelerating RFCavity!
        All other following RFCavity objects are not contributing to acceleration of p0!
        The drifted length of each RFCavity will be 0 except for the first drift
        covering the whole circumference.
        The acceleration method should be applied only once per turn!
        With one RFSystems object per ring layout (with all RFCavity objects at the 
        same longitudinal position) the longitudinal separatrix function 
        is exact and makes a valid local statement about stability!

        self.cavities lists the RFCavity objects in the sequence,
        self.accelerating_cavity gives the accelerating cavity 
            (the first entry of self.cavities, usually the one with the lowest harmonic),
        self.dp_step gives the acceleration Delta p0 for the next turn
            (initialised to 0, should be adjusted (turnwise) by the user!)
    """
    def __init__(self, circumference, gamma_transition, 
                harmonic_list, voltage_list, dphi_list, integrator = symple.Euler_Cromer):

        if not len(harmonic_list) == len(voltage_list) == len(dphi_list):
            print ("Warning: parameter lists for RFSystems do not have the same length!")
        self.cavities = []
        parameters = zip(harmonic_list, voltage_list, dphi_list)
        length = circumference
        for harmonic, voltage, dphi in parameters:
            self.cavities.append(
                            RFCavity(self, circumference, length, gamma_transition, 
                                                    harmonic, voltage, dphi, integrator)
                                )
            length = 0
        self.accelerating_cavity = self.cavities[0]
        self.dp_step = 0
        self.accelerating_cavity._stat_kick = self.accelerating_cavity.kick # need to store other name!
        def acc_kick(dp):

            self.accelerating_cavity._stat_kick()

    def track(self, bunch):
        if self.dp_step:
            assert (self.integrator is symple.Euler-Cromer)
            gamma_old   = bunch.gamma
            beta_old    = bunch.beta
            p0_old      = bunch.p0
            bunch.p0    = p0_old + self.dp_step # updates bunch.gamma and bunch.beta as well!
            geo_emittance_factor = np.sqrt(gamma_old * beta_old / (bunch.gamma * bunch.beta))
            bunch.x    *= geo_emittance_factor
            bunch.xp   *= geo_emittance_factor
            bunch.y    *= geo_emittance_factor
            bunch.yp   *= geo_emittance_factor
        for cavity in self.cavities:
            cavity.track(bunch)

    def potential(self, dz, bunch):
        """gathers the potentials of the rf system and returns the sum as the total potential"""
        def fetch_potential(cavity):
            return cavity.potential(dz, bunch)
        potential_list = map(fetch_potential, self.cavities)
        return sum(potential_list)

    def hamiltonian(self, dz, dp, bunch):
        """the full separable Hamiltonian of the rf system"""
        kinetic = -0.5 * self.eta(bunch) * bunch.beta * c * dp ** 2
        return kinetic + self.potential(dz, bunch)

    # def separatrix(self, dz, bunch):
    #     pass

    def isin_separatrix(self, dz, dp, bunch):
        # the envisaged way to do it:
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
        - 
        for the moment, acceleration only works for Euler-Cromer,
        as the other multi-step integrators have to be adapted to
        change parameters (adapted to respective gammas) during their integration steps!"""

        
        def acc_kick(dz) -------------> this complete function accelerate to should go into track!

    def phi_synchronous(self, bunch, voltage):
        """calculates the synchronous phase for the given acceleration situation (self.dp_step etc.)"""
        # from HEADTAIL:
        # phi_s = arccos( sgn(eta) * sqrt( 1 - (prate * circumference / e / voltage) ** 2 ) )
        # prate = m * c * gammarate / beta
        # gammarate = delta_gamma * f0
        # f0 = beta * c / circumference
        deltaE = self.dp_step * beta * c
        sgn_eta = np.sign( self.eta(bunch) )
        return np.arccos( sgn_eta * np.sqrt( 1 - (deltaE / (e * voltage)) ** 2 ) )
