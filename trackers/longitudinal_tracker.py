from __future__ import division
'''
@author Adrian Oeftiger
@date 26.05.2014
@copyright CERN
'''


import numpy as np

from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e

sin = np.sin
cos = np.cos

class LongitudinalMap(object):
    """A longitudinal map represents a longitudinal dynamical element 
    (e.g. a kick or a drift...), i.e. an abstraction of a single cavity 
    of an RF system etc.
    LongitudinalMap objects can compose a longitudinal one turn map!
    Definitions of various orders of the slippage factor eta(dp)
    should be implemented in this class. Any derived objects will access self.eta(beam).
    
    Note: the momentum compaction factors are defined by the change of radius
        Delta R / R0 = \sum_i alpha_i * (Delta p / p0)^i
        hence yielding expressions for the higher order slippage factors
        Delta w / w0 = \sum_j  eta_j  * (Delta p / p0)^i
        (for the revolution frequency w)
    """

    self.__metaclass__ = ABCMeta

    def __init__(self, alpha_array):
        """The length of the momentum compaction factor array /alpha_array/
        defines the order of the slippage factor expansion. 
        Depending on the number of entries in /alpha_array/ the according definition
        of eta will be chosen for the wrapper self.eta(beam).

        Note: Please implement higher order slippage factors as static methods
        with name _eta<N> where <N> is the order of delta in (eta * delta)
        and with signature (alpha_array, beam)."""
        eta_chosen = '_eta' + str(len(alpha_array))
        self.eta = lambda beam: getattr(self, eta_chosen)(alpha_array, beam)

    @abstractmethod
    def track(self, beam):
        pass

    @staticmethod
    def _eta0(alpha_array, beam):
        return 0

    @staticmethod
    def _eta1(alpha_array, beam):
        return alpha_array[0] - beam.gamma ** -2

class Drift(LongitudinalMap):
    """the drift (i.e. Delta z) of the particle's z coordinate is given by
    the (separable) Hamiltonian derived by dp (defined by (p - p0) / p0).

    self.length is the drift length."""

    def __init__(self, length):
        self.length = length

    def track(self, beam):
        beam.dz += -self.eta(beam) * beam.dp * self.length

class Kick(LongitudinalMap):
    """The Kick class represents the kick by a single RF element in a ring!
    The kick (i.e. Delta dp) of the particle's dp coordinate is given by
    the (separable) Hamiltonian derived by z, i.e. the force.

    self.p_increment is the momentum step per turn of the synchronous particle,
        it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.

    self.phi_offset reflects an offset of the cavity's reference system."""

    def __init__(self, harmonic, voltage, phi_offset = 0, p_increment = 0):
        self.harmonic   = harmonic
        self.voltage    = voltage
        self.phi_offset = phi_offset
        self.p_increment = p_increment

    def track(self, beam):
        sgn_eta     = np.sign(self.eta(beam))
        amplitude   = sgn_eta * e * self.voltage / (beam.beta * c)
        Phi         = self.harmonic * beam.theta + self.phi_offset

        beam.Deltap += amplitude * sin(Phi) - self.p_increment
        beam.p0    += self.p_increment

    def Qs(self, beam):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           + e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(dphi) + (phi - dphi) * np.sin(dphi))
        ASSUMPTION: this is the only Kick instance in the ring layout 
            (i.e. only a single harmonic RF system)!
        '''
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(beam)) * self.harmonic \
                / (2 * np.pi * beam.p0 * beam.beta * c))
        return Qs

    def Phi_0(self):
        """The synchronous phase calculated from the momentum increase per turn.
        It includes the jump in the e.o.m. (via sign(eta)) at transition energy:
            gamma < gamma_transition <==> Phi_0 ~ pi
            gamma > gamma_transition <==> Phi_0 ~ 0
        ASSUMPTION: this is the only Kick instance adding to overall acceleration
            (i.e. technically the only Kick instance with self.p_increment != 0)!"""
        deltaE = self.p_increment * beta * c
        sgn_eta = np.sign( self.eta(beam) )
        return np.arccos( sgn_eta * np.sqrt( 1 - (deltaE / (e * voltage)) ** 2 ) )

    def potential(self, z, beam):
        """The contribution of this kick to the overall potential V(z).
        ASSUMPTION: there is one Kick instance adding to overall acceleration
            (i.e. technically only one Kick instance with self.p_increment != 0)!"""
        Phi = -self.harmonic * z / beam.R + self.phi_offset
        amplitude  = e * self.voltage / (beam.p0 * 2 * np.pi * self.harmonic)
        modulation = (cos(Phi) - cos(self.Phi_0) + (Phi - self.Phi_0) * sin(self.Phi_0))
        return amplitude * modulation



class LongitudinalOneTurnMap(object):
    """A longitudinal one turn map tracks over a complete turn.
    Any inheriting classes guarantee to provide a self.track(beam) method that 
    tracks around the whole ring!

    LongitudinalOneTurnMap classes possibly comprise several LongitudinalMap objects."""

    self.__metaclass__ = ABCMeta

    def __init__(self, circumference):
        """LongitudinalOneTurnMap objects know their circumference: 
            this is THE place to store the circumference in the simulations!"""
        self.circumference = circumference

    @abstractmethod
    def track(self, beam):
        """advances the longitudinal coordinates of the beam over a full turn."""
        pass

class RFSystems(LongitudinalOneTurnMap):
    """
        Having one RFSystems object in the ring layout (with all kicks applied at the 
        same longitudinal position), the longitudinal separatrix function 
        is exact and makes a valid local statement about stability!
    """

    def __init__(self, circumference, harmonic_list, voltage_list, phi_offset_list, 
                    alpha_array, p_increment = 0):
        """The first entry in harmonic_list, volta_list and phi_offset_list
        defines the parameters for the accelerating Kick object 
        (i.e. the accelerating RF system).

        The length of the momentum compaction factor array alpha_array
        defines the order of the slippage factor expansion. 
        See the LongitudinalMap class for further details.

        self.p_increment is the momentum step per turn of the synchronous particle,
        it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.
        See the Kick class for further details."""

        super(RFSystems, self).__init__(circumference)
        self.p_increment = p_increment

        if not len(harmonic_list) == len(voltage_list) == len(dphi_list):
            print ("Warning: parameter lists for RFSystems do not have the same length!")

        self.kicks = []
        for h, V, dphi in zip(harmonic_list, voltage_list, phi_offset_list):
            self.kicks.append( Kick(h, V, dphi) )
        self.kicks[0].p_increment = self.p_increment
        self.elements = [Drift(self.circumference)] + [kicks]

    def track(self, beam):
        if self.p_increment:
            gamma_old   = beam.gamma
            beta_old    = beam.beta
            p0_old      = beam.p0
            self.kicks[0].p_increment = self.p_increment
        for longMap in self.elements:
            longMap.track(beam)
        if self.p_increment:
            geo_emittance_factor = np.sqrt(gamma_old * beta_old / (beam.gamma * beam.beta))
            self._shrink_transverse_emittance(beam, geo_emittance_factor)

    @staticmethod
    def _shrink_transverse_emittance(beam, geo_emittance_factor):
        beam.x    *= geo_emittance_factor
        beam.xp   *= geo_emittance_factor
        beam.y    *= geo_emittance_factor
        beam.yp   *= geo_emittance_factor

    def potential(self, z, beam):
        """the potential well of the rf system"""
        def fetch_potential(kick):
            return kick.potential(z, beam)
        potential_list = map(fetch_potential, self.elements)
        return sum(potential_list)

    def hamiltonian(self, z, dp, beam):
        """the full separable Hamiltonian of the rf system"""
        kinetic = -0.5 * self.eta(beam) * beam.beta * c * dp ** 2
        return kinetic + self.potential(z, beam)

    def Hsep(self, beam):
        """the Hamiltonian value at the separatrix"""
        pass

    def separatrix(self, z, beam):
        pass

    def is_in_separatrix(self, z, dp, beam):
        pass

class LinearMap(LongitudinalOneTurnMap):
    '''
    Linear Map represented by a Courant-Snyder transportation matrix.
    self.alpha is the linear momentum compaction factor.
    '''

    def __init__(self, circumference, alpha, Qs):
        """alpha is the linear momentum compaction factor,
        Qs the synchroton tune."""
        self.circumference = circumference
        self.alpha = alpha
        self.Qs = Qs

    def track(self, beam):

        eta = self.alpha - beam.gamma ** -2

        omega_0 = 2 * np.pi * beam.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = cos(dQs)
        sindQs = sin(dQs)

        z0 = beam.z
        dp0 = beam.dp

        beam.z = z0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        beam.dp = dp0 * cosdQs + omega_s / eta / c * z0 * sindQs

