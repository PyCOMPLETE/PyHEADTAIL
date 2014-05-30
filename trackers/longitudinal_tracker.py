from __future__ import division
'''
@author Adrian Oeftiger
@date 26.05.2014
@copyright CERN
'''

# @TODO
# think about flexible design to separate numerical methods 
# and physical parameters (as before for the libintegrators.py) 
# while satisfying this design layout.
# currently: only Euler Cromer supported in RFSystems


import numpy as np

from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e

sin = np.sin
cos = np.cos

class LongitudinalMap(object):
    """A longitudinal map represents a longitudinal dynamical element 
    (e.g. a kick or a drift...), i.e. an abstraction of a cavity 
    of an RF system etc.
    LongitudinalMap objects can compose a longitudinal one turn map!
    Definitions of various orders of the slippage factor eta(delta)
    for delta = (Delta p / p0) should be implemented in this class. 
    Any derived objects will access self.eta(beam).
    
    Note: the momentum compaction factors are defined by the change of radius
        \Delta R / R0 = \sum_i \alpha_i * \delta^(i + 1)
        hence yielding expressions for the higher slippage factor orders
        \Delta w / w0 = \sum_j  \eta_j  * \delta^(i + 1)
        (for the revolution frequency w)
    """

    __metaclass__ = ABCMeta

    def __init__(self, alpha_array):
        """The length of the momentum compaction factor array /alpha_array/
        defines the order of the slippage factor expansion. """
        self.alpha_array = alpha_array

    @abstractmethod
    def track(self, beam):
        pass

    def eta(self, delta, beam):
        """Depending on the number of entries in self.alpha_array the 
        according order of \eta = \sum_i \eta_i * \delta^i where
        \delta = \Delta p / p0 will be included in this gathering function.

        Note: Please implement higher slippage factor orders as static methods
        with name _eta<N> where <N> is the order of delta in eta(delta)
        and with signature (alpha_array, beam)."""
        eta = 0
        for i in xrange( len(self.alpha_array) ):   # order = len - 1
            eta_i = getattr(self, '_eta' + str(i))(self.alpha_array, beam)
            eta  += eta_i * (delta ** i)
        return eta

    @staticmethod
    def _eta0(alpha_array, beam):
        return alpha_array[0] - beam.gamma ** -2

class Drift(LongitudinalMap):
    """the drift (i.e. Delta z) of the particle's z coordinate is given by
    the (separable) Hamiltonian derived by dp (defined by (p - p0) / p0).

    self.length is the drift length."""

    def __init__(self, alpha_array, length):
        super(Drift, self).__init__(alpha_array)
        self.length = length

    def track(self, beam):
        beam.z += -self.eta(beam.dp, beam) * beam.dp * self.length

class Kick(LongitudinalMap):
    """The Kick class represents the kick by a single RF element in a ring!
    The kick (i.e. Delta dp) of the particle's dp coordinate is given by
    the (separable) Hamiltonian derived by z, i.e. the force.

    self.p_increment is the momentum step per turn of the synchronous particle,
        it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.

    self.phi_offset reflects an offset of the cavity's reference system."""

    def __init__(self, alpha_array, circumference, harmonic, voltage, 
                                    phi_offset = 0, p_increment = 0):
        super(Kick, self).__init__(alpha_array)
        self.circumference = circumference
        self.harmonic   = harmonic
        self.voltage    = voltage
        self.phi_offset = phi_offset
        self.p_increment = p_increment

    def track(self, beam):
        sgn_eta     = np.sign(self.eta(0, beam))
        amplitude   = sgn_eta * e * self.voltage / (beam.beta * c)
        theta       = (2 * np.pi / self.circumference) * beam.z
        Phi         = self.harmonic * theta + self.phi_offset

        beam.Deltap += amplitude * sin(Phi) - self.p_increment
        beam.p0     += self.p_increment

    def Qs(self, beam):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           + e * V / (p0 * 2 * np.pi * h) * 
            * (np.cos(phi) - np.cos(dphi) + (phi - dphi) * np.sin(dphi))
        ASSUMPTION: this is the only Kick instance in the ring layout 
            (i.e. only a single harmonic RF system)!
        '''
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(0, beam)) * \
                    self.harmonic / (2 * np.pi * beam.p0 * beam.beta * c))
        return Qs

    def Phi_0(self, beam):
        """The synchronous phase calculated from the momentum increase per turn.
        It includes the jump in the e.o.m. (via sign(eta)) at transition energy:
            gamma < gamma_transition <==> Phi_0 ~ pi
            gamma > gamma_transition <==> Phi_0 ~ 0
        ASSUMPTION: this is the only Kick instance adding to acceleration
        (i.e. technically the only Kick instance with self.p_increment != 0)!"""
        deltaE  = self.p_increment * c / beam.beta
        sgn_eta = np.sign( self.eta(0, beam) )
        return np.arccos( sgn_eta * \
                            np.sqrt(1 - (deltaE / (e * self.voltage)) ** 2) )

    def potential(self, z, beam):
        """The contribution of this kick to the overall potential V(z).
        ASSUMPTION: there is one Kick instance adding to overall acceleration
        (i.e. technically only one Kick instance with self.p_increment != 0)!"""
        theta = (2 * np.pi / self.circumference) * z
        Phi   = self.harmonic * theta + self.phi_offset
        Phi_0 = self.Phi_0(beam)
        amplitude  = -e * self.voltage / (beam.p0 * 2 * np.pi * self.harmonic)
        modulation = cos(Phi) - cos(Phi_0) + (Phi - Phi_0) * sin(Phi_0)
        return amplitude * modulation



class LongitudinalOneTurnMap(LongitudinalMap):
    """A longitudinal one turn map tracks over a complete turn.
    Any inheriting classes guarantee to provide a self.track(beam) method that 
    tracks around the whole ring!

    LongitudinalOneTurnMap classes possibly comprise several 
    LongitudinalMap objects."""

    __metaclass__ = ABCMeta

    def __init__(self, alpha_array, circumference):
        """LongitudinalOneTurnMap objects know their circumference: 
        this is THE ONE place to store the circumference in the simulations!"""
        super(LongitudinalOneTurnMap, self).__init__(alpha_array)
        self.circumference = circumference

    @abstractmethod
    def track(self, beam):
        """Contract: advances the longitudinal coordinates 
        of the beam over a full turn."""
        pass

class RFSystems(LongitudinalOneTurnMap):
    """
        With one RFSystems object in the ring layout (with all kicks applied 
        at the same longitudinal position), the longitudinal separatrix function 
        is exact and makes a valid local statement about stability!
    """

    def __init__(self, circumference, harmonic_list, voltage_list, 
                        phi_offset_list, alpha_array, p_increment = 0):
        """The first entry in harmonic_list, voltage_list and phi_offset_list
        defines the parameters for the one accelerating Kick object 
        (i.e. the accelerating RF system).

        The length of the momentum compaction factor array alpha_array
        defines the order of the slippage factor expansion. 
        See the LongitudinalMap class for further details.

        self.p_increment is the momentum step per turn of the synchronous 
        particle, it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.
        See the Kick class for further details."""

        super(RFSystems, self).__init__(alpha_array, circumference)
        self.p_increment = p_increment

        if not len(harmonic_list) == len(voltage_list) == len(phi_offset_list):
            print ("Warning: parameter lists for RFSystems " +
                                        "do not have the same length!")

        self.kicks = []
        for h, V, dphi in zip(harmonic_list, voltage_list, phi_offset_list):
            self.kicks.append( Kick(alpha_array, 
                                            self.circumference, h, V, dphi) )
        self.kicks[0].p_increment = self.p_increment
        self.elements = [Drift(alpha_array, self.circumference)] + [self.kicks]
        self.fundamental_kick = max(self.kicks, key = lambda kick: kick.voltage)

    def track(self, beam):
        if self.p_increment:
            betagamma_old   = beam.betagamma
            self.kicks[0].p_increment = self.p_increment
        for longMap in self.elements:
            longMap.track(beam)
        if self.p_increment:
            self._shrink_transverse_emittance(beam, 
                                np.sqrt(betagamma_old / beam.betagamma) )

    @staticmethod
    def _shrink_transverse_emittance(beam, geo_emittance_factor):
        """accounts for the transverse geometrical emittance shrinking"""
        beam.x    *= geo_emittance_factor
        beam.xp   *= geo_emittance_factor
        beam.y    *= geo_emittance_factor
        beam.yp   *= geo_emittance_factor

    def potential(self, z, beam):
        """the potential well of the rf system"""
        def fetch_potential(kick):
            return kick.potential(z, beam)
        potential_list = map(fetch_potential, self.kicks)
        return sum(potential_list)

    def hamiltonian(self, z, dp, beam):
        """the full separable Hamiltonian of the RF system.
        Its zero value is at the fundamental separatrix."""
        kinetic = -0.5 * self.eta(dp, beam) * beam.beta * c * dp ** 2
        return kinetic + self.potential(z, beam)

    def separatrix(self, z, beam):
        """Returns the separatrix delta_sep = (p - p0) / p0 for the synchronous 
        particle (since eta depends on delta, inverting the separatrix equation 
        0 = H(z_sep, dp_sep) becomes inexplicit in general)."""
        return np.sqrt(2 / (beam.beta * c * self.eta(0, beam)) * self.potential(z, beam))

    def is_in_separatrix(self, z, dp, beam):
        """Returns boolean whether this coordinate is located 
        strictly inside the separatrix."""
        return hamiltonian(z, dp, beam) < 0

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

