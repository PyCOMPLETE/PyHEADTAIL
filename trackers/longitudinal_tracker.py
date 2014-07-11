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
        \Delta R / R0 = \sum_i \\alpha_i * \delta^(i + 1)
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
        and with signature (alpha_array, beam).
        """
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

    self.length is the drift length,
    self.shrinkage_p_increment being non-zero includes the shrinking
    ratio \\beta_{n+1} / \\beta_n (see MacLachlan 1989 in FN-529),
    it is usually neglected. [Otherwise it may continuously be
    adapted by the user according to the total momentum increment.]
    If it is not neglected, the beta factor ratio would yield
    (\\beta + \\Delta \\beta) / \\beta =
                        = 1 - \\Delta \\gamma / (\\beta^2 * \\gamma^2)
    resp.               = 1 - p_increment / (\\gamma^3 * p0)
    since p_increment = \\gamma * m * c / \\beta * \\Delta gamma .
    """

    def __init__(self, alpha_array, length, shrinkage_p_increment=0):
        super(Drift, self).__init__(alpha_array)
        self.length = length
        self.shrinkage_p_increment = shrinkage_p_increment

    def track(self, beam):
        beta_ratio = 1 - self.shrinkage_p_increment / (beam.gamma**3 * beam.p0)
        beam.z = (beta_ratio * beam.z -
                  self.eta(beam.dp, beam) * beam.dp * self.length)


class Kick(LongitudinalMap):
    """The Kick class represents the kick by a single RF element
    in a ring! The kick (i.e. Delta dp) of the particle's dp
    coordinate is given by the (separable) Hamiltonian derived
    by z, i.e. the force.

    self.p_increment is the momentum step per turn of the
    synchronous particle, it can be continuously adjusted externally
    by the user to reflect different slopes in the dipole field ramp.

    self.phi_offset reflects an offset of the cavity's reference system,
    this can be tweaked externally by the user for simulating RF system
    ripple and the like.
    (self._phi_acceleration adds to the offset as well but should
    be used internally in the module (e.g. by RFSystems) for
    acceleration purposes. It may be used for synchronisation with the
    momentum updating by self.p_increment via self.calc_phi_0(beam),
    thus readjusting the zero-crossing of this sinosoidal kick.
    This requires a convention how to mutually displace the Kick
    phases to each other w.r.t. to their contribution to acceleration.)
    """

    def __init__(self, alpha_array, circumference, harmonic, voltage,
                 phi_offset=0, p_increment=0):
        super(Kick, self).__init__(alpha_array)
        self.circumference = circumference
        self.harmonic = harmonic
        self.voltage = voltage
        self.phi_offset = phi_offset
        self.p_increment = p_increment
        self._phi_acceleration = 0

    def track(self, beam):
        sgn_eta = np.sign(self.eta(0, beam))
        amplitude = sgn_eta * e * self.voltage / (beam.beta * c)
        theta = (2 * np.pi / self.circumference) * beam.z
        phi = self.harmonic * theta + self.phi_offset + self._phi_acceleration

        beam.delta_p += amplitude * (sin(phi) - sin(self.calc_phi_0(beam))) #self.p_increment
        beam.p0 += self.p_increment

    def potential(self, z, beam, phi_0=None):
        """The contribution of this kick to the overall potential V(z)."""
        sgn_eta = np.sign(self.eta(0, beam))
        amplitude = (sgn_eta * e * self.voltage /
                     (beam.p0 * 2 * np.pi * self.harmonic))
        if phi_0 is None:
            phi_0 = self.calc_phi_0(beam)
        theta = (2 * np.pi / self.circumference) * z
        phi = self.harmonic * theta + self.phi_offset + self._phi_acceleration
        modulation = cos(phi) - cos(phi_0) + (phi - phi_0) * sin(phi_0)
        return amplitude * modulation

    def Qs(self, beam):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
            + e * V / (p0 * 2 * np.pi * h) *
            * (np.cos(phi) - np.cos(dphi) + (phi - dphi) * np.sin(dphi))
        NOTE: This function only returns the synchroton tune effectuated
        by this single Kick instance, any contribution from other Kick
        objects is not taken into account! (I.e. in general, this
        calculated value is wrong for multi-harmonic RF systems.)
        '''
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(0, beam)) *
                    self.harmonic / (2 * np.pi * beam.p0 * beam.beta * c))
        return Qs

    def calc_phi_0(self, beam):
        """The phase deviation from the unaccelerated case
        calculated via the momentum step self.p_increment
        per turn. It includes the jump in the e.o.m.
        (via sign(eta)) at transition energy:
            gamma < gamma_transition <==> phi_0 ~ pi
            gamma > gamma_transition <==> phi_0 ~ 0
        In the case of only one Kick element in the ring, this phase
        deviation coincides with the synchronous phase!
        """
        if self.p_increment == 0 and self.voltage == 0:
            return 0
        deltaE  = self.p_increment * c * beam.beta / beam.gamma
        phi_rel = np.arcsin(deltaE / (e * self.voltage))
        if self.eta(0, beam) < 0:
            # return np.sign(deltaE) * np.pi - phi_rel
            return np.pi - phi_rel
        else:
            return phi_rel
        # sgn_eta = np.sign(self.eta(0, beam))
        # return np.arccos(
        #     sgn_eta * np.sqrt(1 - (deltaE / (e * self.voltage)) ** 2))


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
        of the beam over a full turn / circumference."""
        pass


class RFSystems(LongitudinalOneTurnMap):
    """
        With one RFSystems object in the ring layout (with all Kick
        objects located at the same longitudinal position), the
        longitudinal separatrix function is exact and makes a valid
        local statement about stability!
    """

    def __init__(self, circumference, harmonic_list, voltage_list,
                 phi_offset_list, alpha_array, p_increment=0, shrinking=False):
        """The first entry in harmonic_list, voltage_list and
        phi_offset_list defines the parameters for the one
        accelerating Kick object (i.e. the accelerating RF system).
        For several accelerating Kick objects one would have to
        extend this class and settle for the relative phases
        between the Kick objects! (For one accelerating Kick object,
        all the other Kick objects' zero crossings are displaced by
        the negative phase shift induced by the accelerating Kick.)

        The length of the momentum compaction factor array alpha_array
        defines the order of the slippage factor expansion.
        (See the LongitudinalMap class for further details.)

        RFSystems comprises a half the circumference drift,
        then all the kicks by the RF Systems in one location,
        then the remaining half the circumference drift.
        This Verlet algorithm ("leap-frog" featuring O(n_turn^2) as
        opposed to symplectic Euler-Cromer with O(n_turn)) makes
        sure that the longitudinal phase space is read out in
        a symmetric way (otherwise phase space should be tilted
        at the entrance or exit of the cavity / kick location!
        cf. discussions with Christian Carli).

        The boolean parameter shrinking determines whether the
        shrinkage ratio \\beta_{n+1} / \\beta_n should be taken
        into account during the second Drift.
        (See the Drift class for further details.)

        - self.p_increment is the momentum step per turn of the
        synchronous particle, it can be continuously adjusted to
        reflect different slopes in the dipole magnet strength ramp.
        (See the Kick class for further details.)
        - self.kicks is a list of the Kick objects (defined by the
        respective lists in the constructor)
        - self.accelerating_kick returns the first Kick object in
        self.kicks which carries the only p_increment != 0
        - self.elements is comprised of a half turn Drift, self.kicks,
        and another half turn Drift
        - self.fundamental_kick returns the Kick object with the lowest
        harmonic of the revolution frequency
        """

        super(RFSystems, self).__init__(alpha_array, circumference)

        if not len(harmonic_list) == len(voltage_list) == len(phi_offset_list):
            print ("Warning: parameter lists for RFSystems " +
                                        "do not have the same length!")

        self._shrinking = shrinking
        self.kicks = []
        for h, V, dphi in zip(harmonic_list, voltage_list, phi_offset_list):
            kick = Kick(alpha_array, self.circumference, h, V, dphi)
            self.kicks.append(kick)
        self.elements = ( [Drift(alpha_array, self.circumference / 2)]
                        + self.kicks
                        + [Drift(alpha_array, self.circumference / 2)]
                        )
        self.accelerating_kick = self.kicks[0]
        self.p_increment = p_increment
        self.fundamental_kick = min(self.kicks, key=lambda kick: kick.harmonic)

    def track(self, beam):
        if self.p_increment:
            betagamma_old = beam.betagamma
        for longMap in self.elements:
            longMap.track(beam)
        if self.p_increment:
            self._shrink_transverse_emittance(
                beam, np.sqrt(betagamma_old / beam.betagamma))

    @staticmethod
    def _shrink_transverse_emittance(beam, geo_emittance_factor):
        """accounts for the transverse geometrical emittance shrinking"""
        beam.x *= geo_emittance_factor
        beam.xp *= geo_emittance_factor
        beam.y *= geo_emittance_factor
        beam.yp *= geo_emittance_factor

    @property
    def p_increment(self):
        return self.accelerating_kick.p_increment
    @p_increment.setter
    def p_increment(self, value):
        self.accelerating_kick.p_increment = value
        if self._shrinking:
            self.elements[-1].shrinkage_p_increment = value

    def potential(self, z, beam):
        """the potential well of the rf system"""
        phi_0 = self.accelerating_kick.calc_phi_0(beam)
        h1 = self.accelerating_kick.harmonic
        def fetch_potential(kick):
            phi_acc_individual = -kick.harmonic / h1 * phi_0
            if kick is not self.accelerating_kick:
                kick._phi_acceleration = phi_acc_individual
            return kick.potential(z, beam)
        potential_list = map(fetch_potential, self.kicks)
        return sum(potential_list)

    def hamiltonian(self, z, dp, beam):
        """the full separable Hamiltonian of the RF system.
        Its zero value is located at the fundamental separatrix
        (between bound and unbound motion)."""
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

    def __init__(self, circumference, alpha, Qs, *slices):
        """alpha is the linear momentum compaction factor,
        Qs the synchroton tune."""
        self.circumference = circumference
        self.alpha = alpha
        self.Qs = Qs

        self.slices = slices

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

        for slices in self.slices:
            slices.update_slices(beam)
