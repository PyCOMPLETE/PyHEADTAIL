'''
@author Kevin Li, Adrian Oeftiger, Michael Schenk
@date 03.10.2014
@copyright CERN
'''
from __future__ import division

import numpy as np
from types import MethodType
from scipy.constants import c

from ..general.decorators import deprecated
from . import Element, clean_slices, utils
from rf_bucket import RFBucket, attach_clean_buckets

from abc import ABCMeta, abstractmethod

from ..general import pmath as pm


# @TODO
# think about flexible design to separate numerical methods
# and physical parameters (as before for the libintegrators.py)
# while satisfying this design layout.
# currently: only Velocity Verlet algorithm hard coded in RFSystems


class LongitudinalMap(Element):
    """
    A longitudinal map represents a longitudinal dynamical element
    (e.g. a kick or a drift...), i.e. an abstraction of a cavity
    of an RF system etc.
    Any track method of a longitudinal element should clean the slices
    from the beam -- use @clean_slices!
    LongitudinalMap objects can compose a longitudinal one turn map!
    Definitions of various orders of the slippage factor eta(delta)
    for delta = (p - p0) / p0 should be implemented in this class.
    Any derived objects will access self.eta(delta, gamma).

    Note: the momentum compaction factors are defined by the change of radius
    \Delta R / R0 = \sum_i \\alpha_i * \delta^(i + 1)
    hence yielding expressions for the higher slippage factor orders
    \Delta w / w0 = \sum_j  \eta_j  * \delta^(i + 1)
    (for the revolution frequency w)
    """
    __metaclass__ = ABCMeta

    def __init__(self, alpha_array, *args, **kwargs):
        """
        The length of the momentum compaction factor array /alpha_array/
        defines the order of the slippage factor expansion.
        """
        super(LongitudinalMap, self).__init__(*args, **kwargs)
        self.alpha_array = alpha_array

    @abstractmethod
    def track(self, beam):
        '''Should be decorated by @clean_slices for any inheriting
        classes changing beam.z .
        '''
        pass

    def eta(self, dp, gamma):
        """
        Depending on the number of entries in self.alpha_array the
        according order of \eta = \sum_i \eta_i * \delta^i where
        \delta = \Delta p / p0 will be included in this gathering function.

        Note: Please implement higher slippage factor orders as static methods
        with name _eta<N> where <N> is the order of delta in eta(delta)
        and with signature (alpha_array, gamma).
        """
        eta = 0
        for i in xrange(len(self.alpha_array)):   # order = len - 1
            eta_func = getattr(self, '_eta' + str(i))
            eta_i = eta_func(self.alpha_array, gamma)
            eta += eta_i * (dp ** i)
        return eta

    @staticmethod
    def _eta0(alpha_array, gamma):
        return alpha_array[0] - gamma**-2


class Drift(LongitudinalMap):
    """
    The drift (i.e. Delta z) of the particle's z coordinate is given by
    the (separable) Hamiltonian derived by dp (defined by (p - p0) / p0).

    self.length is the drift length,
    self.shrinkage_p_increment being non-zero includes the shrinking
    ratio \beta_{n+1} / \beta_n (see MacLachlan 1989 in FN-529),
    it is usually neglected. [Otherwise it may continuously be
    adapted by the user according to the total momentum increment.]
    If it is not neglected, the beta factor ratio would yield
    (\beta + \Delta \beta) / \beta =
                        = 1 - \Delta \gamma / (\beta^2 * \gamma^2)
    resp.               = 1 - p_increment / (\gamma^3 * p0)
    since p_increment = \gamma * m * c / \beta * \Delta gamma .
    """

    def __init__(self, alpha_array, length, shrinkage_p_increment=0,
                 *args, **kwargs):
        super(Drift, self).__init__(alpha_array, *args, **kwargs)
        self.length = length
        self.shrinkage_p_increment = shrinkage_p_increment

    @clean_slices
    def track(self, beam):
        beta_ratio = 1 - self.shrinkage_p_increment / (beam.gamma**3 * beam.p0)
        beam.z = (beta_ratio * beam.z -
                  self.eta(beam.dp, beam.gamma) * beam.dp * self.length)


class Kick(LongitudinalMap):
    """
    The Kick class represents the kick by a single RF element
    in a ring! The kick (i.e. dp_{n+1} - dp_n) of the particle's dp
    coordinate is given by the (separable) Hamiltonian derived
    by z, i.e. the force.

    self.p_increment is the momentum step per turn added by this Kick,
    it can be continuously adjusted externally
    by the user to reflect different slopes in the dipole field ramp.

    self.phi_offset reflects an offset of the cavity's reference system,
    this can be tweaked externally by the user for simulating RF system
    ripple and the like. Include the change of flank of the sine curve
    here, explicitely (i.e. pi below transition and 0 above transition).

    (self._phi_lock adds to the offset as well but should
    be used internally in the module (e.g. by RFSystems) for
    acceleration purposes. It may be used for synchronisation with the
    momentum updating by self.p_increment via self.calc_phi_0(beam),
    thus readjusting the zero-crossing of this sinosoidal kick.
    This requires a convention how to mutually displace the Kick
    phases to each other w.r.t. to their contribution to acceleration.)
    """

    def __init__(self, alpha_array, circumference, harmonic, voltage,
                 phi_offset=0, p_increment=0, D_x=0, D_y=0, *args, **kwargs):
        '''D_x, D_y: horizontal and vertical dispersion

        !! Attention !!
        The user is responsible of making sure the dispersions match the
        dispersions of the beam which were added in the track() of the last map.
        This corresponds to the dispersion of the transverse/longitudinal map
        !following! this Kick (D_x_s1 of the preceding transverse map)
        Example:
        dx = np.array([1, 2., 5]) #the dispersions at the TransverseSegmentMaps
        trans_map = TransverseMap(C, segments, ax, bx, dx, ay, by, Q_x, Q_y)
        map_ = [m for m in trans_map] + [LinearMap(alpha_0, C, Q_s, D_x=???)]
        D_x = 1. # if we place the LinearMap after the transverse maps, we
                 # need to make sure the dispersion matches the dispersion
                 # added in this transverse map. This corresponds to the
                 # dispersion of the first segment of the transverse map!

        Or simply: Match the dispersion of this Element with the dispersion
        of the following transverse element.
        '''
        super(Kick, self).__init__(alpha_array, *args, **kwargs)
        self.circumference = circumference
        self.harmonic = harmonic
        self.voltage = voltage
        self.phi_offset = phi_offset
        self.p_increment = p_increment
        self._phi_lock = 0
        self.D_x = D_x
        self.D_y = D_y
        if not D_x and not D_y:
            self.track = self.track_without_dispersion

    def track(self, beam):
        try:
            self.track_with_dispersion(beam)
            self.track = self.track_with_dispersion
        except AttributeError as e:
            self.warns("Failed to apply transverse dispersion correction "
                       "during longitudinal tracking. Caught AttributeError: \n"
                       + e.message + "\nContinue without adjusting dispersion "
                       "contribution when changing beam.dp...")
            self.track = self.track_without_dispersion

    def track_with_dispersion(self, beam):
        ''' Subtract the dispersion before computing a new dp, then add
        the dispersion using the new dp.
        '''
        beam.x -= self.D_x*beam.dp
        beam.y -= self.D_y*beam.dp

        self.track_without_dispersion(beam)

        beam.x += self.D_x*beam.dp
        beam.y += self.D_y*beam.dp

    def track_without_dispersion(self, beam):
        amplitude = np.abs(beam.charge)*self.voltage / (beam.beta*c)
        phi = (self.harmonic * (2*np.pi*beam.z/self.circumference)
               + self.phi_offset + self._phi_lock)

        delta_p = beam.dp * beam.p0
        delta_p += amplitude * pm.sin(phi) - self.p_increment
        beam.p0 += self.p_increment
        beam.dp = delta_p / beam.p0

    # @property
    # def parameters(self):
    #     return (self.harmonic, self.voltage, self.phi_offset, self.p_increment)
    # @parameters.setter
    # def parameters(self, value):
    #     self.harmonic = value[0]
    #     self.voltage = value[1]
    #     self.phi_offset = value[2]
    #     self.p_increment = value[3]

    # def Qs(self, gamma):
    #     '''
    #     Synchrotron tune derived from the linearized Hamiltonian

    #     .. math::
    #     H = -1/2*eta*beta*c * delta ** 2 + e*V /(p0*2*np.pi*h)
    #       * ( np.cos(phi)-np.cos(dphi) + (phi-dphi) * np.sin(dphi) )
    #     NOTE: This function only returns the synchroton tune effectuated
    #     by this single Kick instance, any contribution from other Kick
    #     objects is not taken into account! (I.e. in general, this
    #     calculated value is wrong for multi-harmonic RF systems.)
    #     '''
    #     beta = np.sqrt(1 - gamma**-2)
    #     p0 = m_p * np.sqrt(gamma**2 - 1) * c
    #     return np.sqrt( e * self.voltage * np.abs(self.eta(0, gamma))
    #                    * self.harmonic / (2*np.pi*p0*beta*c) )

    # def phi_s(self, gamma):
    #     """The phase deviation from the unaccelerated case
    #     calculated via the momentum step self.p_increment
    #     per turn. It includes the jump in the e.o.m.
    #     (via sign(eta)) at transition energy:
    #         gamma < gamma_transition <==> phi_0 ~ pi
    #         gamma > gamma_transition <==> phi_0 ~ 0
    #     In the case of only one Kick element in the ring, this phase
    #     deviation coincides with the synchronous phase!
    #     """
    #     if self.p_increment == 0 and self.voltage == 0:
    #         return 0
    #     beta = np.sqrt(1 - gamma**-2)
    #     deltaE = self.p_increment * beta * c
    #     phi_rel = np.arcsin(deltaE / (e * self.voltage))

    #     if self.eta(0, gamma)<0:
    #         # return np.sign(deltaE) * np.pi - phi_rel
    #         return np.pi - phi_rel
    #     else:
    #         return phi_rel


class LongitudinalOneTurnMap(LongitudinalMap):
    """
    A longitudinal one turn map tracks over a complete turn.
    Any inheriting classes guarantee to provide a self.track(beam)
    method that tracks around the whole ring!

    LongitudinalOneTurnMap classes possibly comprise several
    LongitudinalMap objects.
    """

    __metaclass__ = ABCMeta

    def __init__(self, alpha_array, circumference, *args, **kwargs):
        """LongitudinalOneTurnMap objects know their circumference."""
        super(LongitudinalOneTurnMap, self).__init__(
            alpha_array, *args, **kwargs)
        self.circumference = circumference

    @abstractmethod
    def track(self, beam):
        """
        Contract: advances the longitudinal coordinates
        of the beam over a full turn / circumference.
        """
        pass


class RFSystems(LongitudinalOneTurnMap):
    """
    With one RFSystems object in the ring layout (with all Kick
    objects located at the same longitudinal position), the
    longitudinal separatrix function is exact and makes a valid
    local statement about stability!
    """

    def __init__(self, circumference, harmonic_list, voltage_list,
                 phi_offset_list, alpha_array, gamma_reference,
                 p_increment=0, phase_lock=True,
                 shrink_transverse=True, shrink_longitudinal=False,
                 D_x=0, D_y=0, charge=None, mass=None,
                 *args, **kwargs):
        """
        The first entry in harmonic_list, voltage_list and
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

        The boolean parameter shrink_longitudinal determines whether the
        shrinkage ratio \\beta_{n+1} / \\beta_n should be taken
        into account during the second Drift.
        (See the Drift class for further details.)

        The boolean parameter shrink_transverse allows for transverse
        emittance cooling from acceleration.

        Arguments:
        - self.p_increment is the momentum step per turn of the
        synchronous particle, it can be continuously adjusted to
        reflect different slopes in the dipole magnet strength ramp.
        (See the Kick class for further details.)
        - phase_lock == True means all phi_offsets are given w.r.t. the
        fundamental kick, adjusted at set-up time.
        - phase_lock == False means all phi_offsets are absolute.
        In this case take care about all Kick.p_increment attributes --
        highly non-trivial, as all other p_increment functionality
        in RFSystems is broken. So take care, you're on your own! :-)
        - D_x, D_y: horizontal and vertical dispersion. These arguments
          are passed to the Kicks class. Because both kicks are applied
          consecutively, the dispersion will be the same for both kicks and
          it is therefore sufficient to specify only one dispersion.
          The dispersion must match the dispersion of the following transverse
          map. See the docstring of the Kick class for a more detailed
          description.
        """

        self.charge = charge
        self.mass = mass
        self.gamma = gamma_reference

        super(RFSystems, self).__init__(
            alpha_array, circumference, *args, **kwargs)

        if not len(harmonic_list) == len(voltage_list) == len(phi_offset_list):
            self.warns("Parameter lists for RFSystems do not have the " +
                       "same length!")

        self._shrinking = shrink_longitudinal
        if not shrink_transverse:
            self.track = self.track_no_transverse_shrinking

        self._kicks = [Kick(alpha_array, self.circumference, h, V, dphi,
                            D_x=D_x, D_y=D_y) for h, V, dphi in
                       zip(harmonic_list, voltage_list, phi_offset_list)]
        self._elements = ([Drift(alpha_array, 0.5 * self.circumference)] +
                          self._kicks +
                          [Drift(alpha_array, 0.5 * self.circumference)])
        self._accelerating_kick = self._kicks[0]
        self._shrinking_drift = self._elements[-1]

        self.p_increment = p_increment

        if phase_lock:
            self._phaselock(gamma_reference, charge)

        '''Take care of possible memory leakage when accessing with many
        different gamma values without tracking while setting
        RFSystems.p_increment != 0. Many dict entries will be generated!
        '''
        self._rfbuckets = {}

        self._voltages = utils.ListProxy(self._kicks, "voltage")
        self._harmonics = utils.ListProxy(self._kicks, "harmonic")
        self._phi_offsets = utils.ListProxy(self._kicks, "phi_offset")
        # SORRY FOR THE FOLLOWING CODE!
        # any write access to the above ListProxy instances changes
        # Kick parameters and should therefore result in
        # discarding all saved RFBucket instances:
        voltages_setitem = attach_clean_buckets(
            self._voltages._rewritable_setitem, self)
        harmonics_setitem = attach_clean_buckets(
            self._harmonics._rewritable_setitem, self)
        phi_offsets_setitem = attach_clean_buckets(
            self._phi_offsets._rewritable_setitem, self)

        self._voltages._rewritable_setitem = MethodType(
            voltages_setitem, self._voltages)
        self._harmonics._rewritable_setitem = MethodType(
            harmonics_setitem, self._harmonics)
        self._phi_offsets._rewritable_setitem = MethodType(
            phi_offsets_setitem, self._phi_offsets)
        # any suggestions to improve readibility highly appreciated :-)

    @property
    def voltages(self):
        """List of Kick voltages, ONLY use this interface in RFSystems
        to access and modify any Kick voltage.
        (Otherwise the get_bucket functionality is broken,
         clean_buckets will not be called if not using this interface.)
        """
        return self._voltages

    @voltages.setter
    def voltages(self, value):
        self.voltages[:] = value

    @property
    def harmonics(self):
        """List of Kick harmonics, ONLY use this interface in RFSystems
        to access and modify any Kick harmonic.
        (Otherwise the get_bucket functionality is broken,
        clean_buckets will not be called if not using this interface.)
        """
        return self._harmonics

    @harmonics.setter
    def harmonics(self, value):
        self.harmonics[:] = value

    @property
    def phi_offsets(self):
        """List of Kick phi_offsets, ONLY use this interface in
        RFSystems to access and modify any Kick phi_offset.
        (Otherwise the get_bucket functionality is broken,
        clean_buckets will not be called if not using this interface.)
        """
        return self._phi_offsets

    @phi_offsets.setter
    def phi_offsets(self, value):
        self.phi_offsets[:] = value

    @property
    def p_increment(self):
        """The increment in momentum of the accelerating Kick, i.e.
        defined by the first entry in the RF parameter lists.
        ONLY use this interface in RFSystems to access and modify
        the accelerating Kick.p_increment.
        (Otherwise the get_bucket functionality is broken,
         clean_buckets will not be called if not using this interface.)
        """
        return self._accelerating_kick.p_increment

    @p_increment.setter
    def p_increment(self, value):
        self.clean_buckets()
        self._accelerating_kick.p_increment = value
        if self._shrinking:
            self._shrinking_drift.shrinkage_p_increment = value

    @property
    def Q_s(self):
        return self.get_bucket(charge=self.charge, mass=self.mass,
                               gamma=self.gamma).Q_s

    def pop_kick(self, index):
        '''Remove a Kick instance from this RFSystems instance.
        Return the removed Kick instance.
        Arguments:
            - index: the index according to the defining lists
            voltages, harmonics, phi_offsets.
        Note: can only remove kicks that are not index == 0.
        The accelerating / fundamental kick cannot be removed.
        '''
        if index == 0:
            raise ValueError('Cannot remove accelerating / fundamental kick, '
                             'remove_kick(0) is not accepted.')
        kick = self._kicks.pop(index)
        return kick

    def get_bucket(self, bunch=None, gamma=None, mass=None, charge=None,
                   *args, **kwargs):
        '''Return an RFBucket instance which contains all information
        and all physical parameters of the current longitudinal RF
        configuration. (Factory method)

        Use for plotting or obtaining the Hamiltonian etc.

        Attention: For the moment it is assumed that only the
        accelerating kick (defined by the first entry in the
        parameter lists) has a non-zero p_increment.
        (see RFSystems.p_increment)

        Arguments:
        Either give a bunch or the three parameters
        (gamma, mass, charge) explicitely to return a bucket
        defined by these.
        '''

        if charge is None:
            charge = self.charge

        if mass is None:
            mass = self.mass

        try:
            bunch_signature = (bunch.gamma, bunch.mass, bunch.charge)
        except AttributeError:
            if gamma is None and bunch is not None:
                # probably someone is still using the old interface
                gamma = bunch
                self.warns('Are you trying RFSystems.get_bucket(gamma)? ' +
                           'Either give a bunch to the first argument slot ' +
                           'or use a keyword: get_bucket(gamma=gamma) . ' +
                           "I'm helping you out for now with bunch = gamma.")
            bunch_signature = (gamma, mass, charge)
        if bunch_signature not in self._rfbuckets:
            self._rfbuckets[bunch_signature] = RFBucket(
                self.circumference, bunch_signature[0], bunch_signature[1],
                bunch_signature[2], self.alpha_array, self.p_increment,
                list(self.harmonics), list(self.voltages),
                list(self.phi_offsets))
        return self._rfbuckets[bunch_signature]

    def clean_buckets(self):
        '''Erases all RFBucket records of this RFSystems instance.
        Any change of the Kick parameters should entail calling
        clean_buckets in order to update the Hamiltonian etc.
        '''
        self._rfbuckets = {}

    def phi_s(self, gamma, charge):
        beta = np.sqrt(1 - gamma**-2)
        eta0 = self.eta(0, gamma)

        V = self._accelerating_kick.voltage

        if self.p_increment == 0 and V == 0:
            return 0

        deltaE = self.p_increment * beta * c
        phi_rel = np.arcsin(deltaE / (np.abs(charge) * V))

        return phi_rel

    @staticmethod
    def _shrink_transverse_emittance(beam, geo_emittance_factor):
        """Account for the transverse geometrical emittance shrinking
        due to acceleration cooling.
        """
        beam.x *= geo_emittance_factor
        beam.xp *= geo_emittance_factor
        beam.y *= geo_emittance_factor
        beam.yp *= geo_emittance_factor

    def track(self, beam):
        try:
            self.track_transverse_shrinking(beam)
            self.track = self.track_transverse_shrinking
        except AttributeError as e:
            self.warns("Failed to apply transverse acceleration " +
                       "cooling. Caught AttributeError: \n" +
                       e.message + "\nContinue without shrinking " +
                       "transverse emittance...")
            self.track = self.track_no_transverse_shrinking

    def track_transverse_shrinking(self, beam):
        betagamma_old = beam.betagamma
        for longMap in self._elements:
            longMap.track(beam)
        if self.p_increment:
            self._shrink_transverse_emittance(
                beam, np.sqrt(betagamma_old / beam.betagamma))
            self.clean_buckets()

    def track_no_transverse_shrinking(self, beam):
        for longMap in self._elements:
            longMap.track(beam)
        if self.p_increment:
            self.clean_buckets()

    def _phaselock(self, gamma, charge):
        '''Put all _kicks other than the accelerating kick to
        zero phase difference w.r.t. the accelerating kick.
        Attention: Make sure the p_increment of each non-accelerating
        kick is set to 0 (assuming phi_offset == 0, otherwise adapt!).
        '''
        acc = self._accelerating_kick
        non_accelerating_kicks = (k for k in self._kicks if k is not acc)

        for kick in non_accelerating_kicks:
            kick._phi_lock -= (kick.harmonic/acc.harmonic *
                               self.phi_s(gamma, charge))


    # --- INTERFACE CHANGE notifications to update users of previous versions
    # --- to use the right new methods
    @property
    def rfbucket(self):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the get_bucket(beam) method ' +
                           'to obtain a (constant) blueprint of the ' +
                           'current state of the ' +
                           'longitudinal RF configuration.')

    @property
    def kicks(self):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the voltages, harmonics or phi_offsets ' +
                           'lists of RFSystems to access the kick parameters.')

    @property
    def elements(self):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the voltages, harmonics or phi_offsets ' +
                           'lists of RFSystems to access the kick parameters.')

    @property
    def fundamental_kick(self):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the voltages, harmonics or phi_offsets ' +
                           'lists of RFSystems to access the kick parameters.')

    @property
    def accelerating_kick(self):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the first entries of the voltages, harmonics ' +
                           'or phi_offsets lists of RFSystems to access the ' +
                           'kick parameters.')

    def set_voltage_list(self, voltage_list):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the list RFSystems.voltages .')

    def set_harmonic_list(self, harmonic_list):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the list RFSystems.harmonics .')

    def set_phi_offset_list(self, phi_offset_list):
        '''non-existent anymore!'''
        raise RuntimeError('Interface change: ' +
                           'Use the list RFSystems.phi_offsets .')

    # # should be replaced by something more general:
    # # (something along the lines of
    # #  summing over h_i * V_i * sin(phi_i?))
    # def Qs(self, gamma):
    #     beta = np.sqrt(1 - gamma**-2)
    #     p0 = m_p*np.sqrt(gamma**2 - 1)*c
    #     eta0 = self.eta(0, gamma)

    #     fc = self.fundamental_kick
    #     V = fc.voltage
    #     h = fc.harmonic

    #     return np.sqrt( e*V*np.abs(eta0)*h /
    #                    (2*np.pi*self.p0_reference*self.beta_reference*c) )


class LinearMap(LongitudinalOneTurnMap):
    '''
    Linear Map represented by a Courant-Snyder transfer matrix.
    Makes use only of the linear first order slippage factor eta.
    Higher orders are manifestly neglected:

    .. math::
    \eta(\delta = 0) = \sum_i \eta_i * \delta^i === \eta_0

    where

    .. math::
    \eta_0 := 1 / gamma_{tr}^2 - 1 / gamma^2
    '''

    def __init__(self, alpha_array, circumference, Q_s, D_x=0, D_y=0,*args, **kwargs):
        '''Q_s is the synchrotron tune.
        D_x, D_y are the dispersions in horizontal and vertical direction.

        !! Attention !!
        The user is responsible of making sure the dispersions match the
        dispersions of the beam which were added in the track() of the last map.
        This corresponds to the dispersion of the transverse/longitudinal map
        !following! this LinearMap (D_x_s1 of the preceding transverse map)
        Example:
        dx = np.array([1, 2., 5]) #the dispersions at the TransverseSegmentMaps
        trans_map = TransverseMap(C, segments, ax, bx, dx, ay, by, Q_x, Q_y)
        map_ = [m for m in trans_map] + [LinearMap(alpha_0, C, Q_s, D_x=???)]
        D_x = 1. # if we place the LinearMap after the transverse maps, we
                 # need to make sure the dispersion matches the dispersion
                 # added in this transverse map. This corresponds to the
                 # dispersion of the first segment of the transverse map!

        Or simply: Match the dispersion of this Element with the dispersion
        of the following transverse element.

        '''
        super(LinearMap, self).__init__(alpha_array, circumference,
                                        *args, **kwargs)
        assert (len(np.atleast_1d(Q_s)) == 1), "Q_s can only have one entry!"
        self.Q_s = Q_s
        self.D_x = D_x
        self.D_y = D_y
        if len(alpha_array) > 1:
            self.warns('The higher orders in the given alpha_array are ' +
                       'manifestly neglected.')

    @property
    @deprecated('--> Now becomes "Q_s"\n')
    def Qs(self):
        return self.Q_s

    def track(self, beam):
        try:
            self.track_with_dispersion(beam)
            self.track = self.track_with_dispersion
        except AttributeError as e:
            self.warns("Failed to apply transverse dispersion correction "
                       "during longitudinal tracking. Caught AttributeError: \n"
                       + e.message + "\nContinue without adjusting dispersion "
                       "contribution when changing beam.dp...")
            self.track = self.track_without_dispersion

    def track_with_dispersion(self, beam):
        ''' Subtract the dispersion before computing a new dp, then add
        the dispersion using the new dp.
        '''
        beam.x -= self.D_x*beam.dp
        beam.y -= self.D_y*beam.dp

        self.track_without_dispersion(beam)

        beam.x += self.D_x*beam.dp
        beam.y += self.D_y*beam.dp

    @clean_slices
    def track_without_dispersion(self, beam):
        omega_0 = 2 * np.pi * beam.beta * c / self.circumference
        omega_s = self.Q_s * omega_0

        dQ_s = 2 * np.pi * self.Q_s
        cosdQ_s = np.cos(dQ_s)  # use np because dQ_s is always a scalar
        sindQ_s = np.sin(dQ_s)  # use np because dQ_s is always a scalar

        z0 = beam.z
        dp0 = beam.dp

        # self.eta(0, beam.gamma) is identical to using first order eta!
        longfac = self.eta(0, beam.gamma) * beam.beta * c / omega_s

        beam.z = z0 * cosdQ_s - longfac * dp0 * sindQ_s
        beam.dp = dp0 * cosdQ_s + z0 / longfac * sindQ_s
