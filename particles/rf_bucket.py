'''
Just to test matching generator for now...
'''
from __future__ import division


import numpy as np
import pylab as plt
from scipy.optimize import brentq
from scipy.constants import c, e, m_p
from scipy.integrate import dblquad


sin = np.sin
cos = np.cos


# class RFSystems(LongitudinalOneTurnMap):
#     """
#     With one RFSystems object in the ring layout (with all Kick
#     objects located at the same longitudinal position), the
#     longitudinal separatrix function is exact and makes a valid
#     local statement about stability!
#     """

#     def __init__(self, circumference, harmonic_list, voltage_list, phi_offset_list,
#                  alpha_array, gamma_reference, p_increment=0, phase_lock=True,
#                  shrink_transverse=True, shrink_longitudinal=False, slices_tuple=None):
#         """
#         The first entry in harmonic_list, voltage_list and
#         phi_offset_list defines the parameters for the one
#         accelerating Kick object (i.e. the accelerating RF system).
#         For several accelerating Kick objects one would have to
#         extend this class and settle for the relative phases
#         between the Kick objects! (For one accelerating Kick object,
#         all the other Kick objects' zero crossings are displaced by
#         the negative phase shift induced by the accelerating Kick.)

#         The length of the momentum compaction factor array alpha_array
#         defines the order of the slippage factor expansion.
#         (See the LongitudinalMap class for further details.)

#         RFSystems comprises a half the circumference drift,
#         then all the kicks by the RF Systems in one location,
#         then the remaining half the circumference drift.
#         This Verlet algorithm ("leap-frog" featuring O(n_turn^2) as
#         opposed to symplectic Euler-Cromer with O(n_turn)) makes
#         sure that the longitudinal phase space is read out in
#         a symmetric way (otherwise phase space should be tilted
#         at the entrance or exit of the cavity / kick location!
#         cf. discussions with Christian Carli).

#         The boolean parameter shrinking determines whether the
#         shrinkage ratio \\beta_{n+1} / \\beta_n should be taken
#         into account during the second Drift.
#         (See the Drift class for further details.)

#         - self.p_increment is the momentum step per turn of the
#         synchronous particle, it can be continuously adjusted to
#         reflect different slopes in the dipole magnet strength ramp.
#         (See the Kick class for further details.)
#         - self.kicks is a list of the Kick objects (defined by the
#         respective lists in the constructor)
#         - self.accelerating_kick returns the first Kick object in
#         self.kicks which carries the only p_increment != 0
#         - self.elements is comprised of a half turn Drift, self.kicks,
#         and another half turn Drift
#         - self.fundamental_kick returns the Kick object with the lowest
#         harmonic of the revolution frequency
#         """

#         super(RFSystems, self).__init__(alpha_array, circumference)

#         self._shrinking = shrink_longitudinal
#         self._shrink_transverse = shrink_transverse

#         if not len(harmonic_list) == len(voltage_list) == len(phi_offset_list):
#             print ("Warning: parameter lists for RFSystems " +
#                                         "do not have the same length!")

#         self.kicks = [Kick(alpha_array, self.circumference, h, V, dphi)
#                       for h, V, dphi in zip(harmonic_list, voltage_list, phi_offset_list)]
#         self.elements = ( [Drift(alpha_array, self.circumference / 2)]
#                         + self.kicks
#                         + [Drift(alpha_array, self.circumference / 2)]
#                         )
#         self.fundamental_kick = min(self.kicks, key=lambda kick: kick.harmonic)
#         self.p_increment = p_increment

#     @staticmethod
#     def _shrink_transverse_emittance(beam, geo_emittance_factor):
#         """accounts for the transverse geometrical emittance shrinking"""
#         beam.x *= geo_emittance_factor
#         beam.xp *= geo_emittance_factor
#         beam.y *= geo_emittance_factor
#         beam.yp *= geo_emittance_factor

#     def track(self, beam):
#         if self.p_increment:
#             betagamma_old = beam.betagamma
#         for longMap in self.elements:
#             longMap.track(beam)
#         if self.p_increment:
#             try:
#                 self._shrink_transverse_emittance(beam, np.sqrt(betagamma_old / beam.betagamma))
#                 self.track = self.track_transverse_shrinking
#             except AttributeError:
#                 self.track = self.track_no_transverse_shrinking
#             self.p0_reference += self.p_increment

#         if self.slices_tuple:
#             for slices in self.slices_tuple:
#                 slices.update_slices(beam)

#     def track_transverse_shrinking(self, beam):
#         if self.p_increment:
#             betagamma_old = beam.betagamma
#         for longMap in self.elements:
#             longMap.track(beam)
#         if self.p_increment:
#             self._shrink_transverse_emittance(beam, np.sqrt(betagamma_old / beam.betagamma))
#             self.p0_reference += self.p_increment

#         if self.slices_tuple:
#             for slices in self.slices_tuple:
#                 slices.update_slices(beam)

#     def track_no_transverse_shrinking(self, beam):
#         for longMap in self.elements:
#             longMap.track(beam)
#         if self.p_increment:
#             self.p0_reference += self.p_increment

#         if self.slices_tuple:
#             for slices in self.slices_tuple:
#                 slices.update_slices(beam)


class RFBucket(object):

    def __init__(self, circumference, gamma_reference, alpha_0, p_increment,
                 harmonic_list, voltage_list, phi_offset_list, phase_lock=True):

        self.circumference = circumference
        self.gamma_reference = gamma_reference
        # self.get_gamma_reference = closed_orbit.get_gamma_reference
        # self.set_gamma_reference = closed_orbit.set_gamma_reference

        self.alpha_0 = alpha_0
        self.p_increment = p_increment

        self.h = harmonic_list
        self.V = voltage_list
        self.dphi = phi_offset_list

        # # Reference energy and make eta0, resp. "machine gamma_tr" available for all routines
        # self.gamma_reference = gamma_reference
        # self.alpha0 = alpha_array[0]
        if phase_lock:
            self._phaselock()

        zmax = self.circumference / (2*np.amin(harmonic_list))
        self.zmin, self.zmax = -1.01*zmax, +1.01*zmax
        self._get_bucket_boundaries()
        # self.H0_from_sigma = self.H0

    # @property
    # def gamma_reference(self):
    #     return self.get_gamma_reference()
    # @gamma_reference.setter
    # def gamma_reference(self, value):
    #     self.set_gamma_reference(value)
    #     self._beta_reference= np.sqrt(1 - self.gamma_reference**-2)
    #     self._betagamma_reference = np.sqrt(self.gamma_reference**2 - 1)
    #     self._p0_reference = self.betagamma_reference * m_p * c

    @property
    def gamma_reference(self):
        return self._gamma_reference
    @gamma_reference.setter
    def gamma_reference(self, value):
        self._gamma_reference = value
        self._beta_reference= np.sqrt(1 - self._gamma_reference**-2)
        self._betagamma_reference = np.sqrt(self._gamma_reference**2 - 1)
        self._p0_reference = self._betagamma_reference * m_p * c

    @property
    def beta_reference(self):
        return self._beta_reference
    @beta_reference.setter
    def beta_reference(self, value):
        self._gamma_reference = (1. / np.sqrt(1 - value**2))

    @property
    def betagamma_reference(self):
        return self._betagamma_reference
    @betagamma_reference.setter
    def betagamma_reference(self, value):
        self._gamma_reference = (np.sqrt(value ** 2 + 1))

    @property
    def p0_reference(self):
        return self._p0_reference
    @p0_reference.setter
    def p0_reference(self, value):
        self._gamma_reference = (value / (m_p * self.beta_reference * c))

    @property
    def R(self):
        return self.circumference/(2*np.pi)

    @property
    def eta0(self):
        return self.alpha_0 - 1/self.gamma_reference**2

    @property
    def beta_z(self):
        return np.abs(self.eta0 * self.R / self.Qs)

    @property
    def Qs(self):
        ix = np.argmax(self.V)
        V = self.V[ix]
        h = self.h[ix]
        return np.sqrt( e*V*np.abs(self.eta0)*h / (2*np.pi*self.p0_reference*self.beta_reference*c) )

    @property
    def phi_s(self):
        V = np.amax(self.V)

        if self.p_increment == 0 and V == 0:
            return 0

        deltaE  = self.p_increment*self.beta_reference*c
        phi_rel = np.arcsin(deltaE / (e*V))

        if self.eta0<0:
            # return np.sign(deltaE) * np.pi - phi_rel
            return np.pi - phi_rel
        else:
            return phi_rel

    def _phaselock(self):
        ix = np.argmax(self.V)
        h_fundamental = self.h[ix]

        for i in range(len(self.dphi)):
            if i == ix:
                continue
            self.dphi[i] -= self.h[i]/h_fundamental * self.phi_s

    @property
    def Hmax(self):
        return self.hamiltonian(self.zs, 0)

    # FIELDS AND POTENTIALS OF MULTIHARMONIC ACCELERATING BUCKET
    # ==========================================================
    def field(self, V, h, dphi):
        def v(z):
            return e*V/self.circumference * np.sin(h*z/self.R + dphi)
        return v

    def Ef(self, z):
        return reduce(lambda x, y: x + y,
                      [self.field(V, h, dphi)(z) for V, h, dphi in zip(self.V, self.h, self.dphi)])

    def E_acc(self, z):
        deltaE  = self.p_increment*self.beta_reference*c
        return self.Ef(z) - deltaE/self.circumference

    def potential(self, V, h, dphi):
        def v(z):
            return e*V/(2*np.pi*h) * np.cos(h*z/self.R + dphi)
        return v

    def Vf(self, z):
        return reduce(lambda x, y: x + y,
                      [self.potential(V, h, dphi)(z) for V, h, dphi in zip(self.V, self.h, self.dphi)])

    def V_acc(self, z):
        '''Sign makes sure we stay convex - just nicer'''
        z_extrema = self._get_zero_crossings(self.E_acc)
        deltaE  = self.p_increment*self.beta_reference*c

        if deltaE < 0:
            print '*** WARNING! Deceleration not gonna work. Please implement it correctly here in line ~355.'
            exit(-1)
        else:
            if np.sign(self.eta0) < 0:
                zc, zmax = z_extrema[-1], z_extrema[0]
            else:
                zmax, zc = z_extrema[-1], z_extrema[0]

        return -np.sign(self.eta0) * ((self.Vf(z) - self.Vf(zmax)) + (z - zmax) * deltaE/self.circumference)

    # ROOT AND BOUNDARY FINDING ROUTINES
    # ==================================
    def get_z_left_right(self, zc):
        z_cut = self._get_zero_crossings(lambda x: self.V_acc(x) - self.V_acc(zc))
        zleft, zright = z_cut[0], z_cut[-1]

        return zleft, zright

    def _get_zero_crossings(self, f, zedges=None):
        if zedges is None:
            zmin, zmax = self.zmin*1.01, self.zmax*1.01
        else:
            zmin, zmax = zedges

        zz = np.linspace(zmin, zmax, 1000)

        a = np.sign(f(zz))
        b = np.diff(a)
        ix = np.where(b)[0]
        s = []
        for i in ix:
            s.append(brentq(f, zz[i], zz[i + 1]))
        s = np.array(s)

        return s

    def _get_bucket_boundaries(self):
        '''
        Treat all crazy situations here
        '''
        z_extrema = self._get_zero_crossings(self.E_acc)
        z_cut = self._get_zero_crossings(self.V_acc)

        try:
            if np.sign(self.eta0) < 0:
                if len(z_extrema)==2:
                    self.zleft, self.zs, self.zright = z_extrema[0], z_extrema[-1], z_cut[0]
                elif len(z_extrema)==3:
                    self.zleft, self.zs, self.zright = z_extrema[0], z_extrema[1], z_cut[0]
                else:
                    raise ValueError("\n*** This length of z_extrema is not known how to be treated. Aborting.\n")
                self.zcut = z_cut[0]
            else:
                if len(z_extrema)==2:
                    self.zleft, self.zs, self.zright = z_cut[0], z_extrema[0], z_extrema[-1]
                elif len(z_extrema)==3:
                    self.zleft, self.zs, self.zright = z_cut[0], z_extrema[1], z_extrema[-1]
                else:
                    raise ValueError("\n*** This length of z_extrema is not known how to be treated. Aborting.\n")
                self.zcut = z_cut[0]
        except IndexError:
            self.zleft, self.zs, self.zright = z_extrema[0], z_extrema[1], z_extrema[-1]

    # HAMILTONIANS, SEPARATRICES AND RELATED FUNCTIONS
    # ================================================
    def hamiltonian(self, z, dp):
        '''Sign makes sure we stay convex - can then always use H<0'''
        return (-(np.sign(self.eta0) * 1/2 * self.eta0 *
                self.beta_reference * c * dp**2 * self.p0_reference
                + self.V_acc(z)) / self.p0_reference)

    def H0_from_sigma(self, z0):
        return np.abs(self.eta0)*self.beta_reference*c * (z0/self.beta_z)**2

    def H0_from_epsn(self, epsn):
        z0 = np.sqrt(epsn/(4*np.pi) * self.beta_z * e/self.p0_reference)
        return np.abs(self.eta0)*self.beta_reference*c * (z0/self.beta_z)**2

    def Hcut(self, zc):
        return self.hamiltonian(zc, 0)

    def equihamiltonian(self, zc):
        def s(z):
            r = np.sign(self.eta0) * 2/(self.eta0*self.beta_reference*c) * (-self.Hcut(zc) - self.V_acc(z)/self.p0_reference)
            return np.sqrt(r.clip(min=0))
        return s

    def separatrix(self, z):
        f = self.equihamiltonian(self.zright)
        return f(z)

    def p_max(self, zc):
        f = self.equihamiltonian(zc)
        return np.amax(f(self.zs))

    def is_in_separatrix(self, z, dp):
        """
        Returns boolean whether this coordinate is located
        strictly inside the separatrix.
        """
        return np.logical_and(np.logical_and(self.zleft < z, z < self.zright), self.hamiltonian(z, dp) > 0)

    def bucket_area(self):
        xmin, xmax = self.zleft, self.zright
        Q, error = dblquad(lambda y, x: 1, xmin, xmax, lambda x: 0, lambda x: self.separatrix(x))

        return Q * 2*self.p0_reference/e

    # DYNAMICAL LIST SETTERS
    # ======================
    def set_voltage_list(self, voltage_list):
        for i, V in enumerate(voltage_list):
            self.V[i] = V
        self._get_bucket_boundaries()

    def set_harmonic_list(self, harmonic_list):
        for i, h in enumerate(harmonic_list):
            self.h[i] = h
        self._get_bucket_boundaries()

    def set_phi_offset_list(self, phi_offset_list):
        for i, dphi in enumerate(phi_offset_list):
            self.dphi[i] = dphi
        self._get_bucket_boundaries()
