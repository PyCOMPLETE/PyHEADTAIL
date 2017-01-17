""".. copyright:: CERN"""
from __future__ import division

import numpy as np

from scipy.constants import c
from scipy.optimize import newton
from scipy.integrate import dblquad
from functools import partial, wraps

from ..cobra_functions.curve_tools import zero_crossings as cvt_zero_crossings
from ..general.decorators import deprecated
from . import Printing


def attach_clean_buckets(rf_parameter_changing_method, rfsystems_instance):
    '''Wrap an rf_parameter_changing_method (that changes relevant RF
    parameters, i.e. Kick attributes). Needs to be an instance method,
    presumably an RFSystems instance (hence the self argument in
    cleaned_rf_parameter_changing_method).
    In detail, attaches a call to the rfsystems_instance.clean_buckets
    method after calling the wrapped function.
    '''
    @wraps(rf_parameter_changing_method)
    def cleaned_rf_parameter_changing_method(self, *args, **kwargs):
        res = rf_parameter_changing_method(*args, **kwargs)
        rfsystems_instance.clean_buckets()
        return res
    return cleaned_rf_parameter_changing_method


class RFBucket(Printing):
    """Holds a blueprint of the current RF bucket configuration.
    Should be requested via RFSystems.get_bucket(gamma).

    Contains all information and all physical parameters of the
    current longitudinal RF configuration for a (real, not macro-)
    particle.

    Use for plotting or obtaining the Hamiltonian etc.

    Warning: zmin and zmax do not (yet) account for phi_offset of
    Kick objects, i.e. the left and right side of the bucket are not
    accordingly moved w.r.t. the harmonic phase offset.
    """

    """Sampling points to find zero crossings."""
    sampling_points = 1000

    def __init__(self, circumference, gamma, mass,
                 charge, alpha_array, p_increment,
                 harmonic_list, voltage_list, phi_offset_list,
                 z_offset=None, *args, **kwargs):
        '''Implements only the leading order momentum compaction factor.

        Arguments:
        - mass is the mass of the particle type in the beam
        - charge is the charge of the particle type in the beam
        - z_offset determines the centre for the bucket interval
        over which the root finding (of the electric force field to
        calibrate the separatrix Hamiltonian value to zero) is done.
        z_offset is per default determined by the zero crossing located
        closest to z == 0.
        '''

        self.charge = charge
        self.mass = mass

        self._gamma = gamma
        self._beta = np.sqrt(1 - gamma**-2)
        self._p0 = np.sqrt(gamma**2 - 1) * mass * c

        self.alpha0 = alpha_array[0]
        self.p_increment = p_increment

        self.circumference = circumference
        self.h = harmonic_list
        self.V = voltage_list
        self.dphi = phi_offset_list

        """Additional electric force fields to be added on top of the
        RF electric force field.
        """
        self._add_forces = []
        """Additional electric potential energies to be added on top
        of the RF electric potential energy.
        """
        self._add_potentials = []

        zmax = self.circumference / (2*np.amin(self.h))

        if z_offset is None:
            # i_fund = np.argmin(self.h) # index of fundamental RF element
            # phi_offset = self.dphi[i_fund]
            # # account for bucket size between -pi and pi.
            # # below transition there should be no relocation of the
            # # bucket interval by an offset of pi! we only need relative
            # # offset w.r.t. normal phi setting at given gamma (0 resp. pi)
            # if self.eta0 < 0:
            #     phi_offset -= np.pi
            # z_offset = -phi_offset * self.R / self.h[i_fund]
            ### the above approach does not take into account higher harmonics!
            ### Within a 2 bucket length interval we find all zero crossings
            ### of the non-accelerated total_force and identify the outermost
            ### separatrix UFPs via their minimal (convexified) potential value
            domain_to_find_bucket_centre = np.linspace(-1.999*zmax, 1.999*zmax,
                                                       self.sampling_points)
            z0 = self.zero_crossings(
                partial(self.total_force, acceleration=False),
                domain_to_find_bucket_centre)
            convex_pot0 = (
                np.array(self.total_potential(z0, acceleration=False)) *
                np.sign(self.eta0) / self.charge)  # charge for numerical reasons
            outer_separatrix_pot0 = np.min(convex_pot0)
            outer_separatrix_z0 = z0[np.isclose(convex_pot0,
                                                outer_separatrix_pot0)]
            # outer_separatrix_z0 should contain exactly 2 entries
            z_offset = np.mean(outer_separatrix_z0)
        self.z_offset = z_offset

        """Minimum and maximum z values on either side of the
        stationary bucket to cover the maximally possible bucket area,
        defined by the fundamental harmonic.
        (This range is always larger than the outmost unstable fix
        points of the real bucket including self.p_increment .)
        """
        self.interval = (z_offset - 1.01*zmax, z_offset + 1.01*zmax)

    @property
    def gamma(self):
        return self._gamma

    @property
    def beta(self):
        return self._beta

    @property
    def p0(self):
        return self._p0

    @property
    def deltaE(self):
        return self.p_increment * self.beta * c

    @property
    def harmonic_list(self):
        return self.h
    @harmonic_list.setter
    def harmonic_list(self, value):
        self.h = value

    @property
    def voltage_list(self):
        return self.V
    @voltage_list.setter
    def voltage_list(self, value):
        self.V = value

    @property
    def phi_offset_list(self):
        return self.dphi
    @phi_offset_list.setter
    def phi_offset_list(self, value):
        self.dphi = value

    @property
    def z_ufp(self):
        '''Return the (left-most) unstable fix point on the z axis
        within self.interval .
        '''
        try:
            return self._z_ufp
        except AttributeError:
            self._z_sfp, self._z_ufp = self._get_zsfp_and_zufp()
            return self._z_ufp

    @property
    def z_sfp(self):
        '''Return the (left-most) stable fix point on the z axis.
        within self.interval .
        '''
        try:
            return self._z_sfp
        except AttributeError:
            self._z_sfp, self._z_ufp = self._get_zsfp_and_zufp()
            return self._z_sfp

    @property
    def z_ufp_separatrix(self):
        '''Return the (left-most) unstable fix point at the outermost
        separatrix of the bucket.
        (i.e. a bucket boundary defining unstable fix point)
        '''
        if self.eta0 * self.p_increment > 0:
            # separatrix ufp right of sfp
            return self.z_ufp[-1]
        else:
            # separatrix ufp left of sfp
            return self.z_ufp[0]

    @property
    def z_sfp_extr(self):
        '''Return the (left-most) absolute extremal stable fix point
        within the bucket.
        '''
        sfp_extr_index = np.argmax(self.hamiltonian(self.z_sfp, 0,
                                                    make_convex=True))
        return self.z_sfp[sfp_extr_index]

    @property
    def z_left(self):
        '''Return the left bucket boundary within self.interval .'''
        try:
            return self._z_left
        except AttributeError:
            self._z_left, self._z_right, _ = self._get_bucket_boundaries()
            return self._z_left

    @property
    def z_right(self):
        '''Return the right bucket boundary within self.interval .'''
        try:
            return self._z_right
        except AttributeError:
            self._z_left, self._z_right, _ = self._get_bucket_boundaries()
            return self._z_right

    @property
    @deprecated("--> Will become z_left.\n")
    def zleft(self):
        '''Return the left bucket boundary within self.interval .'''
        try:
            return self._z_left
        except AttributeError:
            self._z_left, self._z_right, _ = self._get_bucket_boundaries()
            return self._z_left

    @property
    @deprecated("--> Will become z_right.\n")
    def zright(self):
        '''Return the right bucket boundary within self.interval .'''
        try:
            return self._z_right
        except AttributeError:
            self._z_left, self._z_right, _ = self._get_bucket_boundaries()
            return self._z_right

    @property
    def R(self):
        return self.circumference/(2*np.pi)

    # should make use of eta functionality of LongitudinalMap at some point
    @property
    def eta0(self):
        return self.alpha0 - self.gamma**-2

    @property
    def beta_z(self):
        return np.abs(self.eta0 * self.R / self.Q_s)

    @property
    @deprecated('--> Use Q_s instead!')
    def Qs(self):
        return self.Q_s

    @property
    def Q_s(self):
        """Linear synchrotron tune for small amplitudes i.e., in the
        center of the bucket. Analytical formula neglects any
        added forces / potentials via add_fields.
        """
        hV = sum([h * self.V[i] for i, h in enumerate(self.h)])
        # if hV == 0:
        #     ix = np.argmax(self.V)
        #     hV = self.h[ix] * self.V[ix]
        return np.sqrt(np.abs(self.charge)*np.abs(self.eta0)*hV /
                       (2*np.pi*self.p0*self.beta*c))

    def add_fields(self, add_forces, add_potentials):
        '''Include additional (e.g. non-RF) effects to this RFBucket.
        Use this interface for adding space charge influence etc.
        to the bucket parameters and shape.

        Arguments:
        - add_forces are additional electric force fields to be added
        on top of the RF electric force field.
        add_forces is expected to be an iterable of functions of z,
        in units of Coul*Volt/metre.
        - add_potentials are additional electric potential energies
        to be added on top of the RF electric potential energy.
        add_potentials is expected to be an iterable of functions of z,
        in units of Coul*Volt.

        Bucket shape parameters z_ufp, z_sfp, z_left and z_right are
        recalculated.
        '''
        self._add_forces += add_forces
        self._add_potentials += add_potentials
        try:
            delattr(self, "_z_ufp")
            delattr(self, "_z_sfp")
        except AttributeError:
            pass
        try:
            delattr(self, "_z_left")
            delattr(self, "_z_right")
        except AttributeError:
            pass

    # FORCE FIELDS AND POTENTIALS OF MULTI-HARMONIC ACCELERATING BUCKET
    # =================================================================
    def rf_force(self, V, h, dphi, p_increment, acceleration=True):
        def f(z):
            coefficient = np.abs(self.charge)/self.circumference
            focusing_field = reduce(lambda x, y: x+y, [
                V_i * np.sin(h_i*z/self.R + dphi_i)
                for V_i, h_i, dphi_i in zip(V, h, dphi)])
            if not acceleration:
                accelerating_field = 0
            else:
                accelerating_field = -(
                    p_increment*self.beta*c/self.circumference)
            return coefficient * focusing_field + accelerating_field
        return f

    def total_force(self, z, ignore_add_forces=False, acceleration=True):
        '''Return the total electric force field including
        - the acceleration offset and
        - the additional electric force fields (provided via
        self.add_nonRF_influences),
        evaluated at position z in units of Coul*Volt/metre.
        '''
        f = (self.rf_force(self.V, self.h, self.dphi,
                           self.p_increment, acceleration)(z) +
             sum(f(z) for f in self._add_forces
                 if not ignore_add_forces))
        return f


    @deprecated('--> Replace with "rf_force(acceleration=False)" ' +
                'as soon as possible.\n')
    def make_singleharmonic_force(self, V, h, dphi):
        '''Return the electric force field of a single harmonic
        RF element as a function of z in units of Coul*Volt/metre.
        '''
        def force(z):
            return (np.abs(self.charge) * V / self.circumference *
                    np.sin(h * z / self.R + dphi))
        return force

    @deprecated('--> Replace with "total_force(acceleration=False)" ' +
                'as soon as possible.\n')
    def make_total_force(self, ignore_add_forces=False):
        '''Return the stationary total electric force field of
        superimposed RF elements (multi-harmonics) as a function of z.
        Parameters are taken from RF parameters of this
        RFBucket instance.

        Adds the additional electric force fields (provided via
        self.add_nonRF_influences) on top.
        Uses units of Coul*Volt/metre.
        '''
        def total_force(z):
            '''Return stationary total electric force field of
            superimposed RF elements (multi-harmonics) and additional
            force fields as a function of z in units of Coul*Volt/metre.
            '''
            harmonics = (self.make_singleharmonic_force(V, h, dphi)(z)
                         for V, h, dphi in zip(self.V, self.h, self.dphi))
            return (sum(harmonics) + sum(f(z) for f in self._add_forces
                                         if not ignore_add_forces))
        return total_force

    @deprecated('--> Replace with "total_force" as soon as possible.\n')
    def acc_force(self, z, ignore_add_forces=False):
        '''Return the total electric force field including
        - the acceleration offset and
        - the additional electric force fields (provided via
        self.add_nonRF_influences),
        evaluated at position z in units of Coul*Volt/metre.
        '''
        total_force = self.make_total_force(
            ignore_add_forces=ignore_add_forces)
        return total_force(z) - self.deltaE / self.circumference


    def rf_potential(self, V, h, dphi, dp, acceleration=True):
        def vf(z):
            coefficient = np.abs(self.charge)/self.circumference
            focusing_potential = reduce(lambda x, y: x+y, [
                self.R/h[i] * V[i] * np.cos(h[i]*z/self.R + dphi[i])
                for i in xrange(len(V))])
            return coefficient * focusing_potential

        if not acceleration:
            return vf
        else:
            zmax = self.z_ufp_separatrix

            def f(z):
                return (vf(z) - vf(zmax) +
                        (dp*self.beta*c/self.circumference * (z - zmax)))
            return f

    def total_potential(self, z, ignore_add_potentials=False,
                        make_convex=False, acceleration=True):
        '''Return the total electric potential energy including
        - the linear acceleration slope and
        - the additional electric potential energies (provided via
        self.add_nonRF_influences),
        evaluated at position z in units of Coul*Volt.

        Note:
        Adds a potential energy offset: this relocates the extremum
        (defining the unstable fix point UFP of the bucket)
        to obtain zero potential energy at the UFP.
        Thus the Hamiltonian value of the separatrix is calibrated
        to zero.

        Arguments:
        - make_convex: multiplies by sign(eta) for plotting etc.
        To see a literal 'bucket structure' in the sense of a
        local minimum in the Hamiltonian topology, set make_convex=True
        in order to return sign(eta)*hamiltonian(z, dp).
        '''
        v = (self.rf_potential(self.V, self.h, self.dphi,
                               self.p_increment, acceleration)(z) +
             sum(pot(z) for pot in self._add_potentials
                 if not ignore_add_potentials))
        if make_convex:
            v *= np.sign(self.eta0)
        return v

    @deprecated('--> Replace with "rf_potential(acceleration=False)" ' +
                'as soon as possible.\n')
    def make_singleharmonic_potential(self, V, h, dphi):
        '''Return the electric potential energy of a single harmonic
        RF element as a function of z in units of Coul*Volt.
        '''
        def potential(z):
            return (np.abs(self.charge) * V / (2 * np.pi * h) *
                    np.cos(h * z / self.R + dphi))
        return potential

    @deprecated('--> Replace with ' +
                '"total_potential(acceleration=False)" ' +
                'as soon as possible.\n')
    def make_total_potential(self, ignore_add_potentials=False):
        '''Return the stationary total electric potential energy of
        superimposed RF elements (multi-harmonics) as a function of z.
        Parameters are taken from RF parameters of this
        RFBucket instance.

        Adds the additional electric potential energies
        (provided via self.add_nonRF_influences) on top.
        Uses units of Coul*Volt.
        '''
        def total_potential(z):
            '''Return stationary total electric potential energy of
            superimposed RF elements (multi-harmonics) and additional
            electric potentials as a function of z
            in units of Coul*Volt.
            '''
            harmonics = (self.make_singleharmonic_potential(V, h, dphi)(z)
                         for V, h, dphi in zip(self.V, self.h, self.dphi))
            return (sum(harmonics) + sum(pot(z) for pot in self._add_potentials
                                         if not ignore_add_potentials))
        return total_potential

    @deprecated('--> Replace with "total_potential as soon as possible.\n')
    def acc_potential(self, z, ignore_add_potentials=False,
                      make_convex=False):
        '''Return the total electric potential energy including
        - the linear acceleration slope and
        - the additional electric potential energies (provided via
        self.add_nonRF_influences),
        evaluated at position z in units of Coul*Volt.

        Note:
        Adds a potential energy offset: this relocates the extremum
        (defining the unstable fix point UFP of the bucket)
        to obtain zero potential energy at the UFP.
        Thus the Hamiltonian value of the separatrix is calibrated
        to zero.

        Arguments:
        - make_convex: multiplies by sign(eta) for plotting etc.
        To see a literal 'bucket structure' in the sense of a
        local minimum in the Hamiltonian topology, set make_convex=True
        in order to return sign(eta)*hamiltonian(z, dp).
        '''
        pot_tot = self.make_total_potential(
            ignore_add_potentials=ignore_add_potentials)
        z_boundary = self.z_ufp_separatrix
        v_acc = (pot_tot(z) - pot_tot(z_boundary) +
                 self.deltaE / self.circumference * (z - z_boundary))
        if make_convex:
            v_acc *= np.sign(self.eta0)
        return v_acc


    # ROOT AND BOUNDARY FINDING ROUTINES
    # ==================================
    def zero_crossings(self, f, x=None, subintervals=None):
        '''Determine roots of f along x.
        If x is not explicitely given, take stationary bucket interval.
        '''
        if x is None:
            if subintervals is None:
                subintervals = self.sampling_points
            x = np.linspace(*self.interval, num=subintervals)

        return cvt_zero_crossings(f, x)

    def _get_bucket_boundaries(self):
        '''Return the bucket boundaries as well as the whole list
        of acceleration voltage roots, (z_left, z_right, z_roots).
        '''
        z0 = np.atleast_1d(self.zero_crossings(self.total_potential))
        z0 = np.append(z0, self.z_ufp)
        return np.min(z0), np.max(z0), z0

    def _get_zsfp_and_zufp(self):
        '''Return (z_sfp, z_ufp),
        where z_sfp is the synchronous z on stable fix point,
        and z_ufp is the z of the (first) unstable fix point.

        Works for dominant harmonic situations which look like
        a single harmonic (which may be slightly perturbed), i.e.
        only one stable fix point and at most
        2 unstable fix points (stationary case).
        '''
        z0 = np.atleast_1d(self.zero_crossings(self.total_force))

        if not z0.size:
            # no bucket (i.e. bucket area 'negative')
            raise ValueError('With an electric force field this weak ' +
                             'there is no bucket for such strong ' +
                             'momentum increase -- ' +
                             'why do you ask me for bucket boundaries ' +
                             'in this hyperbolic phase space structure?!')

        z0odd = z0[::2]
        z0even = z0[1::2]

        if len(z0) == 1:  # exactly zero bucket area
            return z0, z0

        if self.eta0 * self.p_increment > 0:
            # separatrix ufp right of sfp
            z_sfp, z_ufp = z0odd, z0even
        else:
            # separatrix ufp left of sfp
            z_sfp, z_ufp = z0even, z0odd

        return z_sfp, z_ufp

    # HAMILTONIANS, SEPARATRICES AND RELATED FUNCTIONS
    # ================================================
    def hamiltonian(self, z, dp, make_convex=False):
        '''Return the Hamiltonian at position z and dp in units of
        Coul*Volt/p0.

        Arguments:
        - make_convex: multiplies by sign(eta) for plotting etc.
        To see a literal 'bucket structure' in the sense of a
        local minimum in the Hamiltonian topology, set make_convex=True
        in order to return sign(eta)*hamiltonian(z, dp).
        '''
        h = (-0.5 * self.eta0 * self.beta * c * dp**2 +
             self.total_potential(z) / self.p0)
        if make_convex:
            h *= np.sign(self.eta0)
        return h

    def equihamiltonian(self, zcut):
        '''Return a function dp_at that encodes the equi-Hamiltonian
        contour line that cuts the z axis at (zcut, 0).
        In more detail, dp_at(z) returns the (positive) dp value at
        its given z argument such that
        self.hamiltonian(z, dp_at(z)) == self.hamiltonian(zcut, 0) .
        '''
        def dp_at(z):
            hcut = self.hamiltonian(zcut, 0)
            r = np.abs(2./(self.eta0*self.beta*c) *
                       (self.total_potential(z)/self.p0 - hcut))
            return np.sqrt(r.clip(min=0))
        return dp_at

    def separatrix(self, z):
        '''Return the positive dp value corresponding to the separatrix
        Hamiltonian contour line at the given z.
        '''
        dp_separatrix_at = self.equihamiltonian(self.z_ufp_separatrix)
        return dp_separatrix_at(z)

    def h_sfp(self, make_convex=False):
        '''Return the extremal Hamiltonian value at the corresponding
        stable fix point (self.z_sfp_extr, 0) of the bucket.
        '''
        return self.hamiltonian(self.z_sfp_extr, 0, make_convex)

    def dp_max(self, zcut):
        '''Return the maximal dp value along the equihamiltonian which
        is located at (one of the) self.z_sfp .
        '''
        dp_at = self.equihamiltonian(zcut)
        return np.amax(dp_at(self.z_sfp))

    def is_in_separatrix(self, z, dp, margin=0):
        """Return boolean whether the coordinate (z, dp) is located
        strictly inside the separatrix of this bucket
        (i.e. excluding neighbouring buckets).

        If margin is different from 0, use the equihamiltonian
        defined by margin*self.h_sfp instead of the separatrix.
        (Use margin as a weighting factor in units of the Hamiltonian
        value at the stable fix point to move from the separatrix
        toward the extremal Hamiltonian value at self.z_sfp .)
        """
        within_interval = np.logical_and(self.z_left < z, z < self.z_right)
        within_separatrix = (self.hamiltonian(z, dp, make_convex=True) >
                             margin * self.h_sfp(make_convex=True))
        return np.logical_and(within_interval, within_separatrix)

    def make_is_accepted(self, margin=0):
        """Return the function is_accepted(z, dp) definining the
        equihamiltonian with a value of margin*self.h_sfp .
        For margin 0, the returned is_accepted(z, dp) function is
        identical to self.is_in_separatrix(z, dp).
        """
        return partial(self.is_in_separatrix, margin=margin)

    def emittance_single_particle(self, z=None, sigma=2):
        """The single particle emittance computed along a given equihamiltonian line

        """
        if z is not None:
            zl = -sigma * z
            zr = +sigma * z
            f = self.equihamiltonian(sigma * z)
        else:
            zl = self.z_left
            zr = self.z_right
            f = self.separatrix

        Q, error = dblquad(lambda y, x: 1, zl, zr,
                           lambda x: 0, f)

        return Q * 2*self.p0/np.abs(self.charge)

    def bunchlength_single_particle(self, epsn_z, verbose=False):
        """The corresponding rms bunch length computed form the single particle
        emittance

        """
        def emittance_from_zcut(zcut):
            emittance = self.emittance_single_particle(zcut)
            if np.isnan(emittance):
                raise ValueError

            if verbose:
                self.prints('... distance to target emittance: ' +
                            '{:.4e}'.format(emittance-epsn_z))
            return emittance - epsn_z

        sigma = newton(emittance_from_zcut, 1)

        return sigma

    def guess_H0(self, var, from_variable='epsn', make_convex=True):
        """Pure estimate value of H_0 starting from a bi-Gaussian bunch
        in a linear "RF bucket". Intended for use by iterative matching
        algorithms in the generators module.
        """
        # If Qs = 0, get the fundamental harmonic
        hV = sum([h * V for h, V in zip(self.h, self.V)])
        if hV == 0:
            ix = np.argmax(self.V)
            hV = self.h[ix] * self.V[ix]
        Qs = np.sqrt(np.abs(self.charge)*np.abs(self.eta0)*hV /
                     (2*np.pi*self.p0*self.beta*c))
        beta_z = np.abs(self.eta0 * self.R / Qs)

        # to be replaced with something more flexible (add_forces etc.)
        if from_variable == 'epsn':
            epsn = var
            z0 = np.sqrt(epsn/(4.*np.pi) * beta_z *
                         np.abs(self.charge)/self.p0)  # gauss approx.
        elif from_variable == 'sigma':
            z0 = var

        h0 = self.beta*c * (z0/beta_z)**2
        if make_convex:
            h0 *= np.abs(self.eta0)
        return h0
