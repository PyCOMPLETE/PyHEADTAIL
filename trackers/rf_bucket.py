from __future__ import division

import numpy as np

from scipy.optimize import brentq
from scipy.constants import c, e, m_p
from scipy.integrate import dblquad

from . import Printing

from functools import wraps

def attach_clean_buckets(rf_parameter_changing_method, rfsystems_instance):
    '''Wrap an rf_parameter_changing_method (that changes relevant RF
    parameters, i.e. Kick attributes). Needs to be an instance method,
    presumably an RFSystems instance.
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
    current longitudinal RF configuration.

    Use for plotting or obtaining the Hamiltonian etc.
    """
    def __init__(self, circumference, gamma_reference, alpha_array,
                 p_increment, harmonic_list, voltage_list,
                 phi_offset_list, *args, **kwargs):
        '''Implements only the leading order momentum compaction factor.
        '''

        self.circumference = circumference
        self.mass = m_p
        self.gamma_reference = gamma_reference

        self.alpha0 = alpha_array[0]
        self.p_increment = p_increment

        self.h = harmonic_list
        self.V = voltage_list
        self.dphi = phi_offset_list

        zmax = self.circumference / (2*np.amin(harmonic_list))
        self.zmin, self.zmax = -1.01*zmax, +1.01*zmax
        self._get_bucket_boundaries()

    @property
    def gamma_reference(self):
        return self._gamma_reference
    @gamma_reference.setter
    def gamma_reference(self, value):
        self._gamma_reference = value
        self._beta_reference= np.sqrt(1 - self.gamma_reference**-2)
        # self._betagamma_reference = np.sqrt(self.gamma_reference**2 - 1)
        self._p0_reference = (self.beta_reference * self.gamma_reference
                              * self.mass * c)

    @property
    def beta_reference(self):
        return self._beta_reference
    @beta_reference.setter
    def beta_reference(self, value):
        self._gamma_reference = (1. / np.sqrt(1 - value**2))

    @property
    def p0_reference(self):
        return self._p0_reference
    @p0_reference.setter
    def p0_reference(self, value):
        self._gamma_reference = (value / (self.mass * self.beta_reference * c))

    @property
    def R(self):
        return self.circumference/(2*np.pi)

    @property
    def eta0(self):
        return self.alpha0 - self.gamma_reference**-2

    @property
    def beta_z(self):
        return np.abs(self.eta0 * self.R / self.Qs)

    @property
    def Qs(self):
        ix = np.argmax(self.V)
        V = self.V[ix]
        h = self.h[ix]
        return np.sqrt( e*V*np.abs(self.eta0)*h /
                       (2*np.pi*self.p0_reference*self.beta_reference*c) )

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
                      [self.field(V, h, dphi)(z) for V, h, dphi
                       in zip(self.V, self.h, self.dphi)])

    def E_acc(self, z):
        deltaE  = self.p_increment*self.beta_reference*c
        return self.Ef(z) - deltaE/self.circumference

    def potential(self, V, h, dphi):
        def v(z):
            return e*V/(2*np.pi*h) * np.cos(h*z/self.R + dphi)
        return v

    def Vf(self, z):
        return reduce(lambda x, y: x + y,
                      [self.potential(V, h, dphi)(z) for V, h, dphi
                       in zip(self.V, self.h, self.dphi)])

    def V_acc(self, z):
        '''Sign makes sure we stay convex - just nicer'''
        z_extrema = self._get_zero_crossings(self.E_acc)
        deltaE = self.p_increment*self.beta_reference*c

        if deltaE < 0:
            self.warns('Deceleration is not implemented properly!')
            exit(-1)
        else:
            if np.sign(self.eta0) < 0:
                zc, zmax = z_extrema[-1], z_extrema[0]
            else:
                zmax, zc = z_extrema[-1], z_extrema[0]

        return -np.sign(self.eta0) * (
                (self.Vf(z) - self.Vf(zmax)) +
                (z - zmax) * deltaE/self.circumference
            )

    # ROOT AND BOUNDARY FINDING ROUTINES
    # ==================================
    def get_z_left_right(self, zc):
        z_cut = self._get_zero_crossings(
            lambda x: self.V_acc(x) - self.V_acc(zc))
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
        return (-(np.sign(self.eta0) * 0.5 * self.eta0 *
                self.beta_reference * c * dp**2 * self.p0_reference
                + self.V_acc(z)) / self.p0_reference)

    def H0_from_sigma(self, z0):
        return np.abs(self.eta0)*self.beta_reference*c * (z0/self.beta_z)**2

    def H0_from_epsn(self, epsn):
        z0 = np.sqrt(epsn/(4.*np.pi) * self.beta_z * e/self.p0_reference)
        return np.abs(self.eta0)*self.beta_reference*c * (z0/self.beta_z)**2

    def Hcut(self, zc):
        return self.hamiltonian(zc, 0)

    def equihamiltonian(self, zc):
        def s(z):
            r = (np.sign(self.eta0) * 2./(self.eta0*self.beta_reference*c)
                 * (-self.Hcut(zc) - self.V_acc(z)/self.p0_reference))
            return np.sqrt(r.clip(min=0))
        return s

    def separatrix(self, z):
        f = self.equihamiltonian(self.zright)
        return f(z)

    def p_max(self, zc):
        f = self.equihamiltonian(zc)
        return np.amax(f(self.zs))

    def is_in_separatrix(self, z, dp):
        """Return boolean whether the coordinate (z, dp) is located
        strictly inside the separatrix.
        """
        return np.logical_and(np.logical_and(self.zleft < z, z < self.zright),
                              self.hamiltonian(z, dp) > 0)

    def make_is_accepted(self, margin):
        """Return the function is_accepted(z, dp) definining the
        equihamiltonian with a value of margin*self.Hmax . For margin 0,
        the returned is_accepted(z, dp) function is equivalent to
        self.is_in_separatrix(z, dp).
        """

        def is_accepted(z, dp):
            """ Returns boolean whether the coordinate (z, dp) is
            located inside the equihamiltonian defined by
            margin*self.Hmax . """
            return np.logical_and(
                np.logical_and(self.zleft < z, z < self.zright),
                self.hamiltonian(z, dp) > margin * self.Hmax)
        return is_accepted

    def bucket_area(self):
        xmin, xmax = self.zleft, self.zright
        Q, error = dblquad(lambda y, x: 1, xmin, xmax, lambda x: 0, lambda x: self.separatrix(x))

        return Q * 2*self.p0_reference/e
