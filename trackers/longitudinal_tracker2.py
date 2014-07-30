'''
Just to test matching generator for now...
'''
from __future__ import division


import numpy as np
import pylab as plt
from scipy.optimize import brentq
from scipy.constants import c, e, m_p


class RFSystems(object):

    def __init__(self, circumference, gamma, alpha, delta_p, V, h, dphi):

        self.circumference = circumference
        self.gamma = gamma
        self.eta = alpha - 1/gamma**2
        self.delta_p = delta_p

        self.V = V
        self.h = h
        self.dphi = dphi

        zmax = self.circumference / (2*np.amin(h))
        self.zmin, self.zmax = -1.01*zmax, +1.01*zmax

        self._get_bucket_boundaries()

    @property
    def beta(self):
        return np.sqrt(1 - 1/self.gamma**2)

    @property
    def p0(self):
        return m_p * c * np.sqrt(self.gamma**2 - 1)

    @property
    def R(self):
        return self.circumference / (2 * np.pi)

    @property
    def beta_z(self):
        Qs = 0.01
        return np.abs(self.eta * self.R / Qs)

    @property
    def Hmax(self):
        return self.hamiltonian(self.zs, 0)

    def field(self, V, h, dphi):

        def v(z):
            phi = h*z/self.R
            return e*V/self.circumference * np.sin(phi + dphi)

        return v

    def Ef(self, z):

        return reduce(lambda x, y: x + y,
                      [self.field(V, h, dphi)(z) for V, h, dphi in zip(self.V, self.h, self.dphi)])

    def E_acc(self, z):

        return self.Ef(z) - e*self.delta_p/self.circumference

    def potential(self, V, h, dphi):

        def v(z):
            phi = h*z/self.R
            return e*V/(2*np.pi*h) * np.cos(phi + dphi)

        return v

    def Vf(self, z):

        return reduce(lambda x, y: x + y,
                      [self.potential(V, h, dphi)(z) for V, h, dphi in zip(self.V, self.h, self.dphi)])

    def V_acc(self, z):
        '''Sign makes sure we stay convex - just nicer'''
        z_extrema = self._get_zero_crossings(self.E_acc)

        if np.sign(self.eta) < 0:
            zc, zmax = z_extrema[-1], z_extrema[0]
        else:
            zmax, zc = z_extrema[-1], z_extrema[0]

        return -np.sign(self.eta) * ((self.Vf(z) - self.Vf(zmax)) + (z - zmax) * e*self.delta_p/self.circumference)

    def get_z_left_right(self, zc):
        # zz = np.linspace(self.zmin, self.zmax, 1000)
        # plt.figure(12)
        # plt.plot(zz, self.V_acc(zz)-self.V_acc(zc))
        # plt.axhline(0, c='r', lw=2)
        # plt.show()
        z_cut = self._get_zero_crossings(lambda x: self.V_acc(x) - self.V_acc(zc))
        zleft, zright = z_cut[0], z_cut[-1]

        return zleft, zright

    def Hcut(self, zc):
        return self.hamiltonian(zc, 0)

    def equihamiltonian(self, zc):
        def s(z):
            r = np.sign(self.eta) * 2/(self.eta*self.beta*c) * (-self.Hcut(zc) - self.V_acc(z)/self.p0)
            return np.sqrt(r.clip(min=0))
        return s

    def separatrix(self, z):
        f = self.equihamiltonian(self.zright)
        return f(z)

    def p_max(self, zc):
        f = self.equihamiltonian(zc)
        return np.amax(f(self.zs))

    def hamiltonian(self, z, dp):
        '''Sign makes sure we stay convex - can then always use H<0'''
        return -(np.sign(self.eta) * 1/2 * self.eta*self.beta*c * dp**2 * self.p0 + self.V_acc(z)) / self.p0
        # Hmax = np.amax(np.abs(1/2 * self.eta*self.beta*c * dp**2 + self.V_acc(z)/self.p0))
        # print Hmax
        # return -(np.sign(self.eta) * 1/2 * self.eta*self.beta*c * dp**2 + self.V_acc(z)/self.p0 + Hmax) * self.p0/e*self.circumference/c

    def H0(self, z0):
        return np.abs(self.eta)*self.beta*c * (z0 / self.beta_z) ** 2

    def _get_phi_s(self):

        V, self.accelerating_cavity = np.amax(self.V), np.argmax(self.V)
        if self.eta<0:
            return np.pi - np.arcsin(self.delta_p/V)
        elif self.eta>0:
            return np.arcsin(self.delta_p/V)
        else:
            return 0

    def _phaselock(self):
        phi_s = self._get_phi_s()
        cavities = range(len(self.V))
        del cavities[self.accelerating_cavity]

        for i in cavities:
            self.dphi[i] -= self.h[i]/self.h[self.accelerating_cavity] * self._get_phi_s()

        # print self.dphi

    def _get_zero_crossings(self, f):
        zz = np.linspace(self.zmin, self.zmax, 1000)

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
            if np.sign(self.eta) < 0:
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
