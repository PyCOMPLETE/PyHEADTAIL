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

        self.zmax = self.circumference / (2*np.amin(h))
        self.zmax += 0.01*self.zmax

        self.Qs = 0.017

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
        return np.abs(self.eta * self.R / self.Qs)

    def field(self, V, h, dphi):
        def v(z):
            phi = h*z/self.R
            return e*V/self.circumference * np.sin(phi + dphi)
        return v

    def Ef(self, z):
        # return self.field(self.V[0], self.h[0], self.dphi[0])(z) + self.field(self.V[1], self.h[1], self.dphi[1])(z)
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
        # return self.potential(self.V[0], self.h[0], self.dphi[0])(z) + self.potential(self.V[1], self.h[1], self.dphi[1])(z)
        return reduce(lambda x, y: x + y,
                      [self.potential(V, h, dphi)(z) for V, h, dphi in zip(self.V, self.h, self.dphi)])

    def V_acc(self, z):
        '''Sign makes sure we stay convex - just nicer'''
        ze = self._get_zero_crossings(self.E_acc)

        if np.sign(self.eta) < 0:
            zmax = ze[0]
        else:
            zmax = ze[-1]

        return -np.sign(self.eta) * ((self.Vf(z) - self.Vf(zmax)) + (z - zmax) * e*self.delta_p/self.circumference)
        # return -np.sign(self.eta) * ((self.Vf(z) - 0*self.Vf(zmax)) + (z - 0*zmax) * e*self.delta_p/self.circumference)

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
        zz = np.linspace(-self.zmax, self.zmax, 200)

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
        self.z_extrema = self._get_zero_crossings(self.E_acc)
        self.z_zeros = self._get_zero_crossings(self.V_acc)
        self.p_sep = np.amax(self.separatrix(self.z_extrema))

        try:
            if np.sign(self.eta) < 0:
                self.z_sep = [self.z_extrema[0], self.z_zeros[0]]
            else:
                self.z_sep = [self.z_zeros[0], self.z_extrema[-1]]
        except IndexError:
            self.z_sep = [self.z_extrema[0], self.z_extrema[-1]]

    def separatrix(self, z):
        r = -np.sign(self.eta)*2 / (self.eta*self.beta*c*self.p0) * self.V_acc(z)
        return np.sqrt(r.clip(min=0))

    def hamiltonian(self, z, dp):
        '''Sign makes sure we stay convex - can then always use H<0'''
        return -(np.sign(self.eta) * 1/2 * self.eta*self.beta*c * dp**2 * self.p0 + self.V_acc(z)) / self.p0
        # Hmax = np.amax(np.abs(1/2 * self.eta*self.beta*c * dp**2 + self.V_acc(z)/self.p0))
        # print Hmax
        # return -(np.sign(self.eta) * 1/2 * self.eta*self.beta*c * dp**2 + self.V_acc(z)/self.p0 + Hmax) * self.p0/e*self.circumference/c
    def H0(self, z0):
        return np.sign(self.eta) * self.eta * self.beta * c * (z0 / self.beta_z) ** 2
