from __future__ import division
'''
@file matching
@author Kevin Li
@date February 2014
@brief Module for matching transverse and longitudinal distributions
@copyright CERN
'''


import numpy as np


from scipy.integrate import quad, dblquad
from scipy.optimize import brentq
from scipy.constants import c, e, m_p


import pylab as plt


class StationaryExponential(object):

    def __init__(self, H):
        self.H = H
        self.H0 = 1

    def function(self, phi, dp):
        return (np.exp(self.H(phi, dp) / self.H0) - 1) / (np.exp(self.H(0, 0) / self.H0) - 1)

    # @property
    # def H0(self):
    #     return self.H0
    # @H0.setter
    # def H0(self, value):
    #     self.H0 = value


# class Particles(object): pass


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
        return self.eta * self.R / self.Qs

    def field(self, V, h, dphi):
        def v(z):
            return e*V/self.circumference * np.sin(h*z/self.R + dphi)
        return v

    def Ef(self, z):
        return self.field(self.V[0], self.h[0], self.dphi[0])(z) + self.field(self.V[1], self.h[1], self.dphi[1])(z)
        # return sum([self.field(V, h, dphi)(z) for V, h, dphi in zip(self.V, self.h, self.dphi)])

    def E_acc(self, z):
        return self.Ef(z) - e*self.delta_p/self.circumference

    def potential(self, V, h, dphi):
        def v(z):
            return e*V/(2*np.pi*h) * np.cos(h*z/self.R + dphi)
        return v

    def Vf(self, z):
        return self.potential(self.V[0], self.h[0], self.dphi[0])(z) + self.potential(self.V[1], self.h[1], self.dphi[1])(z)
        # return sum([self.potential(V, h, dphi)(z) for V, h, dphi in zip(self.V, self.h, self.dphi)])

    def V_acc(self, z):
        '''Sign makes sure we stay convex - just nicer'''
        s = self.get_zmax()
        # t = self.get_pmax()

        if np.sign(self.eta) < 0:
            zmax = s[0]
        else:
            zmax = s[-1]

        return -np.sign(self.eta) * ((self.Vf(z) - self.Vf(zmax)) + (z - zmax) * e*self.delta_p/self.circumference)

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

    def get_zmax(self):
        zz = np.linspace(-self.zmax, self.zmax, 200)

        a = np.sign(self.E_acc(zz))
        b = np.diff(a)
        ix = np.where(b)[0]
        s = []
        for i in ix:
            s.append(brentq(self.E_acc, zz[i], zz[i + 1]))
        s = np.array(s)

        self.s = s
        return [s[0], s[-1]]

    def get_pmax(self):
        zz = linspace(-self.zmax, self.zmax, 200)

        a = np.sign(self.V_acc(zz))
        b = np.diff(a)
        ix = np.where(b)[0]
        s = []
        for i in ix:
            s.append(brentq(self.V_acc, zz[i], zz[i + 1]))
        s = np.array(s)

        self.t = s
        return [s[0], s[-1]]

    def p_separatrix(self, z):

        return np.sqrt(2/(self.eta*self.beta*c*self.p0) * self.V_acc(z))

    # def z(self, phi):

    #     return phi * self.R / self.h

    def hamiltonian(self, z, dp):
        '''Sign makes sure we stay convex - can then always use H<0'''
        return -(np.sign(self.eta) * 1/2 * self.eta*self.beta*c * dp**2 * self.p0 + self.V_acc(z)) / self.p0




    # def potential(self, phi):

    #     V1, h1 = 2e6, 4620
    #     V2, h2 = -0. * V1, 2 * h1
    #     V3, h3 = -0. * V1, 4 * h1

    #     return (e * V1 / (2 * np.pi * self.p0 * h1) * np.cos(phi)
    #           + e * V2 / (2 * np.pi * self.p0 * h2) * np.cos(h2 / h1 * phi)
    #           + e * V3 / (2 * np.pi * self.p0 * h3) * np.cos(h3 / h1 * phi))

    # def hamiltonian(self, phi, dp):

    #     return -1 / 2 * self.eta * self.beta * c * dp ** 2 + self.potential(phi) - self.potential(np.pi) + phi * self.dp_absolute

    # def p_sep(self, phi):

    #     return np.sqrt(2 / (self.eta * self.beta * c) * (self.potential(phi) - self.potential(np.pi) + phi * self.dp_absolute))


class PhaseSpace(object):
    '''
    Class for general matching of bunch particle distribution to local machine
    optics. Since the standard matching is taking place within the beam class
    itself, here, there are only the more complex matching functions to any RF
    systems configuration. Tha class takes as argument a beam and an RF system.
    '''

    def __init__(self, bunch, rfsystem):

        self.rf = rfsystem

    def _std_targeted(self, psi, sigma):
        print 'Iterative evaluation of bunch length...'

        counter = 0
        eps = 1

        # Test for maximum bunch length
        z0 = self.rf.circumference
        H0 = self.rf.eta * self.rf.beta * c * (z0 / self.rf.beta_z) ** 2
        psi.H0 = H0
        s = self._std_computed(psi.function, self.rf.p_sep, -np.pi+0.01, np.pi+0.01)
        zS = self.rf.z(np.sqrt(s))
        print "\n--> Maximum rms bunch length in bucket:", zS, " m.\n"
        if sigma > zS * 0.95:
            print "\n*** WARNING! Bunch appears to be too long for bucket!\n"

        # Initial values
        z0 = sigma
        H0 = self.rf.eta * self.rf.beta * c * (z0 / self.rf.beta_z) ** 2
        psi.H0 = H0

        # Iteratively obtain true H0 to make target sigma
        zH = z0
        while abs(eps)>1e-6:
            s = self._std_computed(psi.function, self.rf.p_sep, -np.pi+0.01, np.pi+0.01)

            zS = self.rf.z(np.sqrt(s))
            eps = zS - z0
            print counter, zH, zS, eps
            zH -= 0.5 * eps
            H0 = self.rf.eta * self.rf.beta * c * (zH / self.rf.beta_z) ** 2
            psi.H0 = H0

            counter += 1
            if counter > 100:
                print "\n*** WARNING: too many interation steps! There are several possible reasons for that:"
                print "1. Is the Hamiltonian correct?"
                print "2. Is the stationary distribution function convex around zero?"
                print "3. Is the bunch too long to fit into the bucket?"
                print "4. Is this algorithm not qualified?"
                print "Aborting..."
                sys.exit(-1)

        return psi.function

    def _std_computed(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin to xmax
        along the contours p_sep using numerical integration methods.
        '''
        xx = np.linspace(xmin, xmax, 1000)
        plt.plot(xx, p_sep(xx))
        plt.plot(xx, -p_sep(xx))
        plt.show()
        # PP = psi(XX, YY)
        # VV = var(XX, YY)

        # psi_max = np.amax(PP)
        # var_max = np.amax(VV)

        Q, error = dblquad(lambda y, x: psi(x, y), xmin, xmax,
                    lambda x: 0, lambda x: p_sep(x))
        V, error = dblquad(lambda y, x: x ** 2 * psi(x, y), xmin, xmax,
                    lambda x: 0, lambda x: p_sep(x))

        return V / Q

    def generate(self, n_particles, sigma):
        '''
        Generate a 2d phase space of n_particles particles randomly distributed
        according to the particle distribution function psi within the region
        [xmin, xmax, ymin, ymax].
        '''
        x = np.zeros(n_particles)
        y = np.zeros(n_particles)

        i, j = 0, 0
        xmin, xmax = -np.pi, np.pi
        ymin, ymax = -3e-3, 3e-3

        dx = xmax - xmin
        dy = ymax - ymin

        psi = self._std_targeted(StationaryExponential(self.rf.hamiltonian), sigma)

        while j < n_particles:
            u = xmin + dx * np.random.random()
            v = ymin + dy * np.random.random()
            s = np.random.random()
            i += 1
            if s < psi(u, v):
                x[j] = u
                y[j] = v
                j += 1

        return x, y, j / i * dx * dy, psi

# def match_longitudinal(length, bucket, matching=None):

#     if not matching and isinstance(bucket, float):
#         def match(bunch):
#             match_none(bunch, length, bucket)
#     elif matching == 'simple':
#         def match(bunch):
#             try:
#                 match_simple(bunch, length, bucket)
#             except AttributeError:
#                 raise TypeError("Bad bucket instance for matching!")
#     elif matching == 'full':
#         def match(bunch):
#             try:
#                 match_full(bunch, length, bucket)
#             except AttributeError:
#                 raise TypeError("Bad bucket instance for matching!")
#     else:
#         raise ValueError("Unknown matching " + str(matching) + " for bucket " + str(bucket))

#     return match

# # TODO: improve implementation of match_simple
# def match_simple(bunch, sigma_dz, bucket):
#     R = bucket.circumference / (2 * np.pi)
#     eta = bucket.eta(bunch)
#     Qs = bucket.Qs(bunch)

#     # Assuming a gaussian-type stationary distribution
#     sigma_dp = sigma_dz * Qs / eta / R
#     unmatched_inbucket(bunch, sigma_dz, sigma_dp, bucket)

# def unmatched_inbucket(bunch, sigma_dz, sigma_dp, bucket):
#     for i in xrange(bunch.n_macroparticles):
#         if not bucket.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
#             while not bucket.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
#                 bunch.dz[i] = sigma_dz * np.random.randn()
#                 bunch.dp[i] = sigma_dp * np.random.randn()

# def match_longitudinal(bunch):

#     R = bucket.circumference / (2 * np.pi)
#     eta = bucket.eta(bunch)
#     Qs = bucket.Qs(bunch)

#     # Assuming a gaussian-type stationary distribution
#     sigma_dp = sigma_dz * Qs / eta / R

def cut_along_separatrix(bunch, sigma_z, sigma_dp, cavity):

    for i in xrange(bunch.n_macroparticles):
        if not cavity.is_in_separatrix(bunch.z[i], bunch.dp[i], bunch):
            while not cavity.is_in_separatrix(bunch.z[i], bunch.dp[i], bunch):
                bunch.z[i] = sigma_z * np.random.randn()
                bunch.dp[i] = sigma_dp * np.random.randn()

def match_to_bucket(bunch, length, cavity):

    R = cavity.circumference / (2 * np.pi)
    eta = cavity.eta(bunch)
    Qs = cavity.Qs(bunch)

    zmax = np.pi * R / cavity.h
    pmax = 2 * Qs / eta / cavity.h
    Hmax1 = cavity.hamiltonian(zmax, 0, bunch)
    Hmax2 = cavity.hamiltonian(0, pmax, bunch)
    # assert(Hmax1 == Hmax2)
    Hmax = Hmax1
    epsn_z = np.pi / 2 * zmax * pmax * bunch.p0 / e
    print '\nStatistical parameters from RF cavity:'
    print 'zmax:', zmax, 'pmax:', pmax, 'epsn_z:', epsn_z

    # Assuming a gaussian-type stationary distribution
    sigma_dz = length # np.std(bunch.dz)
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * bunch.beta * c * sigma_dp ** 2
    epsn_z = 4 * np.pi * Qs / eta / R * sigma_dz ** 2 * bunch.p0 / e
    print '\nStatistical parameters from initialisation:'
    print 'sigma_dz:', sigma_dz, 'sigma_dp:', sigma_dp, 'epsn_z:', epsn_z

    print '\n--> Bunchlength:'
    sigma_dz = bunchlength(bunch, cavity, sigma_dz)
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * bunch.beta * c * sigma_dp ** 2

    psi = stationary_exponential(cavity.hamiltonian, Hmax, H0, bunch)

    zl = np.linspace(-zmax, zmax, 1000) * 1.5
    pl = np.linspace(-pmax, pmax, 1000) * 1.5
    zz, pp = np.meshgrid(zl, pl)
    HH = psi(zz, pp)
    HHmax = np.amax(HH)

    for i in xrange(bunch.n_macroparticles):
        while True:
            s = (np.random.rand() - 0.5) * 2 * zmax
            t = (np.random.rand() - 0.5) * 2 * pmax
            u = (np.random.rand()) * HHmax * 1.01
            C = psi(s, t)
            if u < C:
                break
        bunch.dz[i] = s
        bunch.dp[i] = t

    epsz = np.sqrt(np.mean(bunch.dz * bunch.dz) * np.mean(bunch.dp * bunch.dp)
                 - np.mean(bunch.dz * bunch.dp) * np.mean(bunch.dz * bunch.dp))
    print '\nStatistical parameters from distribution:'
    print 'sigma_dz:', np.std(bunch.dz), 'sigma_dp:', np.std(bunch.dp), \
          'epsn_z:', 4*np.pi*np.std(bunch.dz)*np.std(bunch.dp)*bunch.p0/e, 4*np.pi*epsz*bunch.p0/e

def bunchlength(bunch, cavity, sigma_dz):

    print 'Iterative evaluation of bunch length...'

    counter = 0
    eps = 1

    R = cavity.circumference / (2 * np.pi)
    eta = cavity.eta(bunch)
    Qs = cavity.Qs(bunch)

    zmax = np.pi * R / cavity.h
    Hmax = cavity.hamiltonian(zmax, 0, bunch)

    # Initial values
    z0 = sigma_dz
    p0 = z0 * Qs / eta / R            #Matching condition
    H0 = eta * bunch.beta * c * p0 ** 2

    z1 = z0
    while abs(eps)>1e-6:
        # cf1 = 2 * Qs ** 2 / (eta * h) ** 2
        # dplim = lambda dz: np.sqrt(cf1 * (1 + np.cos(h / R * dz) + (h / R * dz - np.pi) * np.sin(cavity.phi_s)))
        # dplim = lambda dz: np.sqrt(2) * Qs / (eta * h) * np.sqrt(np.cos(h / R * dz) - np.cos(h / R * zmax))
        # Stationary distribution
        # psi = lambda dz, dp: np.exp(cavity.hamiltonian(dz, dp, bunch) / H0) - np.exp(Hmax / H0)

        # zs = zmax / 2.

        psi = stationary_exponential(cavity.hamiltonian, Hmax, H0, bunch)
        dplim = cavity.separatrix.__get__(cavity)
        N = dblquad(lambda dp, dz: psi(dz, dp), -zmax, zmax,
                    lambda dz: -dplim(dz, bunch), lambda dz: dplim(dz, bunch))
        I = dblquad(lambda dp, dz: dz ** 2 * psi(dz, dp), -zmax, zmax,
                    lambda dz: -dplim(dz, bunch), lambda dz: dplim(dz, bunch))

        # Second moment
        z2 = np.sqrt(I[0] / N[0])
        eps = z2 - z0

        # print z1, z2, eps
        z1 -= eps

        p0 = z1 * Qs / eta / R
        H0 = eta * bunch.beta * c * p0 ** 2

        counter += 1
        if counter > 100:
            print "\n*** WARNING: too many interation steps! There are several possible reasons for that:"
            print "1. Is the Hamiltonian correct?"
            print "2. Is the stationary distribution function convex around zero?"
            print "3. Is the bunch too long to fit into the bucket?"
            print "4. Is this algorithm not qualified?"
            print "Aborting..."
            sys.exit(-1)

    return z1
