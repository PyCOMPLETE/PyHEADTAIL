from __future__ import division
'''
@file matching
@author Kevin Li, Adrian Oeftiger
@date February 2014
@brief Module for matching transverse and longitudinal distributions
@copyright CERN
'''


import numpy as np
from numpy.random import RandomState

from abc import ABCMeta, abstractmethod

from scipy.constants import c, e
from scipy.integrate import quad, dblquad


class PhaseSpace(object):
    """Knows how to distribute particle coordinates for a beam
    according to certain distribution functions.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate(self, beam):
        """Creates the beam macroparticles according to a
        distribution function (depends on the implementing class).
        """
        pass


class GaussianX(PhaseSpace):
    """Horizontal Gaussian particle phase space distribution."""

    def __init__(self, n_macroparticles, sigma_x, sigma_xp, generator_seed=None):
        """Initiates the horizontal beam coordinates
        to the given Gaussian shape.
        """
        self.n_macroparticles = n_macroparticles
        self.sigma_x  = sigma_x
        self.sigma_xp = sigma_xp

        self.random_state = RandomState()
        self.random_state.seed(generator_seed)

    @classmethod
    def from_optics(cls, n_macroparticles, alpha_x, beta_x, epsn_x, betagamma, generator_seed=None):
        """Initialise GaussianX from the given optics functions.
        beta_x is given in meters and epsn_x in micrometers.
        """
        sigma_x  = np.sqrt(beta_x * epsn_x * 1e-6 / betagamma)
        sigma_xp = sigma_x / beta_x
        return cls(n_macroparticles, sigma_x, sigma_xp, generator_seed)

    def generate(self, beam):
        beam.x = self.sigma_x * self.random_state.randn(self.n_macroparticles)
        beam.xp = self.sigma_xp * self.random_state.randn(self.n_macroparticles)


class GaussianY(PhaseSpace):
    """Vertical Gaussian particle phase space distribution."""

    def __init__(self, n_macroparticles, sigma_y, sigma_yp, generator_seed=None):
        """Initiates the vertical beam coordinates
        to the given Gaussian shape.
        """
        self.n_macroparticles = n_macroparticles
        self.sigma_y  = sigma_y
        self.sigma_yp = sigma_yp

        self.random_state = RandomState()
        self.random_state.seed(generator_seed)
        
    @classmethod
    def from_optics(cls, n_macroparticles, alpha_y, beta_y, epsn_y, betagamma, generator_seed=None):
        """Initialise GaussianY from the given optics functions.
        beta_y is given in meters and epsn_y in micrometers.
        """
        sigma_y  = np.sqrt(beta_y * epsn_y * 1e-6 / betagamma)
        sigma_yp = sigma_y / beta_y
        return cls(n_macroparticles, sigma_y, sigma_yp, generator_seed)

    def generate(self, beam):
        beam.y = self.sigma_y * self.random_state.randn(self.n_macroparticles)
        beam.yp = self.sigma_yp * self.random_state.randn(self.n_macroparticles)


class GaussianZ(PhaseSpace):
    """Longitudinal Gaussian particle phase space distribution."""

    def __init__(self, n_macroparticles, sigma_z, sigma_dp, is_accepted = None, generator_seed=None):
        """Initiates the longitudinal beam coordinates to a given
        Gaussian shape. If the argument is_accepted is set to
        the is_in_separatrix(z, dp, beam) method of a RFSystems
        object (or similar), macroparticles will be initialised
        until is_accepted returns True.
        """
        self.n_macroparticles = n_macroparticles
        self.sigma_z = sigma_z
        self.sigma_dp = sigma_dp
        self.is_accepted = is_accepted

        self.random_state = RandomState()
        self.random_state.seed(generator_seed)
        
    @classmethod
    def from_optics(cls, n_macroparticles, beta_z, epsn_z, p0, is_accepted = None,
                    generator_seed=None):
        """Initialise GaussianZ from the given optics functions.
        For the argument is_accepted see __init__.
        """
        sigma_z = np.sqrt(beta_z*epsn_z/(4*np.pi) * e/p0)
        # print sigma_z
        # exit(-1)
        sigma_dp = sigma_z / beta_z
        return cls(n_macroparticles, sigma_z, sigma_dp, is_accepted, generator_seed)

    def generate(self, beam):
        beam.z = self.sigma_z * self.random_state.randn(self.n_macroparticles)
        beam.dp = self.sigma_dp * self.random_state.randn(self.n_macroparticles)
        if self.is_accepted:
            self._redistribute(beam)

    def _redistribute(self, beam):
        n = self.n_macroparticles
        for i in xrange(n):
            while not self.is_accepted(beam.z[i], beam.dp[i], beam):
                beam.z[i]  = self.sigma_z * self.random_state.randn(n)
                beam.dp[i] = self.sigma_dp * self.random_state.randn(n)

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

def stationary_exponential(H, Hmax, H0, bunch):

    def psi(dz, dp):
        result = np.exp(H(dz, dp, bunch) / H0) - np.exp(Hmax / H0)
        return result

    return psi

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
    p0 = z0 * Qs / (eta * R)            #Matching condition
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

        p0 = z1 * Qs / (eta * R)
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
