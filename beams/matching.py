from __future__ import division
'''
@file matching
@author Kevin Li
@date February 2014
@brief Module for matching transverse and longitudinal distributions
@copyright CERN
'''


import numpy as np
from beams.distributions import stationary_exponential
from scipy.integrate import quad, dblquad
from scipy.constants import c, e


def match_transverse(epsn_x, epsn_y, ltm=None):

    beta_x, beta_y = ltm.beta_x, ltm.beta_y

    def match(bunch):
        sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (bunch.gamma * bunch.beta))
        sigma_xp = sigma_x / beta_x
        sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (bunch.gamma * bunch.beta))
        sigma_yp = sigma_y / beta_y

        bunch.x *= sigma_x
        bunch.xp *= sigma_xp
        bunch.y *= sigma_y
        bunch.yp *= sigma_yp

    return match

def match_longitudinal(length, bucket, matching=None):

    if not matching and isinstance(bucket, float):
        def match(bunch):
            match_none(bunch, length, bucket)
    elif matching == 'simple':
        def match(bunch):
            try:
                match_simple(bunch, length, bucket)
            except AttributeError:
                raise TypeError("Bad bucket instance for matching!")
    elif matching == 'full':
        def match(bunch):
            try:
                match_full(bunch, length, bucket)
            except AttributeError:
                raise TypeError("Bad bucket instance for matching!")
    else:
        raise ValueError("Unknown matching " + str(matching) + " for bucket " + str(bucket))

    return match

def match_none(bunch, length, bucket):

    sigma_dz = length
    epsn_z = bucket
    sigma_dp = epsn_z / (4 * np.pi * sigma_dz) * e / bunch.p0

    print sigma_dz, sigma_dp
    bunch.dz *= sigma_dz
    bunch.dp *= sigma_dp

# @profile
def match_simple(bunch, sigma_dz, bucket):    
    R = bucket.circumference / (2 * np.pi)
    eta = bucket.eta(bunch)
    Qs = bucket.Qs(bunch)
    
    # Assuming a gaussian-type stationary distribution
    sigma_dp = sigma_dz * Qs / eta / R
    unmatched(bunch, sigma_dz, sigma_dp, bucket)
                
def unmatched(bunch, sigma_dz, sigma_dp, bucket):

    R = bucket.circumference / (2 * np.pi)
    eta = bucket.eta(bunch)
    Qs = bucket.Qs(bunch)

    # Assuming a gaussian-type stationary distribution
    H0 = eta * bunch.beta * c * sigma_dp ** 2
    epsn_z = 4 * np.pi * Qs / eta / R * sigma_dz ** 2 * bunch.p0 / e
    
    n_particles = len(bunch.dz)
    for i in xrange(n_particles):
        if not bucket.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
            while not bucket.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
                bunch.dz[i] = sigma_dz * np.random.randn()
                bunch.dp[i] = sigma_dp * np.random.randn()
                
def match_full(bunch, length, bucket):

    R = bucket.circumference / (2 * np.pi)
    eta = bucket.eta(bunch)
    Qs = bucket.Qs(bunch)

    zmax = np.pi * R / bucket.h
    pmax = 2 * Qs / eta / bucket.h
    Hmax1 = bucket.hamiltonian(zmax, 0, bunch)
    Hmax2 = bucket.hamiltonian(0, pmax, bunch)
    # assert(Hmax1 == Hmax2)
    Hmax = Hmax1
    epsn_z = np.pi / 2 * zmax * pmax * bunch.p0 / e
    print '\nStatistical parameters from RF bucket:'
    print 'zmax:', zmax, 'pmax:', pmax, 'epsn_z:', epsn_z

    # Assuming a gaussian-type stationary distribution
    sigma_dz = length # np.std(bunch.dz)
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * bunch.beta * c * sigma_dp ** 2
    epsn_z = 4 * np.pi * Qs / eta / R * sigma_dz ** 2 * bunch.p0 / e
    print '\nStatistical parameters from initialisation:'
    print 'sigma_dz:', sigma_dz, 'sigma_dp:', sigma_dp, 'epsn_z:', epsn_z
    
    print '\n--> Bunchlength:'
    sigma_dz = bunchlength(bunch, bucket, sigma_dz)
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * bunch.beta * c * sigma_dp ** 2

    psi = stationary_exponential(bucket.hamiltonian, Hmax, H0, bunch)

    zl = np.linspace(-zmax, zmax, 1000) * 1.5
    pl = np.linspace(-pmax, pmax, 1000) * 1.5
    zz, pp = np.meshgrid(zl, pl)
    HH = psi(zz, pp)
    HHmax = np.amax(HH)

    n_particles = len(bunch.x)
    for i in xrange(n_particles):
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
