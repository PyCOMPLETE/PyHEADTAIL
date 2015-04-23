'''
@date:   15/04/2015
@author: Stefan Hegglin
'''
from __future__ import division

import sys, os
BIN = os.path.dirname(__file__) # ./PyHEADTAIL/testing/unittests/
BIN = os.path.abspath( BIN ) # absolute path to unittests
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/testing/
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/
BIN = os.path.dirname( BIN ) # ../ -->  ./
sys.path.append(BIN)

import unittest

import numpy as np
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.particles.generators import Gaussian6DTwiss
import PyHEADTAIL.cobra_functions.stats as cf
from PyHEADTAIL.trackers.simple_long_tracking import LinearMap



class TestCobra(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.tolerance = 3
        self.n_samples = 100000
        self.data1_var = 1.0
        #some random data to use for cov/eps/... computations
        self.data1 = np.random.normal(0, np.sqrt(self.data1_var), self.n_samples)
        self.data2 = np.random.normal(5., 2.1, self.n_samples)
        self.data3 = np.random.laplace(loc=-2., scale=0.5, size=self.n_samples)

    def tearDown(self):
        pass

    def test_consistency_for_std(self):
        """ Test whether the cf.std, np.std return the correct std
        for a simulated data set
        """
        s_cobra = cf.std(self.data1)
        s_numpy = cf.std(self.data1)
        self.assertAlmostEqual(s_cobra, s_numpy, places=self.tolerance,
                               msg='cobra std() yields a different result ' +
                               'than numpy.std()')

    def test_consistency_covariance(self):
        """ Test whether cov and np.cov return the same covariance
        for a simulated data set
        """
        v_cobra = cf.cov(self.data1, self.data2)
        v_numpy = np.cov(self.data1, self.data2)[0,1]
        self.assertAlmostEquals(v_cobra, v_numpy, places=self.tolerance,
                                msg='cobra cov() yields a different result ' +
                                'than numpy.cov()')

    def test_eta_prime_is_zero(self):
        """ Test whether computing eta_prime of a beam generated using
        Gaussian6dTwiss is 0. This should be true for alphax, alphay = 0.
        Otherwise a correlation will be present. The error decreases with
        increasing number of particles"""
        bunch = self.generate_gaussian6dBunch(1000000, 0, 0, 1, 1, 5, 100)
        eta_prime_x = cf.dispersion(bunch.xp, bunch.dp)
        weak_tol = 2
        self.assertAlmostEquals(eta_prime_x, 0., places=weak_tol,
                                msg='eta_prime_x is not zero but ' + str(eta_prime_x))
        eta_prime_y = cf.dispersion(bunch.yp, bunch.dp)
        self.assertAlmostEquals(eta_prime_y, 0., places=weak_tol,
                                msg='eta_prime_y is not zero but ' + str(eta_prime_y))


    def test_cov_of_uncorrelated_data_is_zero(self):
        """ Test whether the covariance of two uncorrelated normally distributed
        data vectors is zero. Only works for big sample sizes / big tolerance
        """
        d1 = np.random.normal(100, 2., self.n_samples)
        d2 = np.random.normal(200, 0.2, self.n_samples)
        self.assertAlmostEquals(cf.cov(d1, d2), 0.0,
                                places=self.tolerance,
                                msg='cobra cov() of two uncorrelated ' +
                                'Gaussians != 0')

    def generate_gaussian6dBunch(self,n_macroparticles, alpha_x, alpha_y, beta_x,
                                  beta_y, dispx, dispy,
                                  gamma = 3730.27):
        Q_s = 0.0020443
        C = 26658.883
        alpha_0 = [0.0003225]
        linear_map = LinearMap(alpha_0, C, Q_s)

        intensity = 1.05e11
        sigma_z = 0.0059958
        gamma_t = 1. / np.sqrt(alpha_0)
        p0 = np.sqrt(gamma**2 - 1) * m_p * c
        beta_z = (linear_map.eta(dp=0, gamma=gamma) * linear_map.circumference /
              (2 * np.pi * linear_map.Qs))
        epsn_x = 3.75e-6 # [m rad]
        epsn_y = 3.75e-6 # [m rad]
        epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e)
        bunch = Gaussian6DTwiss(
            macroparticlenumber=n_macroparticles, intensity=intensity, charge=e,
            gamma_reference=gamma, mass=m_p, circumference=C,
            alpha_x=0., beta_x=1., epsn_x=epsn_x,
            alpha_y=0., beta_y=1., epsn_y=epsn_y,
            beta_z=beta_z, epsn_z=epsn_z).generate()
        # Scale to correct beta and alpha
        bunch.x *= np.sqrt(beta_x)
        bunch.xp = -alpha_x/np.sqrt(beta_x) * bunch.x + 1./np.sqrt(beta_x) * bunch.xp
        bunch.y = np.sqrt(beta_y)*bunch.y
        bunch.yp = -alpha_y/np.sqrt(beta_y) * bunch.y + 1./np.sqrt(beta_y) * bunch.yp
        bunch.x += dispx * bunch.dp
        bunch.y += dispy * bunch.dp
        return bunch

if __name__ == '__main__':
    unittest.main()

