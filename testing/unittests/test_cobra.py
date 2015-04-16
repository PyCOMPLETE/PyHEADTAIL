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
from PyHEADTAIL.general.printers import AccumulatorPrinter
import PyHEADTAIL.cobra_functions.stats as cf


class TestCobra(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.tolerance = 6
        self.n_samples = 1000000
        self.data1_var = 0.001
        #some random data to use for cov/eps/... computations
        self.data1 = np.random.normal(0, np.sqrt(self.data1_var), self.n_samples)
        self.data2 = np.random.normal(5., 0.1, self.n_samples)
        self.data3 = np.random.laplace(loc=-2., scale=0.5, size=self.n_samples)

    def tearDown(self):
        pass
    def test_covariance_for_variance(self):
        """ Test whether the cov_onepass, std, cov return the correct variance
        for a simulated data set
        """
        v1 = cf.cov(self.data1, self.data1)
        v2 = cf.cov_onepass(self.data1, self.data1)
        v3 = cf.std(self.data1)
        self.assertAlmostEqual(v1, self.data1_var, places=self.tolerance,
                               msg='stats.cov() is wrong when computing' +
                               'the variance of a dataset')
        self.assertAlmostEqual(v2, self.data1_var, places=self.tolerance,
                               msg='stats.cov_onepass is wrong when computing' +
                               'the variance of a dataset')

    def test_consistency_covariance(self):
        """ Test whether cov_onepass, cov return the same covariance for a
        simulated data set
        """
        v1 = cf.cov(self.data1, self.data2)
        v2 = cf.cov_onepass(self.data1, self.data2)
        self.assertAlmostEquals(v1, v2, places=self.tolerance,
                                msg='cov, cov_onepass results differ')

    def test_consistency_effective_emittance(self):
        """ Test whether effective_emittance and emittance_old return the
        same values """
        #eps1 = cf.effective_emittance(self.data1, self.data3)
        eps1 = cf.emittance(self.data2, self.data3, None)
        eps2 = cf.emittance_old(self.data2, self.data3)
        self.assertAlmostEquals(eps1, eps2, places=self.tolerance,
                                msg='the new effective emittance computation' +
                                'yields different results than the old one')

if __name__ == '__main__':
    unittest.main()

