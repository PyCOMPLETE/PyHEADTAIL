'''
@date:   30/09/2015
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

# try to import pycuda, if not available --> skip this test file
try:
    import pycuda.autoinit
    has_pycuda = True
except ImportError:
    has_pycuda = False

@unittest.skipUnless(has_pycuda, 'pycuda not found, skipping')
class TestGPUInterface(unittest.TestCase):
    '''
    Run some tests of the GPU interface. If pycuda could not be imported,
    has_pycuda is set to False and the tests within this class will be skipped
    (see the unittest.skipUnless decorator)
    '''
    def setUp(self):
        self.bunch = self.create_all1_bunch()

    def tearDown(self):
        pass

    def test_if_beam_is_numpy(self):
        self.assertTrue(self.check_if_npndarray(),
            msg='beam.x is not of type np.ndarray')

    def check_if_npndarray(self):
        '''
        Convenience function which checks if beam.x is an
        np.ndarray type
        '''
        return isinstance(self.bunch.x, np.ndarray)

    def create_all1_bunch(self):
        x = np.ones(100)
        y = x.copy()
        z = x.copy()
        xp = x.copy()
        yp = x.copy()
        dp = x.copy()
        coords_n_momenta_dict = {
            'x': x, 'y': y, 'z': z,
            'xp': xp, 'yp': yp, 'dp': dp
        }
        return Particles(
            len(x), 1, e, 1, #never mind the other params
            1, 18., coords_n_momenta_dict
        )

if __name__ == '__main__':
    unittest.main()
