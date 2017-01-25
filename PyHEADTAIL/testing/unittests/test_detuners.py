'''
@date 18/03/2015
@author: Stefan Hegglin
'''

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

import PyHEADTAIL.trackers.detuners as pure_py
from PyHEADTAIL.particles.particles import Particles

class TestDetuner(unittest.TestCase):
    '''Tests for the detuners in trackers/detuners*.py,
    including comparisons between the cython and pure python
    versions
    '''
    def setUp(self):
        self.macroparticlenumber = 4
        self.particlenumber_per_mp = 13
        self.circumference = 180.
        self.gamma = 19.1
        self.alpha_x = 0.2
        self.alpha_y = -0.19
        self.beta_x = 0.43
        self.beta_y = 0.11
        self.app_x = 1e-9
        self.app_y = 3e-8
        self.app_xy = 4e-8
        self.Qp_x = [0.1, 1.1]
        self.Qp_y = [10.1, 0.2]
        self.segment_length = 0.4
        self.dmu_x = 0.2
        self.dmu_y = 0.3

    def tearDown(self):
        pass

    def test_chromaticity(self):
        '''Tests whether the pure python and cython version
        of the chromaticity class are consistent
        '''
        bunch_p = self.create_bunch()
        chromaticity_p = pure_py.Chromaticity(self.Qp_x, self.Qp_y)
        chromaticity_p.generate_segment_detuner(self.dmu_x, self.dmu_y)
        (dqx_pure, dqy_pure) = chromaticity_p[0].detune(bunch_p)

    def test_amplitudedetuning(self):
        '''Tests whether the pure python and cython version
        of the AmplitudeDetung class yield consistent results
        '''
        adetuning_p = pure_py.AmplitudeDetuning(self.app_x,
                self.app_y, self.app_xy)
        adetuning_p.generate_segment_detuner(self.dmu_x, self.dmu_y,
                alpha_x=self.alpha_x, alpha_y=self.alpha_y,
                beta_x=self.beta_x, beta_y=self.beta_y)
        bunch_p = self.create_bunch()
        (dqx_p, dqy_p) = adetuning_p[0].detune(bunch_p)

    def test_from_oct_currents_LHC(self):
        '''Tests whether the pure python and cython version
        of the AmplitudeDetuning.from_ocupole_currents_LHC()
        classmethods yield consistent results
        '''
        i_focusing = 330
        i_defocusing = 490
        detuner_p = pure_py.AmplitudeDetuning.from_octupole_currents_LHC(
                i_focusing, i_defocusing)

    def create_bunch(self):
        np.random.seed(0)
        x = np.random.uniform(-1, 1, self.macroparticlenumber)
        y = np.copy(x)
        z = np.copy(x)
        xp = np.random.uniform(-0.5, 0.5, self.macroparticlenumber)
        yp = np.copy(xp)
        dp = np.copy(xp)
        coords_n_momenta_dict = {
            'x': x, 'y': y, 'z': z,
            'xp': xp, 'yp': yp, 'dp': dp
        }
        return Particles(
            self.macroparticlenumber, self.particlenumber_per_mp, e, m_p,
            self.circumference, self.gamma, coords_n_momenta_dict
        )


if __name__ == '__main__':
    unittest.main()
