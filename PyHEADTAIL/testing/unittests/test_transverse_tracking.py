'''
@date:   17/03/2015
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
import PyHEADTAIL.trackers.transverse_tracking as pure_py
from PyHEADTAIL.trackers.detuners import AmplitudeDetuning
from PyHEADTAIL.general.printers import SilentPrinter

class TestTransverseTracking(unittest.TestCase):
    '''Tests for functions and classes in the
    trackers/transverse_tracking.py and transverse_tracking_cython.pyx
    '''
    def setUp(self):
        self.macroparticlenumber = 1000
        self.circumference = 17.1
        self.nsegments = 10
        self.s = np.linspace(0., self.circumference, self.nsegments+1)
        self.alpha_x = np.linspace(-1.51, 1.52, self.nsegments)
        self.alpha_y = self.alpha_x.copy()
        self.beta_x = np.linspace(0.1, 1., self.nsegments)
        self.beta_y = self.beta_x.copy()
        self.Qx = 17.89
        self.Qy = 19.11
        self.Dx = 100*np.ones(len(self.alpha_x)) # or len(self.s)?
        self.Dy = self.Dx.copy()

    def tearDown(self):
        pass

    def test_tracking_without_detuner(self):
        '''Tests whether the cython and pure python version of the
        TransverseMap yield the same results (x,y,xp,yp) when tracking a beam
        without detuners
        '''
        pure_python_map = pure_py.TransverseMap(
            self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy,
            printer=SilentPrinter()
        )

        beam_p = self.create_bunch()

        for s in pure_python_map:
            s.track(beam_p)

    def test_tracking_with_detuner(self):
        '''Tests whether the cython and pure python version of the
        TransverseMap yield the same results (x,y,xp,yp) when tracking a beam
        with detuners
        '''
        adetuner = AmplitudeDetuning(1e-2, 5e-2, 1e-3)
        pure_python_map = pure_py.TransverseMap(
            self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy,
            adetuner, printer=SilentPrinter()
        )

        beam_p = self.create_bunch()

        for s in pure_python_map:
            s.track(beam_p)

    def create_bunch(self):
        np.random.seed(0) #set seed to make results reproducible
        x = np.random.uniform(-0.1, 0.1, self.macroparticlenumber)
        y = np.random.uniform(-0.1, 0.1, self.macroparticlenumber)
        z = x.copy()
        xp = np.random.uniform(0., 1., self.macroparticlenumber)
        yp = np.random.uniform(0., 1., self.macroparticlenumber)
        dp = x.copy()
        coords_n_momenta_dict = {
            'x': x, 'y': y, 'z': z,
            'xp': xp, 'yp': yp, 'dp': dp
        }
        return Particles(
            self.macroparticlenumber, 1, e, 1, #never mind the other params
            1, 18., coords_n_momenta_dict
        )

if __name__ == '__main__':
    unittest.main()
