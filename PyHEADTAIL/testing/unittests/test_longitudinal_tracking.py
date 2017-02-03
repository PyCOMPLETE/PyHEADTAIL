'''
@date:   16/03/2015
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
from PyHEADTAIL.trackers.longitudinal_tracking import (
    Drift, Kick, LongitudinalOneTurnMap, RFSystems, LinearMap
    )
from PyHEADTAIL.general.printers import AccumulatorPrinter


class TestSimpleLongTracking(unittest.TestCase):
    '''Tests for functions/classes in the
    trackers/longitudinal_tracking.py file
    '''
    def setUp(self):
        self.alpha_array = [10]
        self.macroparticlenumber = 5

    def tearDown(self):
        pass

    def test_drift_track(self):
        '''Tests whether the Drift.track() method does not change any
        coordinates other than the z-component
        '''
        beam = self.create_all1_bunch()
        beam2 = self.create_all1_bunch()
        length = 0.1
        drift = Drift(self.alpha_array, length,
                      warningprinter=AccumulatorPrinter())
        drift.track(beam)
        self.assertTrue(np.allclose(beam.x,beam2.x),
                        'x coord of beam has changed in drift.track()')
        self.assertTrue(np.allclose(beam.y,beam2.y),
                        'y coord of beam has changed in drift.track()')
        self.assertTrue(np.allclose(beam.xp,beam2.xp),
                        'xp coord of beam has changed in drift.track()')
        self.assertTrue(np.allclose(beam.yp,beam2.yp),
                        'yp coord of beam has changed in drift.track()')
        self.assertTrue(np.allclose(beam.dp,beam2.dp),
                        'dp coord of beam has changed in drift.track()')

    def test_drift_cleans_slices(self):
        '''Tests whether the slice_sets are deleted when the track() method
        of Drift is called
        '''
        beam = self.create_all1_bunch()
        beam._slice_sets = {'mock': 42}
        length = 0.2
        drift = Drift(self.alpha_array, length)
        drift.track(beam)
        self.assertFalse(beam._slice_sets, 'slice_sets not deleted after' +
                         'calling Drift.track() [changing the z-coordinates' +
                         ' of the beam]')


    def test_linear_map_higher_order_warning(self):
        ''' Tests whether a warning is generated when higher order terms
        are specified for a linear map (alpha_array)
        '''
        alpha_array3 = [1, 2, 3]
        circumference = 1.
        Q_s = 0.011
        warnings = AccumulatorPrinter()
        linear_map = LinearMap(alpha_array3, circumference, Q_s,
                               warningprinter=warnings)
        self.assertTrue(warnings.log, 'No warning generated when specifying' +
                        'higher order terms in LinearMap')

    def test_linear_map_cleans_slices(self):
        '''Tests whether the slice_sets are deleted when the track() method
        of the LinearMap is called
        '''
        circumference = 1.
        Q_s = 0.012
        linear_map = LinearMap(self.alpha_array, circumference, Q_s,
                               printer=AccumulatorPrinter())
        beam = self.create_all1_bunch()
        sliceset_mock = {'mock': 42}
        beam._slice_sets = sliceset_mock
        linear_map.track(beam)
        self.assertFalse(beam._slice_sets, 'slice_sets not deleted after ' +
                         'calling LinearMap.track() [changing the ' +
                         'z-coordinates of the beam]')

    def test_linear_map_track(self):
        '''Tests whether only the dp and z coordinates are modified when
        applying the track() method of LinearMap
        '''
        circumference = 1.
        Q_s = 0.012
        linear_map = LinearMap(self.alpha_array, circumference, Q_s)
        beam = self.create_all1_bunch()
        beam2 = self.create_all1_bunch()
        linear_map.track(beam)
        self.assertTrue(np.allclose(beam.x,beam2.x),
                        'x coord of beam has changed in LinearMap.track()')
        self.assertTrue(np.allclose(beam.y,beam2.y),
                        'y coord of beam has changed in LinearMap.track()')
        self.assertTrue(np.allclose(beam.xp,beam2.xp),
                        'xp coord of beam has changed in LinearMap.track()')
        self.assertTrue(np.allclose(beam.yp,beam2.yp),
                        'yp coord of beam has changed in LinearMap.track()')

    def test_rfsystems_initialization(self):
        '''Tests the initialization of RFSystems including setting a
        warningprinter/printer. No assert included, test is passed if
        no exception is thrown.
        '''
        circumference = 1.
        harmonic_list = [1, 2, 3]
        voltage_list = harmonic_list
        phi_offset_list = harmonic_list
        gamma = 19.
        rf = RFSystems(circumference, harmonic_list, voltage_list,
                       phi_offset_list, self.alpha_array, gamma,
                       charge = e,
                       warningprinter=AccumulatorPrinter(),
                       printer=AccumulatorPrinter())

    def create_all1_bunch(self):
        x = np.ones(self.macroparticlenumber)
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
            self.macroparticlenumber, 1, e, 1, #never mind the other params
            1, 18., coords_n_momenta_dict
        )


if __name__ == '__main__':
    unittest.main()
