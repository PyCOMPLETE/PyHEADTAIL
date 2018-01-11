'''
@date:   24/11/2015
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
import h5py as hp

from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.monitors.monitors import (
    BunchMonitor, SliceMonitor, CellMonitor)

class TestMonitor(unittest.TestCase):
    ''' Test the BunchMonitor/SliceMonitor'''
    def setUp(self):
        self.n_turns = 10
        self.bunch_fn = 'bunchm'
        self.s_fn = 'sm'
        self.nslices = 5
        self.bunch_monitor = BunchMonitor(filename=self.bunch_fn,
                             n_steps=self.n_turns,
                             write_buffer_every=2, buffer_size=7,
                             stats_to_store=['mean_x', 'macrop'])

    def tearDown(self):
        try:
            os.remove(self.bunch_fn + '.h5')
            os.remove(self.s_fn + '.h5')
            pass
        except:
            pass

    def test_bunchmonitor(self):
        '''
        Test whether the data stored in the h5 file correspond to the
        correct values. Use a mock bunch class which creates an easy
        to check pattern when accessing 'mean_x', 'macrop'
        '''
        mock = self.generate_mock_bunch()
        for i in xrange(self.n_turns):
            self.bunch_monitor.dump(mock)
        bunchdata = hp.File(self.bunch_fn + '.h5')
        b = bunchdata['Bunch']
        self.assertTrue(np.allclose(b['mean_x'],
            np.arange(start=1, stop=self.n_turns+0.5)))
        self.assertTrue(np.allclose(b['macrop'], 99*np.ones(self.n_turns)))

    def test_slicemonitor(self):
        '''
        Test whether the slicemonitor works as expected, use the mock slicer
        '''
        nslices = 3
        mock_slicer = self.generate_mock_slicer(nslices)
        mock_bunch = self.generate_mock_bunch()
        slice_monitor = SliceMonitor(filename=self.s_fn, n_steps=self.n_turns,
                slicer=mock_slicer, buffer_size=11, write_buffer_every=9,
                slice_stats_to_store=['propertyA'],
                bunch_stats_to_store=['mean_x', 'macrop'])
        for i in xrange(self.n_turns):
            slice_monitor.dump(mock_bunch)
        s = hp.File(self.s_fn + '.h5')
        sd = s['Slices']
        sb = s['Bunch']
        self.assertTrue(np.allclose(sb['mean_x'],
            np.arange(start=1, stop=self.n_turns+0.5)))
        self.assertTrue(np.allclose(sb['macrop'], 99*np.ones(self.n_turns)))
        for k in xrange(nslices):
            for j in xrange(self.n_turns):
                self.assertTrue(np.allclose(sd['propertyA'][k,j],
                    k + (j+1)*1000), 'Slices part of SliceMonitor wrong')

    def test_cellmonitor(self):
        '''
        Test whether the cellmonitor works as expected.
        '''
        bunch = self.generate_real_bunch()
        cell_monitor = CellMonitor(
            filename=self.s_fn, n_steps=self.n_turns,
            n_azimuthal_slices=4, n_radial_slices=3,
            radial_cut=bunch.sigma_dp() * 3,
            beta_z=np.abs(0.003 * bunch.circumference / (2*np.pi*0.004)),
            write_buffer_every=9,
        )
        for i in xrange(self.n_turns):
            cell_monitor.dump(bunch)
        s = hp.File(self.s_fn + '.h5')
        sc = s['Cells']

        # to be extended


    def generate_mock_bunch(self):
        '''
        Create a mock class which defines certain attributes which can be
        stored via the BunchMonitor
        '''
        class Mock():
            def __init__(self):
                self.counter = np.zeros(3, dtype=np.int32) #1 for each of mean/std/...
                self.macrop = 99

            def mean_x(self):
                self.counter[0] += 1
                return self.counter[0]

            def mean_y(self):
                self.counter[1] += 1
                return self.counter[1]

            def get_slices(self, slicer, **kwargs):
                return slicer

        return Mock()


    def generate_real_bunch(self):

        #beam parameters
        intensity = 1.234e9
        circumference = 111.
        gamma = 20.1

        #simulation parameters
        macroparticlenumber = 2048
        particlenumber_per_mp = intensity / macroparticlenumber

        x = np.random.uniform(-1, 1, macroparticlenumber)
        y = np.random.uniform(-1, 1, macroparticlenumber)
        z = np.random.uniform(-1, 1, macroparticlenumber)
        xp = np.random.uniform(-0.5, 0.5, macroparticlenumber)
        yp = np.random.uniform(-0.5, 0.5, macroparticlenumber)
        dp = np.random.uniform(-0.5, 0.5, macroparticlenumber)
        coords_n_momenta_dict = {
            'x': x, 'y': y, 'z': z,
            'xp': xp, 'yp': yp, 'dp': dp
        }
        return Particles(
            macroparticlenumber, particlenumber_per_mp, e, m_p,
            circumference, gamma, coords_n_momenta_dict
        )

    def generate_mock_slicer(self, nslices):
        ''' Create a mock slicer to test behaviour'''
        class Mock():
            def __init__(self, nslices):
                self.n_slices = nslices
                self.counter = 0

            @property
            def propertyA(self):
                ''' Return an array of length nslices, np.arange(nslices)
                Add the number of calls * 1000 to the array
                This makes it easy to compare the results
                '''
                self.counter += 1
                prop = np.arange(0, self.n_slices, 1, dtype=np.float64)
                prop += self.counter*1000
                return prop

        return Mock(nslices)


if __name__ == '__main__':
    unittest.main()
