'''
@date:   11/03/2015
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
from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer



def check_elements_equal(np_array1d):
    value = np_array1d[0]
    for elem in np.nditer(np_array1d):
        if elem != value:
            return False
    return True


class TestSlicing(unittest.TestCase):

    def setUp(self):
        #beam parameters
        self.intensity = 1.234e9
        self.circumference = 111.
        self.gamma = 20.1

        #simulation parameters
        self.macroparticlenumber = 100 #must be multiple of nslices
        self.particlenumber_per_mp = self.intensity/self.macroparticlenumber

        #create a bunch
        self.bunch = self.create_bunch()

        #create a params for slicers
        self.nslices = 5
        self.z_cuts = (-20.,30.) #asymmetric to check uniform_charge_slicer
        self.n_sigma_z = 5
        self.basic_slicer = UniformBinSlicer(self.nslices,
                                                z_cuts=self.z_cuts)
        self.basic_slice_set = self.basic_slicer.slice(self.bunch)



    def tearDown(self):
        pass


    def test_long_cuts(self):
        (cut_tail, cut_head) = self.basic_slicer.get_long_cuts(self.bunch)
        self.assertAlmostEqual(self.z_cuts[0], cut_tail,
                               'get_long_cuts incorrect (tail cut)')
        self.assertAlmostEqual(self.z_cuts[1], cut_head,
                               'get_long_cuts incorrect (head cut)')

    def test_equality(self):
        '''Tests whether two slicers with the same config are equal
        in the sense of the == and != operator (calling __eq__)
        '''
        unif_bin_slicer = UniformBinSlicer(self.nslices, z_cuts=self.z_cuts)
        unif_bin_slicer2 = UniformBinSlicer(self.nslices, z_cuts=self.z_cuts)
        self.assertTrue(unif_bin_slicer == unif_bin_slicer2,
                        'comparing two uniform bin slicers with '+
                        'identical config using == returns False')
        self.assertFalse(unif_bin_slicer != unif_bin_slicer2,
                         'comparing two uniform bin slicers with '+
                         'identical config using != returns True')

    def test_unif_charge_slicer(self):
        '''Tests whether the charges are equally distributed between
        the charges. Only works if nslices divides macroparticlenumber
        '''
        unif_charge_slicer = UniformChargeSlicer(n_slices=self.nslices,
                                                 z_cuts=self.z_cuts)
        slice_set = unif_charge_slicer.slice(self.bunch)
        p_per_slice = slice_set.n_macroparticles_per_slice
        self.assertTrue(check_elements_equal(p_per_slice),
                        'slices in UniformChargeSlicer don\'t have' +
                        'the same number of macroparticles in them')

    def test_sliceset_macroparticles(self):
        '''Tests whether the sum of all particles per slice
        is equal to the specified number of macroparticles when specifying
        z_cuts which lie outside of the bunch
        '''
        #create a bunch and a slice set encompassing the whole bunch
        z_min, z_max = -2., 2.
        bunch = self.create_bunch(zmin=z_min, zmax=z_max)
        z_cuts = (z_min-1,z_max+1)
        slice_set = UniformChargeSlicer(n_slices=self.nslices,
                                        z_cuts=z_cuts).slice(bunch)
        n_particles = sum(slice_set.n_macroparticles_per_slice)
        self.assertEqual(self.macroparticlenumber, n_particles,
                         'the SliceSet lost/added some particles')

    def test_sliceset_dimensions(self):
        '''Tests whether the dimensions of several slice_set properties
        match the specified number of slices
        '''
        self.assertTrue(self.basic_slice_set.slice_widths.size ==
                        self.nslices, 'slice_widths has wrong dimension')
        #print(self.basic_slice_set.slice_positions)
        self.assertTrue(self.basic_slice_set.slice_positions.size ==
                        self.nslices, 'slice_positions has wrong dimension')




    def create_bunch(self, zmin=-1., zmax=1.):
        z = np.linspace(zmin, zmax, num=self.macroparticlenumber)
        y = np.copy(z)
        x = np.copy(z)
        xp = np.linspace(-0.5, 0.5, num=self.macroparticlenumber)
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
