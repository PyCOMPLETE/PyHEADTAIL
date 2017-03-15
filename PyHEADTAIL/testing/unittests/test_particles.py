'''
@date:   10/07/2015
@author: Stefan Hegglin, Adrian Oeftiger
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
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.general.printers import SilentPrinter

class TestParticles(unittest.TestCase):

    def setUp(self):
        #beam parameters
        self.intensity = 1.234e9
        self.circumference = 111.
        self.gamma = 20.1

        #simulation parameters
        self.macroparticlenumber = 2048
        self.particlenumber_per_mp = self.intensity/self.macroparticlenumber

        #create a bunch
        self.bunch = self.create_bunch()
        self.slicer = self.create_slicer()

    def tearDown(self):
        pass

    def test_particles_initialisation(self):
        '''Tests whether the parameters passed to Particles()
        are initialized correctly
        '''
        self.assertEqual(self.macroparticlenumber,
                         self.bunch.macroparticlenumber,
                         'initialisation of macroparticlenumber incorrect')
        self.assertEqual(self.gamma,self.bunch.gamma,
                         'initialisation of gamma incorrect')
        self.assertEqual(self.intensity,self.bunch.intensity,
                         'initialisation of intensity incorrect')
        self.assertEqual(self.circumference,self.bunch.circumference,
                         'initialisation of circumference incorrect')

    def test_coords_dict_copy(self):
        '''Tests whether get_coords_n_momenta() returns a copy'''
        coords_n_momenta_copy = self.bunch.get_coords_n_momenta_dict()
        self.assertFalse(coords_n_momenta_copy
                         is self.bunch.coords_n_momenta,
                         'get_coords_n_momenta() returns a reference')

    def test_update_faulty_coords(self):
        '''Tests whether an exception is raised if coords/momenta have
        different lengths than the number of macroparticles
        '''
        coords_n_momenta = self.bunch.get_coords_n_momenta_dict()
        len_mismatch = -1
        coords_n_momenta['x'] = np.zeros(self.macroparticlenumber-len_mismatch)
        with self.assertRaises(Exception):
            self.bunch.update(coords_n_momenta)

    def test_coords_add(self):
        '''Tests whether adding two extra coordinates to the coords_n_momenta
        dictionary works or raises an exception iff the keys are already used
        '''
        extra_coords = {
            'a': np.zeros(self.macroparticlenumber),
            'ap': np.ones(self.macroparticlenumber)
        }
        n_keys_before = len(self.bunch.coords_n_momenta)
        self.bunch.add(extra_coords)
        n_keys_after = len(self.bunch.coords_n_momenta)
        self.assertEqual(n_keys_before+len(extra_coords),n_keys_after,
                         'Particles.update() not working correctly')
        duplicate_coords = {
            'b': np.zeros(self.macroparticlenumber),
            'y': np.ones(self.macroparticlenumber)
        }
        with self.assertRaises(Exception):
            self.bunch.add(duplicate_coords)

    def test_setters_getters(self):
        '''Tests all setters and getters properties of the Particles class'''
        properties=[prop for prop in dir(Particles)
                    if isinstance(getattr(Particles, prop), property)]
        for p in properties:
            self.setter_getter_test(p)

    def setter_getter_test(self, prop):
        '''Tests the setter/getter of property prop via the
        getattr()/setattr() functions. Called by test_setters_getters()
        '''
        new_value = 0.9 * getattr(self.bunch, prop)
        setattr(self.bunch, prop, new_value)
        if isinstance(getattr(self.bunch, prop), np.ndarray):
            self.assertTrue(np.allclose(getattr(self.bunch, prop), new_value),
                            msg='getter/setter for property '
                                + prop + ' incorrect')
        else:
            self.assertAlmostEqual(getattr(self.bunch, prop), new_value,
                                   msg='getter/setter for property '
                                       + prop + ' incorrect')

    def test_get_slices(self):
        '''Tests the get_slices() method on consistency after
        multiple calls.
        '''
        slice_set = self.bunch.get_slices(self.slicer)
        self.assertEqual(slice_set,self.bunch.get_slices(self.slicer),
                         'get_slices() incorrect')

    def test_clean_slices(self):
        '''Tests whether clean_slices() works correctly'''
        slice_set = self.bunch.get_slices(self.slicer)
        self.bunch.clean_slices()
        self.assertTrue(len(self.bunch._slice_sets) == 0,
                        'clean_slices() does not delete the slice set')

    def test_means(self):
        ''' Tests the mean() method of the Particle class '''
        self.assertAlmostEquals(self.bunch.mean_xp(), np.mean(self.bunch.xp),
                                places=5, msg='np.mean() and bunch.mean_xp() '
                                'yield different results')

    def test_sigmaz(self):
        '''Test the sigma_z() method of the Particle class
        Only check the first 3 digits because the sample is small (2048)
        '''
        self.assertAlmostEquals(self.bunch.sigma_z(), np.std(self.bunch.z),
                                places=3, msg='np.std() and bunch.sigma_z() '
                                'yield different results')

    def test_alpha_trans_only(self):
        '''Test whether the computation of alpha, beta, gamma,
        eps works when the beam has no longitudinal phase space.
        '''
        beam_transverse = self.create_transverse_only_bunch()
        beam_transverse.alpha_Twiss_x()
        beam_transverse.alpha_Twiss_y()
        beam_transverse.beta_Twiss_x()
        beam_transverse.beta_Twiss_y()
        beam_transverse.gamma_Twiss_x()
        beam_transverse.gamma_Twiss_y()
        beam_transverse.epsn_x()
        beam_transverse.epsn_y()

    def test_check_error_thrown_dispersion_trans_only(self):
        '''Test whether an AttributeError gets raised when trying to
        compute the dispersion of a beam with no longitudinal phase
        space.
        '''
        beam_transverse = self.create_transverse_only_bunch()
        with self.assertRaises(AttributeError):
            beam_transverse.dispersion_y()
        with self.assertRaises(AttributeError):
            beam_transverse.dispersion_x()

    def test_effective_emittance_vs_emittance(self):
        '''Test whether the effective emittance is the same as the
        emittance for a transverse-only beam.
        '''
        beam_transverse = self.create_transverse_only_bunch()
        self.assertAlmostEquals(
            beam_transverse.epsn_x(),
            beam_transverse.effective_normalized_emittance_x(),
            places = 5,
            msg='beam.effective_normalized_emittance_x() ' +
            'yields a different result than beam.epsn_x() '+
            'for a transverse only beam.'
        )

        self.assertAlmostEquals(
            beam_transverse.epsn_y(),
            beam_transverse.effective_normalized_emittance_y(),
            places = 5,
            msg='beam.effective_normalized_emittance_y() ' +
            'yields a different result than beam.epsn_y() '+
            'for a transverse only beam.'
        )

    def test_id_is_sequence(self):
        '''The beam.id should be a monotonically increasing sequence.'''
        bunch = self.create_bunch()
        self.assertTrue(np.all(bunch.id ==
                               np.arange(1, bunch.macroparticlenumber + 1)),
                        msg='beam.id should be a monotonically increasing'
                        'sequence!')

    def test_sort_particles(self):
        '''Test whether sorting of particles works properly and all particle
        attribute arrays are properly reordered.
        '''
        bunch = self.create_bunch()
        old = {}
        for attr in ['id'] + list(bunch.coords_n_momenta):
            old[attr] = getattr(bunch, attr).copy()
        bunch.sort_for('z')
        new_idx = bunch.id - 1
        for attr, oldarray in old.iteritems():
            self.assertTrue(np.all(oldarray[new_idx] == getattr(bunch, attr)),
                            msg="beam.sort_for('z') should reorder all beam "
                            "particle arrays, but beam." + str(attr) + " is "
                            "missing.")

    def create_bunch(self):
        x = np.random.uniform(-1, 1, self.macroparticlenumber)
        y = np.random.uniform(-1, 1, self.macroparticlenumber)
        z = np.random.uniform(-1, 1, self.macroparticlenumber)
        xp = np.random.uniform(-0.5, 0.5, self.macroparticlenumber)
        yp = np.random.uniform(-0.5, 0.5, self.macroparticlenumber)
        dp = np.random.uniform(-0.5, 0.5, self.macroparticlenumber)
        coords_n_momenta_dict = {
            'x': x, 'y': y, 'z': z,
            'xp': xp, 'yp': yp, 'dp': dp
        }
        return Particles(
            self.macroparticlenumber, self.particlenumber_per_mp, e, m_p,
            self.circumference, self.gamma, coords_n_momenta_dict
        )

    def create_transverse_only_bunch(self):
        x = np.random.uniform(-1, 1, self.macroparticlenumber)
        y = np.random.uniform(-1, 1, self.macroparticlenumber)
        xp = np.random.uniform(-0.5, 0.5, self.macroparticlenumber)
        yp = np.random.uniform(-0.5, 0.5, self.macroparticlenumber)
        coords_n_momenta_dict = {
            'x': x, 'y': y,
            'xp': xp, 'yp': yp
        }
        return Particles(
            self.macroparticlenumber, self.particlenumber_per_mp, e, m_p,
            self.circumference, self.gamma, coords_n_momenta_dict
        )

    def create_slicer(self):
        n_slices = 2
        n_sigma_z = 0.1
        return UniformBinSlicer(n_slices,n_sigma_z)


if __name__ == '__main__':
    unittest.main()
