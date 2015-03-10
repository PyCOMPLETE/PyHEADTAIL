'''
@date:   10/03/2015
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

    def tearDown(self):
        pass

    def test_particles_initialisation(self):
        '''Tests whether the parameters passed to Particles()
        are initialized correctly
        '''
        #self.macroparticlenumber = 50
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
        '''Tests whether get_coords_n_momenta() returns a copy or a reference
        '''
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
        '''Tests all setters and getters properties of the Particles class
        '''
        properties=[prop for prop in dir(Particles)
                    if isinstance(getattr(Particles, prop), property)]
        for p in properties:
            self.setter_getter_test(p)

    def setter_getter_test(self,prop):
        '''Tests the setter/getter of property prop via the
        getattr()/setattr() functions. Called by test_setters_getters()
        '''
        new_value = 0.9*getattr(self.bunch,prop)
        setattr(self.bunch,prop,new_value)
        self.assertAlmostEqual(getattr(self.bunch,prop),new_value,
                               msg = 'getter/setter for property '
                                     + prop + ' incorrect')

    def create_bunch(self):
        x = np.random.uniform(-1,1,self.macroparticlenumber)
        y = np.copy(x)
        z = np.copy(x)
        xp = np.random.uniform(-0.5,0.5,self.macroparticlenumber)
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
