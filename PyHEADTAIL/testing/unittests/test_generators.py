'''
@date:   31/03/2015
@author: Stefan Hegglin
Tests for generator
'''
from __future__ import division


import sys, os
BIN = os.path.dirname(__file__) # ./PyHEADTAIL/testing/unittests/
BIN = os.path.abspath(BIN) # absolute path to unittests
BIN = os.path.dirname(BIN) # ../ -->  ./PyHEADTAIL/testing/
BIN = os.path.dirname(BIN) # ../ -->  ./PyHEADTAIL/
BIN = os.path.dirname(BIN) # ../ -->  ./
sys.path.append(BIN)

import unittest

import numpy as np
import scipy.constants as constants
from PyHEADTAIL.trackers.longitudinal_tracking import RFSystems
import PyHEADTAIL.particles.generators as gf
from PyHEADTAIL.general.printers import SilentPrinter

class TestParticleGenerators(unittest.TestCase):
    '''Test class for the new ParticleGenerator (generator_functional.py)'''
    def setUp(self):
        np.random.seed(0)
        self.nparticles = 1000
        self.epsx = 0.5
        self.intensity = 1e11
        self.charge = constants.e
        self.mass = constants.m_p
        self.circumference = 99
        self.gamma = 27.1
        self.generator = gf.ParticleGenerator(
            self.nparticles, self.intensity,
            self.charge, self.mass, self.circumference, self.gamma,
            distribution_x=gf.gaussian2D(0.5),alpha_x=-0.7, beta_x=4, D_x=0,
            distribution_z=gf.gaussian2D(3.0),
            printer=SilentPrinter())
        self.beam = self.generator.generate()

    def tearDown(self):
        pass

    def test_particles_length(self):
        '''Tests whether the coordinate arrays of the resulting beam
        have the correct length'''
        self.assertEqual(self.beam.x.size, self.nparticles,
                         'Length of x-beam coordinate array not correct')

    def test_particles_coordinates(self):
        '''Tests whether only the coordinates specified in the initializer
        are initialized in the beam (e.g. yp is not)
        '''
        with self.assertRaises(AttributeError):
            self.beam.yp

    def test_update_beam_with_existing_coords(self):
        '''Tests whether updating already existing coords produces
        beam coordinates of the correct size
        '''
        self.generator.update(self.beam)
        self.assertEqual(self.beam.x.size, self.nparticles,
                         'Updating existing coordinates leads to wrong' +
                         'coordinate lengths')

    def test_update_beam_with_new_coords(self):
        '''Tests whether adding new coordinates to the beam
        works as expected
        '''
        x_copy = self.beam.x.copy()
        longitudinal_generator = gf.ParticleGenerator(
            self.nparticles, self.intensity, self.charge,
            self.mass, self.circumference, self.gamma,
            distribution_z=gf.gaussian2D(3.0))
        longitudinal_generator.update(self.beam)
        self.assertEqual(self.beam.dp.size, self.nparticles,
                         'Updating the beam with new coordinates leads to' +
                         'faulty coordinates')
        for n in xrange(self.nparticles):
            self.assertAlmostEqual(x_copy[n], self.beam.x[n],
                msg='Updating the beam with new coordinates invalidates' +
                'existing coordinates')

    def test_distributions(self):
        '''Tests whether the specified distributions return the coords
        in the correct format (dimensions). If new distributions are added,
        add them to the test here!
        '''
        # Gaussian
        dist = gf.gaussian2D(0.1)
        self.distribution_testing_implementation(dist)

        # Uniform
        dist = gf.uniform2D(-2., 3.)
        self.distribution_testing_implementation(dist)

    def test_import_distribution(self):
        '''Tests whether import_distribution produces coordinate arrays of the
        correct size'''
        nparticles = 5
        coords = [np.linspace(-2, 2, nparticles),
                  np.linspace(-3, 3, nparticles)]
        import_generator = gf.ParticleGenerator(
                nparticles, 1e11, constants.e, constants.m_p, 100, 10,
                distribution_y=gf.import_distribution2D(coords))
        beam = import_generator.generate()
        self.assertEqual(len(beam.y), nparticles,
                'import_generator produces coords with the wrong length')
        self.assertEqual(len(beam.yp), nparticles,
                'import_generator produces coords with the wrong length')

    def test_rf_bucket_distribution(self):
        '''Tests the functionality of the rf-bucket matchor'''
        #SPS Q20 flattop
        nparticles = 100
        h1 = 4620
        h2 = 4*4620
        V1 = 10e6
        V2 = 1e6
        dphi1 = 0
        dphi2 = 0
        alpha = 0.00308
        p_increment = 0
        long_map = RFSystems(self.circumference, [h1, h2], [V1, V2],
                [dphi1, dphi2], [alpha], self.gamma, p_increment, charge=self.charge, mass=self.mass)
        bucket = long_map.get_bucket(gamma=self.gamma)
        bunch = gf.ParticleGenerator(
                nparticles, 1e11, constants.e, constants.m_p,
                self.circumference, self.gamma,
                distribution_z=gf.RF_bucket_distribution(
                    bucket, epsn_z=0.002, printer=SilentPrinter())).generate()

    def test_cut_bucket_distribution(self):
        '''Tests functionality of the cut-bucket matchor '''
        nparticles = 100
        h1 = 4620
        h2 = 4*4620
        V1 = 10e6
        V2 = 1e6
        dphi1 = 0
        dphi2 = 0
        alpha = 0.00308
        p_increment = 0
        long_map = RFSystems(self.circumference, [h1, h2], [V1, V2],
                [dphi1, dphi2], [alpha], self.gamma, p_increment, charge=self.charge, mass=self.mass)
        bucket = long_map.get_bucket(gamma=self.gamma)
        is_accepted_fn = bucket.make_is_accepted(margin=0.)
        bunch = gf.ParticleGenerator(
                nparticles, 11, constants.e, constants.m_p,
                self.circumference, self.gamma,
                distribution_z=gf.cut_distribution(
                is_accepted=is_accepted_fn,
                distribution=gf.gaussian2D(0.01))).generate()
        self.assertEqual(nparticles, len(bunch.z),
                         'bucket_cut_distribution loses particles')
        self.assertTrue(np.sum(is_accepted_fn(bunch.z, bunch.dp)) == nparticles,
                        'not all particles generated with the cut RF matcher' +
                        ' lie inside the specified separatrix')

    def test_import_distribution_raises_error(self):
        '''Tests whether the generation fails when the number of particles
        and the size of the specified distribution list do not match
        '''
        nparticles = 10
        coords = [np.linspace(-2, 2, nparticles+1),
                  np.linspace(-3, 3, nparticles+1)]
        import_generator = gf.ParticleGenerator(
                nparticles, 1e11, constants.e, constants.m_p, 100, 10,
                distribution_y=gf.import_distribution2D(coords))
        with self.assertRaises(AssertionError):
            beam = import_generator.generate()

    def distribution_testing_implementation(self, distribution):
        '''Call this method with the distribution as a parameter.
        distribution(n_particles) should be a valid command
        '''
        distribution_size = 100
        X = distribution(distribution_size)
        x = X[0]
        p = X[1]
        self.assertEqual(x.size, distribution_size,
                         'space-direction ([0]) of ' + str(distribution) +
                         'has wrong dimension')
        self.assertEqual(p.size, distribution_size,
                         'momentum-direction ([1]) of ' + str(distribution) +
                         'has wrong dimension')


if __name__ == '__main__':
    unittest.main()
