'''
@date:   12/03/2015
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
from PyHEADTAIL.aperture.aperture import (
    RectangularApertureX, RectangularApertureY, RectangularApertureZ,
    CircularApertureXY, EllipticalApertureXY
    )


class TestAperture(unittest.TestCase):
    def setUp(self):
        self.macroparticlenumber = 9719
        self.particlenumber_per_mp = 77
        self.circumference = 71.2
        self.gamma = 18.1

    def tearDown(self):
        pass

    def test_no_loss_RectX(self):
        '''Tests whether the RectangularApertureX does not tag
        particles as lost when all particles are inside the region
        '''
        bunch = self.create_unif_bunch(xmin=-5., xmax=-1.)
        n_particles_before = bunch.macroparticlenumber
        coords = bunch.get_coords_n_momenta_dict()
        x_low = min(coords['x']) - 0.1
        x_high = max(coords['x']) + 0.1
        rect_x_aperture = RectangularApertureX(x_low, x_high)
        rect_x_aperture.track(bunch)
        n_particles_after = bunch.macroparticlenumber
        self.assertEqual(n_particles_before, n_particles_after,
                         'error in RectangularApertureX: loses particles ' +
                         'which are inside the specified boundaries')

    def test_no_loss_RectY(self):
        '''Tests whether the RectangularApertureY does not tag
        particles as lost when all mparticles are inside the region
        '''
        bunch = self.create_unif_bunch()
        n_particles_before = bunch.macroparticlenumber
        coords = bunch.get_coords_n_momenta_dict()
        y_low = min(coords['y']) - 0.1
        y_high = max(coords['y']) + 0.1
        rect_y_aperture = RectangularApertureY(y_low, y_high)
        rect_y_aperture.track(bunch)
        n_particles_after = bunch.macroparticlenumber
        self.assertEqual(n_particles_before, n_particles_after,
                         'error in RectangularApertureY: loses particles ' +
                         'which are inside the specified boundaries')

    def test_no_loss_RectZ(self):
        '''Tests whether the RectangularApertureZ does not tag
        particles as lost when all mparticles are inside the region
        '''
        bunch = self.create_unif_bunch()
        n_particles_before = bunch.macroparticlenumber
        coords = bunch.get_coords_n_momenta_dict()
        z_low = min(coords['z']) - 0.1
        z_high = max(coords['z']) + 0.1
        rect_z_aperture = RectangularApertureZ(z_low, z_high)
        rect_z_aperture.track(bunch)
        n_particles_after = bunch.macroparticlenumber
        self.assertEqual(n_particles_before, n_particles_after,
                         'error in RectangularApertureZ: loses particles ' +
                         'which are inside the specified boundaries')

    def test_no_loss_ElliptXY(self):
        '''Tests whether the EllipticalApertureXY does not tag
        particles as lost when the ellipse encompasses the all mparticles
        '''
        bunch = self.create_unif_bunch()
        n_particles_before = bunch.macroparticlenumber
        coords = bunch.get_coords_n_momenta_dict()
        x_ext = 2*max(coords['x'])
        y_ext = 2*max(coords['y'])
        ell_xy_aperture = EllipticalApertureXY(x_ext,y_ext)
        ell_xy_aperture.track(bunch)
        n_particles_after = bunch.macroparticlenumber
        self.assertEqual(n_particles_before, n_particles_after,
                         'error in EllipticalApertureXY: loses particles ' +
                         'which are inside the specified boundaries')

    def test_total_loss_CircXY(self):
        '''Tests whether the CircularApertureXY tags particles outside
        the specified boundary as lost. After tracking, the number
        of particles should be 0
        '''
        bunch = self.create_unif_bunch(xmin=-1., xmax=-0.5,
                                       ymin=-1., ymax=-0.5)
        warnings = AccumulatorPrinter()
        circxy_aperture = CircularApertureXY(radius=0.1,
                                             warningprinter=warnings)
        circxy_aperture.track(bunch)
        n_particles_after = bunch.macroparticlenumber
        self.assertTrue(n_particles_after == 0,
                        'error in CircularApertureXY: the number of ' +
                        'particles should be zero after this operation')
        self.assertTrue(len(warnings.log) > 0,
                        'no warning generated when all particles were lost')

    def create_unif_bunch(self, xmin=-1., xmax=1., ymin=-1., ymax=1.,
                          zmin=-1., zmax=1, xpmin=-1., xpmax=1.,
                          ypmin=-1., ypmax=1., dpmin=-1., dpmax=1.):
        x = np.random.uniform(xmin, xmax, self.macroparticlenumber)
        y = np.random.uniform(ymin, ymax, self.macroparticlenumber)
        z = np.random.uniform(zmin, zmax, self.macroparticlenumber)
        xp = np.random.uniform(xpmin, xpmax, self.macroparticlenumber)
        yp = np.random.uniform(ypmin, ypmax, self.macroparticlenumber)
        dp = np.random.uniform(dpmin, dpmax, self.macroparticlenumber)
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
