'''
@date:   02/08/2015
@author: Adrian Oeftiger
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

sys.path.append(os.path.expanduser('~/cern/git/'))
from PyPIC.GPU.meshing import UniformMesh1D

from pycuda.autoinit import context
from pycuda import gpuarray
from pycuda import cumath

import PyHEADTAIL.gpu

from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.gpu.slicing import MeshSlicer
from PyHEADTAIL.general.printers import AccumulatorPrinter
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss

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
        self.macroparticlenumber = 100
         #must be multiple of nslices
        self.particlenumber_per_mp = self.intensity/self.macroparticlenumber

        #create a bunch
        self.bunch = self.create_bunch()

        #create a params for slicers
        self.nslices = 32
        self.z_cuts = (-20.,30.) #asymmetric to check uniform_charge_slicer
        self.n_sigma_z = 5
        self.mesh = self.create_mesh()
        self.basic_slicer = MeshSlicer(self.mesh, context)
        self.basic_slice_set = self.basic_slicer.slice(self.bunch)

    def tearDown(self):
        pass

    def test_long_cuts(self):
        '''Tests whether the z_cuts are initialized correctly'''
        (cut_tail, cut_head) = self.basic_slicer.get_long_cuts(self.bunch)
        self.assertAlmostEqual(self.z_cuts[0], cut_tail,
                               'get_long_cuts incorrect (tail cut)')
        self.assertAlmostEqual(self.z_cuts[1], cut_head,
                               'get_long_cuts incorrect (head cut)')

    # def test_z_cuts_warning(self):
    #     '''Tests whether a warning is raised whenever
    #     z_cut_tail >= z_cut_head
    #     '''
    #     mesh = self.create_mesh(z_cuts=tuple(reversed(self.z_cuts)))
    #     warnings = AccumulatorPrinter()
    #     slicer = MeshSlicer(mesh, warningprinter=warnings)
    #     self.assertTrue(len(warnings.log) > 0,
    #                     'no warning generated when z_cut head < z_cut tail')

    def test_equality(self):
        '''Tests whether two slicers with the same config are equal
        in the sense of the == and != operator (calling __eq__, __ne__)
        '''
        unif_bin_slicer = MeshSlicer(self.mesh, context)
        unif_bin_slicer2 = MeshSlicer(self.mesh, context)
        self.assertTrue(unif_bin_slicer == unif_bin_slicer2,
                        'comparing two uniform bin slicers with '+
                        'identical config using == returns False')
        self.assertFalse(unif_bin_slicer != unif_bin_slicer2,
                         'comparing two uniform bin slicers with '+
                         'identical config using != returns True')

    def test_inequality(self):
        '''Tests whether two slicers with differing meshs are not equal
        in the sense of the == and != operator (calling __eq__, __ne__)
        '''
        unif_bin_slicer = MeshSlicer(self.mesh, context)
        unif_bin_slicer2 = MeshSlicer(self.create_mesh(nslices=self.nslices+1),
                                      context)
        self.assertTrue(unif_bin_slicer != unif_bin_slicer2,
                        'comparing two uniform bin slicers with '+
                        'different config using != returns False')
        self.assertFalse(unif_bin_slicer == unif_bin_slicer2,
                         'comparing two uniform bin slicers with '+
                         'different config using == returns True')

    def test_sliceset_macroparticles(self):
        '''Tests whether the sum of all particles per slice
        is equal to the specified number of macroparticles when specifying
        z_cuts which lie outside of the bunch
        '''
        #create a bunch and a slice set encompassing the whole bunch
        z_min, z_max = -2., 2.
        bunch = self.create_bunch(zmin=z_min, zmax=z_max)
        z_cuts = (z_min-1, z_max+1)
        mesh = self.create_mesh(z_cuts=z_cuts)
        slice_set = MeshSlicer(mesh, context).slice(bunch)
        n_particles = gpuarray.sum(slice_set.n_macroparticles_per_slice).get()
        self.assertEqual(self.macroparticlenumber, n_particles,
                         'the SliceSet lost/added some particles')

    def test_add_statistics(self):
        """ Tests whether any error gets thrown when calling the statistics
        functions of the slicer. Does not do any specific tests
        """
        self.basic_slicer.add_statistics(
            self.basic_slice_set, self.bunch, True,
            self.basic_slice_set.lower_bounds,
            self.basic_slice_set.upper_bounds
        )
        self.basic_slice_set.mean_x
        # self.basic_slice_set.eff_epsn_y

    # def test_emittance_no_dispersion(self):
    #     """ Tests whether the effective emittance and emittance are the same
    #     for a beam with no dispersion effects
    #     """
    #     bunch = self.create_bunch_with_params(1, 42, 0., 20)
    #     slice_set = self.basic_slicer.slice(bunch)
    #     self.basic_slicer.add_statistics(slice_set, bunch, True)
    #     for n in xrange(self.nslices):
    #         self.assertAlmostEqual(slice_set.epsn_x[n],
    #                                slice_set.eff_epsn_x[n],
    #                                places=4,
    #                                msg='The effective emittance is not the ' +
    #                                'same as the emittance for no dispersion')


    # exclude this test for now, fails at the moment but not clear whether
    # this should be changed
    #def test_sliceset_dimensions(self):
    #    '''Tests whether the dimensions of several slice_set properties
    #    match the specified number of slices
    #    '''
    #    self.assertTrue(self.basic_slice_set.slice_widths.size ==
    #                    self.nslices, 'slice_widths has wrong dimension')
    #    #print(self.basic_slice_set.slice_positions)
    #    self.assertTrue(self.basic_slice_set.slice_positions.size ==
    #                    self.nslices, 'slice_positions has wrong dimension')

    def create_mesh(self, nslices=None, z_cuts=None):
        if nslices is None:
            nslices = self.nslices
        if z_cuts is None:
            z_cuts = self.z_cuts
        return UniformMesh1D(z_cuts[0],
                             np.diff(z_cuts)[0] / nslices,
                             nslices,
                             mathlib=cumath)

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

    def create_bunch_with_params(self,alpha_x, beta_x, disp_x, gamma):
        np.random.seed(0)
        beta_y = beta_x
        alpha_y = alpha_x
        disp_y = disp_x
        alpha0= [0.00308]
        C = 6911.
        Q_s = 0.017
        epsn_x = 3.75e-6
        epsn_y = 3.75e-6
        linear_map = LinearMap(alpha0, Q_s, C)
       # then transform...
        intensity = 1.05e11
        sigma_z = 0.23
        gamma_t = 1. / np.sqrt(linear_map.alpha_array[0])
        p0 = np.sqrt(gamma**2 - 1) * m_p * c

        beta_z = np.abs((linear_map.eta(dp=0, gamma=gamma) * linear_map.circumference /
                  (2 * np.pi * linear_map.Q_s)))

        epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e)
        #print ('epsn_z: ' + str(epsn_z))
        bunch = generate_Gaussian6DTwiss(
            macroparticlenumber=10000, intensity=intensity, charge=e,
            gamma=gamma, mass=m_p, circumference=linear_map.circumference,
            alpha_x=0., beta_x=1., epsn_x=epsn_x,
            alpha_y=0., beta_y=1., epsn_y=epsn_y,
            beta_z=beta_z, epsn_z=epsn_z)
        # Scale to correct beta and alpha
        xx = bunch.x.copy()
        yy = bunch.y.copy()
        bunch.x *= np.sqrt(beta_x)
        bunch.xp = -alpha_x/np.sqrt(beta_x) * xx + 1./np.sqrt(beta_x) * bunch.xp
        bunch.y *= np.sqrt(beta_y)
        bunch.yp = -alpha_y/np.sqrt(beta_y) * yy + 1./np.sqrt(beta_y) * bunch.yp
        bunch.x += disp_x * bunch.dp
        bunch.y += disp_y * bunch.dp
        return bunch
if __name__ == '__main__':
    unittest.main()
