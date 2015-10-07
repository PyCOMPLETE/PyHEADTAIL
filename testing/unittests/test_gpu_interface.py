'''
@date:   30/09/2015
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
from PyHEADTAIL.general.contextmanager import GPU
import PyHEADTAIL.trackers.transverse_tracking as tt
import PyHEADTAIL.trackers.simple_long_tracking as lt
from PyHEADTAIL.trackers.detuners import AmplitudeDetuning
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField, WakeTable
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.feedback.widebandfeedback import Pickup, Kicker
# try to import pycuda, if not available --> skip this test file
try:
    import pycuda.autoinit
except ImportError:
    has_pycuda = False
else:
    has_pycuda = True

@unittest.skipUnless(has_pycuda, 'pycuda not found, skipping')
class TestGPUInterface(unittest.TestCase):
    '''
    Run some tests of the GPU interface. If pycuda could not be imported,
    has_pycuda is set to False and the tests within this class will be skipped
    (see the unittest.skipUnless decorator)
    '''
    def setUp(self):
        # machine parameters / optics
        self.circumference = 17.1
        self.nsegments = 4
        self.s = np.linspace(0., self.circumference, self.nsegments+1)
        self.alpha_x = np.linspace(-1.51, 1.52, self.nsegments)
        self.alpha_y = self.alpha_x.copy()
        self.beta_x = np.linspace(0.1, 10., self.nsegments)
        self.beta_y = self.beta_x.copy() + 1.
        self.Qx = 17.89
        self.Qy = 19.11
        self.Dx = 100*np.ones(len(self.alpha_x)) # or len(self.s)?
        self.Dy = self.Dx.copy() - self.beta_y*10
        self.Qs = 0.1

        self.gamma = 15.
        self.h1 = 1
        self.h2 = 2
        self.V1 = 8e3
        self.V2 = 0
        self.dphi1 = 0
        self.dphi2 = np.pi


    def tearDown(self):
        pass


    def test_if_beam_is_numpy(self):
        '''
        Check if beam.x is a numpy array before and after the with statement
        '''
        bunch = self.create_all1_bunch()
        self.assertTrue(self.check_if_npndarray(bunch),
            msg='beam.x is not of type np.ndarray')
        with GPU(bunch) as device:
            foo = 1
        self.assertTrue(self.check_if_npndarray(bunch),
            msg='beam.x is not of type np.ndarray')


    def test_transverse_track_no_detuning(self):
        '''
        Track the GPU bunch through a TransverseMap with no detuning
        Check if results on CPU and GPU are the same
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        transverse_map = tt.TransverseMap(
            self.circumference, self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy)
        self.assertTrue(self._track_cpu_gpu(transverse_map, bunch_cpu,
            bunch_gpu), 'Transverse tracking w/o detuning CPU/GPU differs')


    def test_transverse_track_with_detuning(self):
        '''
        Track the GPU bunch through a TransverseMap with detuning
        Check if results on CPU and GPU are the same
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        adetuner = AmplitudeDetuning(1e-2, 5e-2, 1e-3)
        transverse_map = tt.TransverseMap(
            self.circumference, self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy,
            adetuner)
        self.assertTrue(self._track_cpu_gpu(transverse_map, bunch_cpu,
            bunch_gpu), 'Transverse tracking with detuning CPU/GPU differs')


    def test_longitudinal_linear_map(self):
        '''
        Track through a LinearMap and compare CPU/GPU versions
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        alpha_array = [0.5]
        longitudinal_map = lt.LinearMap(alpha_array, self.circumference,
            self.Qs)
        self.assertTrue(self._track_cpu_gpu([longitudinal_map], bunch_cpu,
            bunch_gpu), 'Longitudinal tracking LinearMap CPU/GPU differs')

    def test_longitudinal_drift(self):
        '''
        Track along a Drift and compare GPU/CPU
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        alpha_array = [0.05]
        shrinkage_p_increment = 0.2
        length = 100.
        longitudinal_map = lt.Drift(alpha_array, length, shrinkage_p_increment)
        self.assertTrue(self._track_cpu_gpu([longitudinal_map], bunch_cpu,
            bunch_gpu), 'Longitudinal tracking Drift CPU/GPU differs')

    def test_longitudinal_RFSystems_map(self):
        '''
        Track through an RFSystems and compare CPU/GPU versions
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        longitudinal_map = lt.RFSystems(
                self.circumference, [self.h1, self.h2], [self.V1, self.V2],
                [self.dphi1, self.dphi2], [0.05], self.gamma, 0,
                D_x=self.Dx[0], D_y=self.Dy[0]
            )
        self.assertTrue(self._track_cpu_gpu([longitudinal_map], bunch_cpu,
            bunch_gpu), 'Longitudinal tracking RFSystems CPU/GPU differs')

    def test_wakefield_circresonator(self):
        '''
        Track through a CircularResonator wakefield
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        bunch_cpu.z += np.arange(len(bunch_cpu.z))
        bunch_gpu.z += np.arange(len(bunch_cpu.z))
        n_slices=5
        frequency = 1e9
        R_shunt = 23e3 # [Ohm]
        Q = 1.
        unif_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=0)
        res = CircularResonator(R_shunt=R_shunt, frequency=frequency, Q=Q)
        wake_field = WakeField(unif_bin_slicer, res)
        self.assertTrue(self._track_cpu_gpu([wake_field], bunch_cpu, bunch_gpu),
            'Tracking Wakefield CircularResonator CPU/GPU differs')


    def test_transverse_damper(self):
        '''
        Track through a transverse damper
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        dampingrate_x = 0.01
        dampingrate_y =  0.05
        damp = TransverseDamper(dampingrate_x, dampingrate_y)
        self.assertTrue(self._track_cpu_gpu([damp], bunch_cpu, bunch_gpu),
            'Tracking TransverseDamper CPU/GPU differs')

    def _test_widebandfeedback(self):
        '''
        !!!!! Wiedeband feedback not ready yet! Skip test
        Track trough a Kicker (class in widebandfeedback)
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        slices = bunch_cpu.get_slices(UniformBinSlicer(n_slices=5, n_sigma_z=0))
        pickup = Pickup(slices)
        kicker = Kicker(pickup)
        self.assertTrue(self._track_cpu_gpu([kicker], bunch_cpu, bunch_gpu),
            'Tracking widebandfeedback CPU/GPU differs. Check if reslicing ' +
            'needed and test is wrong!')



    def _track_cpu_gpu(self, list_of_maps, bunch1, bunch2):
        '''
        Tracks both bunches through the list of maps (once GPU, once CPU)
        and checks whether it yields the same result. Returns True/False.
        Make sure bunch1, bunch2 are two identical objects (not one!)
        Change the actual implementation of the GPU interface/strategy here
        '''
        # GPU
        with GPU(bunch1) as device:
            for m in list_of_maps:
                m.track(bunch1)
        # CPU
        for m in list_of_maps:
            m.track(bunch2)
        for att in bunch1.coords_n_momenta | set(['id']):
            if not np.allclose(getattr(bunch1, att), getattr(bunch2, att),
                               rtol=1.e-5, atol=1.e-8):
                return False
        return True


    def check_if_npndarray(self, bunch):
        '''
        Convenience function which checks if beam.x is an
        np.ndarray type
        '''
        return isinstance(bunch.x, np.ndarray)

    def create_all1_bunch(self):
        np.random.seed(1)
        x = np.random.normal(size=100)
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
            len(x), 1, e, 1, #never mind the other params
            1, 18., coords_n_momenta_dict
        )

if __name__ == '__main__':
    unittest.main()
