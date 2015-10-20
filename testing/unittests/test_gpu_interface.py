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
from PyHEADTAIL.trackers.detuners import AmplitudeDetuning, Chromaticity
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField, WakeTable
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.feedback.widebandfeedback import Pickup, Kicker
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor


try:
    import PyCERNmachines.CERNmachines as m
except ImportError:
    has_PyCERNmachines= False
else:
    has_PyCERNmachines = True

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
        self.nsegments = 10
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

        self.gamma = 15 #lhc=7000, sps=27
        self.h1 = 1
        self.h2 = 2
        self.V1 = 8e3
        self.V2 = 0
        self.dphi1 = 0
        self.dphi2 = np.pi

        self.n_macroparticles = 2


    def tearDown(self):
        #os.remove('bunchmonitor.tmp.h5')
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

    @unittest.skipUnless(has_PyCERNmachines, 'No PyCERNmachines.')
    def test_transverse_track_with_Adetuning(self):
        '''
        Track the GPU bunch through a TransverseMap with amplitude detuning
        Check if results on CPU and GPU are the same
        Special test since it requires the PyCERNmachines module to create
        the LHC beam. Skip if not available.
        '''
        # Use a real bunch (not all set to 1), otherwise the Action/detuning
        # gets too big and the numbers blow up.
        Dx = np.append(np.linspace(0., 20., self.nsegments),[0])
        # add some dispersion/alpha
        lhc = m.LHC(n_segments=self.nsegments, machine_configuration='450GeV',
                    app_x=1e-9, app_y=2e-9, app_xy=-1.5e-11,
                    chromaticity_on=False, amplitude_detuning_on=True,
                    alpha_x=1.2*np.ones(self.nsegments), D_x=Dx)
        # create pure Python map, PyCERNmachine uses Cython.
        adetuner = AmplitudeDetuning(lhc.app_x, lhc.app_y, lhc.app_xy)
        transverse_map = tt.TransverseMap(
            lhc.circumference, lhc.s, lhc.alpha_x, lhc.beta_x,
            lhc.D_x, lhc.alpha_y, lhc.beta_y, lhc.D_y, lhc.Q_x, lhc.Q_y,
            adetuner)
        bunch_cpu = self.create_lhc_bunch(lhc)
        bunch_gpu = self.create_lhc_bunch(lhc)
        self.assertTrue(self._track_cpu_gpu(transverse_map, bunch_cpu,
            bunch_gpu), 'Transverse tracking with Adetuning CPU/GPU differs')

    def test_transverse_track_with_Cdetuning(self):
        '''
        Track the GPU bunch through a TransverseMap with chromaticity (detuning)
        Check if results on CPU and GPU are the same
        '''
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        detuner = Chromaticity(Qp_x=[5, 1], Qp_y=[7, 2])
        transverse_map = tt.TransverseMap(
            self.circumference, self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy,
            detuner)
        self.assertTrue(self._track_cpu_gpu(transverse_map, bunch_cpu,
            bunch_gpu), 'Transverse tracking with chromaticity CPU/GPU differs')
        #self.assertTrue(False, 'check')


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

    @unittest.skipUnless(has_PyCERNmachines, 'No PyCERNmachines.')
    def test_SPS_nonlinear(self):
        sps = m.SPS(n_segments=self.nsegments,
            machine_configuration='Q26-injection', Qp_x=2, Qp_y=-2.2,
            )
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        one_turn_map = sps.one_turn_map
        self.assertTrue(self._track_cpu_gpu(one_turn_map, bunch_cpu, bunch_gpu),
            'Tracking through SPS Q26 injection one turn map CPU/GPU differs.')


    @unittest.skipUnless(has_PyCERNmachines, 'No PyCERNmachines.')
    def test_PSB_linear(self):
        psb = m.PSB(n_segments=self.nsegments,
            machine_configuration='160MeV', longitudinal_focusing='linear')
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        one_turn_map = psb.one_turn_map
        self.assertTrue(self._track_cpu_gpu(one_turn_map, bunch_cpu, bunch_gpu),
            'Tracking through SPS Q26 injection one turn map CPU/GPU differs.')


    def test_bunchmonitor(self):
        ''' Test the bunchmonitor and all statistics functions
        '''
        bunchmonitor = BunchMonitor('bunchmonitor.tmp', 1000,
            write_buffer_to_file_every=512, buffer_size=4096)
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()

        self._monitor_cpu_gpu([bunchmonitor], bunch_cpu, bunch_gpu)




    def _monitor_cpu_gpu(self, monitors, bunch1, bunch2):
        '''
        Test whether monitor.dump(bunch1/bunch2) yield the same
        HDF5 file
        '''

        for m in monitors:
            m.dump(bunch2)

        with GPU(bunch1) as device:
            for m in monitors:
                m.dump(bunch1)
        self.assertTrue(False, 'Check hdf5')





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
        #print bunch1.x
        #print bunch2.x
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
        x = np.ones(self.n_macroparticles)
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
            macroparticlenumber=len(x), particlenumber_per_mp=100, charge=e,
            mass=m_p, circumference=self.circumference, gamma=self.gamma,
            coords_n_momenta_dict=coords_n_momenta_dict
        )

    def create_gaussian_bunch(self):
        return self.create_lhc_bunch()

    def create_lhc_bunch(self, lhc):
        np.random.seed(0)
        return lhc.generate_6D_Gaussian_bunch(
            n_macroparticles=self.n_macroparticles, intensity=1e11,
            epsn_x=3e-6, epsn_y=2.5e-5, sigma_z=0.11)

if __name__ == '__main__':
    unittest.main()
