'''
@date:   30/09/2015
@author: Stefan Hegglin
'''
from __future__ import division

import sys, os
import os.path
BIN = os.path.dirname(__file__) # ./PyHEADTAIL/testing/unittests/
BIN = os.path.abspath( BIN ) # absolute path to unittests
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/testing/
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/
BIN = os.path.dirname( BIN ) # ../ -->  ./
sys.path.append(BIN)

import unittest
import h5py as hp
import numpy as np
from scipy.constants import c, e, m_p


# try to import pycuda, if not available --> skip this test file
try:
    import pycuda.autoinit
except ImportError:
    has_pycuda = False
else:
    has_pycuda = True

from PyHEADTAIL.general.printers import SilentPrinter
from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.general.printers import AccumulatorPrinter
from PyHEADTAIL.general.contextmanager import GPU
import PyHEADTAIL.trackers.transverse_tracking as tt
import PyHEADTAIL.trackers.longitudinal_tracking as lt
from PyHEADTAIL.trackers.detuners import AmplitudeDetuning, Chromaticity
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeField, WakeTable
from PyHEADTAIL.impedances.wakes import CircularResonator, ParallelPlatesResonator
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.feedback.widebandfeedback import Pickup, Kicker
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor
from PyHEADTAIL.rfq.rfq import RFQLongitudinalKick, RFQTransverseKick
from PyHEADTAIL.rfq.rfq import RFQTransverseDetuner
import PyHEADTAIL.general.decorators as decorators



try:
    import PyCERNmachines.CERNmachines as m
    # for replacing the cython versions in machines
    import PyCERNmachines.machines as mach
    from PyHEADTAIL.trackers.transverse_tracking import TransverseMap as TransMapPy
    from PyHEADTAIL.trackers.detuners import Chromaticity as ChromaPy
    from PyHEADTAIL.trackers.detuners import AmplitudeDetuning as AmpPy
    mach.TransverseMap = TransMapPy
    mach.Chromaticity = ChromaPy
    mach.AmplitudeDetuning = AmpPy
except ImportError:
    has_PyCERNmachines= False
else:
    has_PyCERNmachines = True


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
        self.Q_s = 0.1

        self.gamma = 15 #lhc=7000, sps=27
        self.h1 = 1
        self.h2 = 2
        self.V1 = 8e3
        self.V2 = 0
        self.dphi1 = 0
        self.dphi2 = np.pi

        self.n_macroparticles = 10000


    def tearDown(self):
        try:
            os.remove(self.monitor_fn+'1' + '.h5')
            os.remove(self.monitor_fn+'2' + '.h5')
        except:
            pass

    @decorators.synchronize_gpu_streams_before
    def test_gpu_sync_decorators(self):
        '''Test the sync functionality with the decorators'''
        # not a real test, only calls the decorator...
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
        transverse_map = tt.TransverseMap(self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy,
            printer=SilentPrinter())
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
                    alpha_x=1.2*np.ones(self.nsegments), D_x=Dx, printer=SilentPrinter())
        # create pure Python map, PyCERNmachine uses Cython.
        adetuner = AmplitudeDetuning(lhc.app_x, lhc.app_y, lhc.app_xy)
        transverse_map = tt.TransverseMap(lhc.s, lhc.alpha_x, lhc.beta_x,
            lhc.D_x, lhc.alpha_y, lhc.beta_y, lhc.D_y, lhc.Q_x, lhc.Q_y,
            adetuner, printer=SilentPrinter())
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
        transverse_map = tt.TransverseMap(self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy,
            detuner, printer=SilentPrinter())
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
            self.Q_s)
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
                D_x=self.Dx[0], D_y=self.Dy[0], charge=e, mass=m_p
            )
        self.assertTrue(self._track_cpu_gpu([longitudinal_map], bunch_cpu,
            bunch_gpu), 'Longitudinal tracking RFSystems CPU/GPU differs')

    @unittest.skipUnless(has_PyCERNmachines, 'No PyCERNmachines.')
    def test_wakefield_platesresonator(self):
        '''
        Track through a ParallelPlatesResonator wakefield
        '''
        Dx = np.append(np.linspace(0., 20., self.nsegments),[0])
        # add some dispersion/alpha
        lhc = m.LHC(n_segments=self.nsegments, machine_configuration='450GeV',
                    app_x=1e-9, app_y=2e-9, app_xy=-1.5e-11,
                    chromaticity_on=False, amplitude_detuning_on=True,
                    alpha_x=1.2*np.ones(self.nsegments), D_x=Dx,
                    printer=SilentPrinter())


        self.n_macroparticles = 200000
        bunch_cpu = self.create_lhc_bunch(lhc)#self.create_gaussian_bunch()
        bunch_gpu = self.create_lhc_bunch(lhc)#self.create_gaussian_bunch()
        n_slices=50#5
        frequency = 8e8#1e9
        R_shunt = 23e3 # [Ohm]
        Q = 1.
        unif_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=1)
        #res = CircularResonator(R_shunt=R_shunt, frequency=frequency, Q=Q)
        res = ParallelPlatesResonator(R_shunt=R_shunt, frequency=frequency, Q=Q,
                                      printer=SilentPrinter())
        wake_field = WakeField(unif_bin_slicer, res)
        self.assertTrue(self._track_cpu_gpu([wake_field], bunch_cpu, bunch_gpu),
            'Tracking Wakefield CircularResonator CPU/GPU differs')

    @unittest.skipUnless(os.path.isfile(
        './autoruntests/wake_table.dat'),
        #wakeforhdtl_PyZbase_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat'),
        'Wakefile not found')
    @unittest.skipUnless(has_PyCERNmachines, 'No PyCERNmachines.')
    def test_wakefield_wakefile(self):
        '''
        Track an LHC bunch and a LHC wakefield
        '''
        wakefile = 'autoruntests/wake_table.dat'#'./wakeforhdtl_PyZbase_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat'
        Qp_x, Qp_y = 1., 1.
        Q_s = 0.0049
        n_macroparticles = 10
        intensity = 1e11
        longitudinal_focusing = 'linear'
        machine = m.LHC(n_segments=1, machine_configuration='450GeV',
                  longitudinal_focusing=longitudinal_focusing,
                  Qp_x=[Qp_x], Qp_y=[Qp_y], Q_s=Q_s,
                  beta_x=[65.9756], beta_y=[71.5255], printer=SilentPrinter())
        epsn_x  = 3.5e-6
        epsn_y  = 3.5e-6
        sigma_z = 1.56e-9*c / 4.
        np.random.seed(0)
        bunch_cpu = machine.generate_6D_Gaussian_bunch(
            n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)
        np.random.seed(0)
        bunch_gpu = machine.generate_6D_Gaussian_bunch(
            n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)
        n_slices_wakefields = 55
        n_sigma_z_wakefields = 3
        slicer_for_wakefields_cpu   = UniformBinSlicer(
            n_slices_wakefields, n_sigma_z=n_sigma_z_wakefields)
        wake_components = [ 'time', 'dipole_x', 'dipole_y',
                        'no_quadrupole_x', 'no_quadrupole_y',
                        'no_dipole_xy', 'no_dipole_yx' ]
        wake_table_cpu      = WakeTable(wakefile, wake_components,
                                        printer=SilentPrinter())
        wake_field_cpu      = WakeField(slicer_for_wakefields_cpu, wake_table_cpu)
        # also checked for 100 turns!
        self.assertTrue(self._track_cpu_gpu([wake_field_cpu], bunch_cpu, bunch_gpu, nturns=2),
            'Tracking through WakeField(waketable) differs')

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
        sps = m.SPS(n_segments=self.nsegments, printer=SilentPrinter(),
            machine_configuration='Q26-injection', Qp_x=2, Qp_y=-2.2,
            charge=e, mass=m_p
            )
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        one_turn_map = sps.one_turn_map
        self.assertTrue(self._track_cpu_gpu(one_turn_map, bunch_cpu, bunch_gpu),
            'Tracking through SPS Q26 injection one turn map CPU/GPU differs.')


    @unittest.skipUnless(has_PyCERNmachines, 'No PyCERNmachines.')
    def test_PSB_linear(self):
        psb = m.PSB(n_segments=self.nsegments, printer=SilentPrinter(),
            machine_configuration='160MeV', longitudinal_focusing='linear')
        bunch_cpu = self.create_all1_bunch()
        bunch_gpu = self.create_all1_bunch()
        one_turn_map = psb.one_turn_map
        self.assertTrue(self._track_cpu_gpu(one_turn_map, bunch_cpu, bunch_gpu),
            'Tracking through SPS Q26 injection one turn map CPU/GPU differs.')


    def test_bunchmonitor(self):
        ''' Test the bunchmonitor and all statistics functions
        '''
        self.n_macroparticles = 100 # use a high number to get acc statistics
        n_steps = 5
        self.monitor_fn = 'monitor'
        bunchmonitor1 = BunchMonitor(self.monitor_fn +'1', n_steps=n_steps,
            write_buffer_every=2, buffer_size=3)
        bunchmonitor2 = BunchMonitor(self.monitor_fn +'2', n_steps=n_steps,
            write_buffer_every=2, buffer_size=3)
        bunch_cpu = self.create_gaussian_bunch()
        bunch_gpu = self.create_gaussian_bunch()
        self._monitor_cpu_gpu(bunchmonitor1, bunchmonitor2, bunch_cpu, bunch_gpu)

    def test_slicemonitor(self):
        '''Test the slicemonitor, especially the statistics per slice functions
        '''
        self.monitor_fn = 'monitor'
        self.n_macroparticles = 1000
        nslices = 3
        n_sigma_z = 1
        n_steps = 5
        slicer = UniformBinSlicer(nslices, n_sigma_z)
        slicemonitor_1 = SliceMonitor(self.monitor_fn +'1', n_steps, slicer,
            write_buffer_every=2, buffer_size=3)
        slicemonitor_2 = SliceMonitor(self.monitor_fn +'2', n_steps, slicer,
            write_buffer_every=2, buffer_size=3)
        bunch_cpu = self.create_gaussian_bunch()
        bunch_gpu = self.create_gaussian_bunch()
        self._monitor_cpu_gpu(slicemonitor_1, slicemonitor_2, bunch_cpu, bunch_gpu)

    def test_RFQ_Kick(self):
        '''
        Test the RFQ tracking element in rfq/rfq_python
        '''
        bunch_cpu = self.create_gaussian_bunch()
        bunch_gpu = self.create_gaussian_bunch()
        rfq_transverse = RFQTransverseKick(v_2=2e9, omega=800e6*2.*np.pi,
            phi_0=0.
        )
        rfq_longitudinal = RFQLongitudinalKick(v_2=2e9, omega=100e6*2.*np.pi,
            phi_0=np.pi/2
        )
        self.assertTrue(self._track_cpu_gpu([rfq_transverse, rfq_longitudinal],
            bunch_cpu, bunch_gpu), 'Tracking through an RFQ differs on CPU/GPU'
        )

    def test_RFQ_detuner(self):
        '''
        Test the RFQ as a detuning element in the transverse map
        '''
        bunch_cpu = self.create_gaussian_bunch()
        bunch_gpu = self.create_gaussian_bunch()
        detuner = RFQTransverseDetuner(v_2=1.5e9, omega=341e5*2.5*np.pi,
            phi_0=-np.pi/4, beta_x_RFQ=200., beta_y_RFQ=99.)
        transverse_map = tt.TransverseMap(self.s, self.alpha_x, self.beta_x,
            self.Dx, self.alpha_y, self.beta_y, self.Dy, self.Qx, self.Qy,
            detuner, printer=SilentPrinter())
        self.assertTrue(self._track_cpu_gpu(transverse_map, bunch_cpu,
            bunch_gpu), 'Transverse tracking with RFQDetuner CPU/GPU differs')

    def _monitor_cpu_gpu(self, monitor1, monitor2, bunch1, bunch2):
        '''
        Test whether monitor.dump(bunch1/bunch2) works. Read the resulting
        h5 files and compare (some) results
        '''
        params_to_check = ['mean_x', 'sigma_dp', 'macroparticlenumber',
            'n_macroparticles_per_slice']
        for i in xrange(monitor2.n_steps):
            bunch2.x += 1
            bunch2.xp -= 0.1
            bunch2.dp *= 0.97
            monitor2.dump(bunch2)
        res1 = self.read_h5_file(monitor2.filename + '.h5', params_to_check)

        with GPU(bunch1) as device:
            for i in xrange(monitor1.n_steps):
                bunch1.x += 1
                bunch1.xp -= 0.1
                bunch1.dp *= 0.97
                monitor1.dump(bunch1)
        res2 = self.read_h5_file(monitor1.filename + '.h5', params_to_check)

        for i in xrange(len(res1)):
            self.assertTrue(np.allclose(res1[i],res2[i]),
                msg='.h5 file generated by monitor of CPU/GPU differ' +
                    str(params_to_check))

    def _track_cpu_gpu(self, list_of_maps, bunch1, bunch2, nturns=1):
        '''
        Tracks both bunches through the list of maps (once GPU, once CPU)
        and checks whether it yields the same result. Returns True/False.
        Make sure bunch1, bunch2 are two identical objects (not one!)
        Change the actual implementation of the GPU interface/strategy here
        '''
        # GPU
        with GPU(bunch1) as device:
            for n in xrange(nturns):
                for m in list_of_maps:
                    m.track(bunch1)

        # CPU
        for n in xrange(nturns):
            for m in list_of_maps:
                m.track(bunch2)
        #print bunch1.x
        #print bunch2.x
        # make sure the beam is sorted according to it's id to be able to compare them
        # this is required since argsort for the slices is not unique
        # (particles within slices can be permuted, gpu/cpu might be different!)
        bunch1.sort_for('id')
        bunch2.sort_for('id')
        for att in bunch1.coords_n_momenta | set(['id']):
            if not np.allclose(getattr(bunch1, att), getattr(bunch2, att),
                               rtol=1.e-5, atol=1.e-8):
                return False
        return True

    def read_h5_file(self, filename, parameters):
        '''Returns a list of the parameters in the filename'''
        f = hp.File(filename)
        res = []
        data = f['Bunch']
        for param in parameters:
            try:
                res.append(data[param][:])
            except: #is a Slices statistic
                pass
        try: # if slicemonitor is also available
            data = f['Slices']
            for param in parameters:
                try:
                    res.append(data[param][:,:])
                except:
                    pass
        except:
            pass
        return res

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
            macroparticlenumber=len(x), particlenumber_per_mp=1000, charge=e,
            mass=m_p, circumference=self.circumference, gamma=self.gamma,
            coords_n_momenta_dict=coords_n_momenta_dict
        )

    def create_gaussian_bunch(self):
        P = self.create_all1_bunch()
        P.x = np.random.randn(self.n_macroparticles)
        P.y = np.random.randn(self.n_macroparticles)
        P.z = np.random.randn(self.n_macroparticles)
        P.xp = np.random.randn(self.n_macroparticles)
        P.yp = np.random.randn(self.n_macroparticles)
        P.dp = np.random.randn(self.n_macroparticles)
        return P

    def create_lhc_bunch(self, lhc):
        np.random.seed(0)
        return lhc.generate_6D_Gaussian_bunch(
            n_macroparticles=self.n_macroparticles, intensity=1e11,
            epsn_x=3e-6, epsn_y=2.5e-5, sigma_z=0.11)

if __name__ == '__main__':
    unittest.main()
