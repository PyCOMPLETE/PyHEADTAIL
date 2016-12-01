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
import copy
# try to import pycuda, if not available --> skip this test file
try:
    import pycuda.autoinit
    import pycuda.gpuarray
except ImportError:
    has_pycuda = False
else:
    has_pycuda = True

import PyHEADTAIL.general.pmath as pm
import PyHEADTAIL.general.contextmanager #patches the GPUArray
from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.particles.slicing import UniformBinSlicer



class TestDispatch(unittest.TestCase):
    '''Test Class for the function dispatch functionality in general.pmath'''
    def setUp(self):
        self.available_CPU = pm._CPU_numpy_func_dict.keys()
        if has_pycuda:
            self.available_GPU = pm._GPU_func_dict.keys()

    def test_set_CPU(self):
        pm.update_active_dict(pm._CPU_numpy_func_dict)
        self.assertTrue(
            set(self.available_CPU).issubset(set(pm.__dict__.keys())),
            'Setting the active dict to CPU fails. Not all CPU functions ' +
            'were spilled to pm.globals()'
            )

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_set_GPU(self):
        pm.update_active_dict(pm._GPU_func_dict)
        self.assertTrue(
            set(self.available_GPU).issubset(set(pm.__dict__.keys())),
            'Setting the active dict to GPU fails. Not all GPU functions ' +
            'were spilled to pm.globals()'
            )
        self.assertFalse(
            set(self.available_CPU).issubset(set(pm.__dict__.keys())),
            'Setting the active dict to GPU fails. Not all CPU functions ' +
            'were deleted from pm.globals() when switching to GPU.'
            )

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_equivalency_CPU_GPU_functions(self):
        '''
        Check that CPU/GPU functions yield the same result (if both
        exist)

        No complete tracking, only bare functions. Only single param
        functions. Use a large sample size to account for std/mean
        fluctuations due to different algorithms
        (single pass/shifted/...)
        '''
        multi_param_fn = [
            'emittance', 'apply_permutation', 'mean_per_slice',
            'std_per_slice', 'emittance_per_slice', 'particles_within_cuts',
            'particles_outside_cuts',
            'macroparticles_per_slice', 'take', 'convolve', 'arange', 'zeros',
            'seq', 'init_bunch_buffer', 'init_slice_buffer', 'device',
            'searchsortedright', 'searchsortedleft', 'cumsum', 'ones', 'wofz',
            'allclose', 'empty', 'empty_like', 'put', 'atleast_1d', 'sincos',
        ]
        np.random.seed(0)
        parameter_cpu = np.random.uniform(low=1e-15, high=1., size=100000)
        parameter_gpu = pycuda.gpuarray.to_gpu(parameter_cpu)
        common_functions = [fn for fn in self.available_CPU
                            if fn in self.available_GPU]
        for fname in common_functions:
            if fname not in multi_param_fn:
                res_cpu = pm._CPU_numpy_func_dict[fname](parameter_cpu)
                res_gpu = pm._GPU_func_dict[fname](parameter_gpu)
                if isinstance(res_gpu, pycuda.gpuarray.GPUArray):
                    res_gpu = res_gpu.get()
                self.assertTrue(np.allclose(res_cpu, res_gpu),
                    'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_emittance_computation(self):
        '''
        Emittance computation only, requires a special funcition call.
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        Use a large number of samples (~500k). The CPU and GPU computations
        are not exactly the same due to differences in the algorithms (i.e.
        biased/unbiased estimator)
        '''
        fname = 'emittance'
        np.random.seed(0)
        parameter_cpu_1 = np.random.normal(loc=1., scale=.1, size=500000)
        parameter_cpu_2 = np.random.normal(loc=1., scale=1., size=500000)
        parameter_cpu_3 = parameter_cpu_2 + 0.001 *np.arange(500000)#np.random.normal(loc=1., scale=1., size=500000)
        parameter_gpu_1 = pycuda.gpuarray.to_gpu(parameter_cpu_1)
        parameter_gpu_2 = pycuda.gpuarray.to_gpu(parameter_cpu_2)
        parameter_gpu_3 = pycuda.gpuarray.to_gpu(parameter_cpu_3)
        params_cpu = [parameter_cpu_1, parameter_cpu_2, parameter_cpu_3]
        params_gpu = [parameter_gpu_1, parameter_gpu_2, parameter_gpu_3]
        res_cpu = pm._CPU_numpy_func_dict[fname](*params_cpu)
        res_gpu = pm._GPU_func_dict[fname](*params_gpu)
        if isinstance(res_gpu, pycuda.gpuarray.GPUArray):
            res_gpu = res_gpu.get()
        self.assertTrue(np.allclose(res_cpu, res_gpu),
            'CPU/GPU version of ' + fname + ' dont yield the same result')
        # check the dp=None case
        params_cpu[2] = None
        params_gpu[2] = None
        res_cpu = pm._CPU_numpy_func_dict[fname](*params_cpu)
        res_gpu = pm._GPU_func_dict[fname](*params_gpu)
        if isinstance(res_gpu, pycuda.gpuarray.GPUArray):
            res_gpu = res_gpu.get()
        self.assertTrue(np.allclose(res_cpu, res_gpu),
            'CPU/GPU version of ' + fname + ' dp=None dont yield the same result')


    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_apply_permutation_computation(self):
        '''
        apply_permutation only, requires a special function call.
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = 'apply_permutation'
        np.random.seed(0)
        n = 10
        parameter_cpu_tosort = np.random.normal(loc=1., scale=.1, size=n)
        parameter_gpu_tosort = pycuda.gpuarray.to_gpu(parameter_cpu_tosort)
        parameter_cpu_perm = np.array(np.random.permutation(n), dtype=np.int32)
        parameter_gpu_perm = pycuda.gpuarray.to_gpu(parameter_cpu_perm)
        params_cpu = [parameter_cpu_tosort, parameter_cpu_perm]
        params_gpu = [parameter_gpu_tosort, parameter_gpu_perm]
        res_cpu = pm._CPU_numpy_func_dict[fname](*params_cpu)
        res_gpu = pm._GPU_func_dict[fname](*params_gpu)
        self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
            'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_per_slice_stats(self):
        '''
        All per_slice functions (mean, cov, ?emittance)
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fnames = ['mean_per_slice', 'std_per_slice']
        np.random.seed(0)
        n = 99999
        b = self.create_gaussian_bunch(n)
        b.sort_for('z')
        slicer = UniformBinSlicer(n_slices=777, n_sigma_z=None)
        s_set = b.get_slices(slicer)
        z_cpu = b.z.copy()
        z_gpu = pycuda.gpuarray.to_gpu(z_cpu)
        sliceset_cpu = s_set
        sliceset_gpu = copy.deepcopy(s_set)
        sliceset_gpu.slice_index_of_particle = pycuda.gpuarray.to_gpu(
            s_set.slice_index_of_particle
        )
        for fname in fnames:
            res_cpu = pm._CPU_numpy_func_dict[fname](sliceset_cpu, z_cpu)
            res_gpu = pm._GPU_func_dict[fname](sliceset_gpu,z_gpu)
            self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
                'CPU/GPU version of ' + fname + ' dont yield the same result')
        fnames = ['emittance_per_slice']
        v_cpu = b.x
        v_gpu = pycuda.gpuarray.to_gpu(v_cpu)
        dp_cpu = z_cpu + np.arange(n)/n
        dp_gpu = pycuda.gpuarray.to_gpu(dp_cpu)
        for fname in fnames:
            res_cpu = pm._CPU_numpy_func_dict[fname](sliceset_cpu, z_cpu, v_cpu, dp_cpu)
            res_gpu = pm._GPU_func_dict[fname](sliceset_gpu, z_gpu, v_gpu, dp_gpu)
            # only check things which aren't nan/None. Ignore RuntimeWarning!
            with np.errstate(invalid='ignore'):
                res_cpu = res_cpu[res_cpu>1e-10]
                res_gpu = res_gpu.get()[res_gpu.get()>1e-10]
            self.assertTrue(np.allclose(res_cpu, res_gpu),
                'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_sliceset_computations(self):
        '''
        macroparticles per slice, particles_within_cuts
        require a sliceset as a parameter
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = ['particles_within_cuts', 'macroparticles_per_slice']
        pm.update_active_dict(pm._CPU_numpy_func_dict)
        np.random.seed(0)
        n = 999
        b = self.create_gaussian_bunch(n)
        b.sort_for('z')
        slicer = UniformBinSlicer(n_slices=20, n_sigma_z=2)
        s_set = b.get_slices(slicer)
        z_cpu = b.z.copy()
        z_gpu = pycuda.gpuarray.to_gpu(z_cpu)
        sliceset_cpu = s_set
        sliceset_gpu = copy.deepcopy(s_set)
        sliceset_gpu.slice_index_of_particle = pycuda.gpuarray.to_gpu(
            s_set.slice_index_of_particle
        )
        params_cpu = [sliceset_cpu]
        params_gpu = [sliceset_gpu]
        for f in fname:
            res_cpu = pm._CPU_numpy_func_dict[f](*params_cpu)
            res_gpu = pm._GPU_func_dict[f](*params_gpu)
            self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
                'CPU/GPU version of ' + f + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_take(self):
        '''
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = 'take'
        N = 1000
        arr_cpu = self.create_gaussian_bunch(n_macroparticles=N).x
        arr_gpu = pycuda.gpuarray.to_gpu(arr_cpu)
        idx = np.array(np.random.permutation(N//2), dtype=np.int32)
        idx[0] = 1
        idx[1] = 1 # test multiple times the same entrygpu
        idx_gpu = pycuda.gpuarray.to_gpu(idx)
        params_cpu = [arr_cpu, idx]
        params_gpu = [arr_gpu, idx_gpu]
        res_cpu = pm._CPU_numpy_func_dict[fname](*params_cpu)
        res_gpu = pm._GPU_func_dict[fname](*params_gpu)
        self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
            'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_convolve(self):
        '''
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = 'convolve'
        N = 1000
        a_cpu = self.create_gaussian_bunch(n_macroparticles=N).x
        v_cpu = self.create_gaussian_bunch(n_macroparticles=2*N).x
        a_gpu = pycuda.gpuarray.to_gpu(a_cpu)
        v_gpu = pycuda.gpuarray.to_gpu(v_cpu)
        mode = 'valid'
        params_cpu = [a_cpu, v_cpu, mode]
        params_gpu = [a_gpu, v_gpu, mode]
        res_cpu = pm._CPU_numpy_func_dict[fname](*params_cpu)
        res_gpu = pm._GPU_func_dict[fname](*params_gpu)
        self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
            'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_arange(self):
        '''
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = 'arange'
        start = 1.12
        stop = 4.124
        step = (stop-start)/(100.+1e-10)
        import math
        nslices = int(math.ceil((stop-start)/step))
        dtype = np.float64
        res_cpu = pm._CPU_numpy_func_dict[fname](start, stop, step, nslices, dtype=np.float64)
        res_gpu = pm._GPU_func_dict[fname](start, stop, step, nslices, dtype=np.float64)
        self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
            'CPU/GPU version of ' + fname + ' dont yield the same result')
        # same exercise with start, stop on GPU
        start_gpu = pycuda.gpuarray.empty(1, dtype=np.float64)
        stop_gpu = pycuda.gpuarray.empty_like(start_gpu)
        step_gpu = pycuda.gpuarray.empty_like(start_gpu)
        start_gpu.fill(1.12)
        stop_gpu.fill(4.124)
        step_gpu.fill(step)
        res_gpu2 = pm._GPU_func_dict[fname](start_gpu, stop_gpu, step_gpu, nslices, dtype=np.float64)
        self.assertTrue(np.allclose(res_cpu, res_gpu2.get()),
            'CPU/GPU version of ' + fname + ' with start/stop on GPU' +
            'dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_zeros(self):
        '''
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = 'zeros'
        N = 99
        dtype = np.float64
        res_cpu = pm._CPU_numpy_func_dict[fname](N, dtype=np.float64)
        res_gpu = pm._GPU_func_dict[fname](N, dtype=np.float64)
        self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
            'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_binaryop_patch(self):
        N = 10
        x_cpu = np.random.normal(0., 1., N)
        x = pycuda.gpuarray.to_gpu(x_cpu)
        y_cpu = np.array([0.3], dtype=np.float64)
        y = pycuda.gpuarray.to_gpu(y_cpu)
        z = (x-y).get()
        self.assertTrue(np.allclose(z, x_cpu - y_cpu),
            'Patching of the __sub__ method does not work correctly')
        z = (x/y).get()
        self.assertTrue(np.allclose(z, x_cpu / y_cpu),
            'Patching of the __div__ method does not work correctly')

    def create_all1_bunch(self, n_macroparticles):
        np.random.seed(1)
        x = np.ones(n_macroparticles)
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
            mass=m_p, circumference=100, gamma=10,
            coords_n_momenta_dict=coords_n_momenta_dict
        )

    def create_gaussian_bunch(self, n_macroparticles):
        P = self.create_all1_bunch(n_macroparticles)
        P.x = np.random.randn(n_macroparticles)
        P.y = np.random.randn(n_macroparticles)
        P.z = np.random.randn(n_macroparticles)
        P.xp = np.random.randn(n_macroparticles)
        P.yp = np.random.randn(n_macroparticles)
        P.dp = np.random.randn(n_macroparticles)
        return P

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
