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

import PyHEADTAIL.general.pmath as pm

# try to import pycuda, if not available --> skip this test file
try:
    import pycuda.autoinit
    import pycuda.gpuarray
except ImportError:
    has_pycuda = False
else:
    has_pycuda = True

class TestDispatch(unittest.TestCase):
    '''Test Class for the function dispatch functionality in general.pmath'''
    def setUp(self):
        self.available_CPU = pm._CPU_numpy_func_dict.keys()
        self.available_GPU = pm._GPU_func_dict.keys()

    def test_set_CPU(self):
        pm.update_active_dict(pm._CPU_numpy_func_dict)
        self.assertTrue(
            set(self.available_CPU).issubset(set(pm.__dict__.keys())),
            'Setting the active dict to CPU fails. Not all CPU functions ' +
            'were spilled to pm.globals()'
            )

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
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions. Only single param funnctions.
        Use a large sample size to account for std/mean fluctuations due to
        different algorithms (single pass/shifted/...)
        '''
        multi_param_fn = ['emittance', 'apply_permutation',]
        np.random.seed(0)
        parameter_cpu = np.random.normal(loc=1., scale=1., size=100000)
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
        No complete tracking, only bare functions. Only single param funnctions.
        Use a large number of samples (~500k). The CPU and GPU computations
        are not exactly the same due to differences in the algorithms (i.e.
        biased/unbiased estimator)
        '''
        fname = 'emittance'
        np.random.seed(0)
        parameter_cpu_1 = np.random.normal(loc=1., scale=.1, size=500000)
        parameter_cpu_2 = np.random.normal(loc=1., scale=1., size=500000)
        parameter_cpu_3 = np.random.normal(loc=1., scale=1., size=500000)
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

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_apply_permutation_computation(self):
        '''
        apply_permutation only, requires a special function call.
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions. Only single param funnctions.
        Use a large number of samples (~500k). The CPU and GPU computations
        are not exactly the same due to differences in the algorithms (i.e.
        biased/unbiased estimator)
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



    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
