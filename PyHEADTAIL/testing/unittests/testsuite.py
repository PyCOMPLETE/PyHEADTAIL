'''
@date:   11/03/2015
@author: Stefan Hegglin
This script runs all the unit tests specified in test_list
'''

import sys
import unittest

try:
    import pycuda.autoinit
except ImportError:
    has_pycuda = False
    print 'No PyCUDA installation found.'

from pprint import pprint

from test_slicing import TestSlicing
from test_particles import TestParticles
from test_generators import TestParticleGenerators
from test_aperture import TestAperture
from test_longitudinal_tracking import TestSimpleLongTracking
from test_transverse_tracking import TestTransverseTracking
from test_listproxy import TestListProxy
from test_detuners import TestDetuner
from test_itest_autorun import TestAutoRun
from test_cobra import TestCobra
from test_gpu_interface import TestGPUInterface
from test_dispatch import TestDispatch
from test_monitor import TestMonitor

#add your test classes here
test_list = [TestSlicing,
             TestParticles,
             TestParticleGenerators,
             TestAperture,
             TestSimpleLongTracking,
             TestListProxy,
             TestDetuner,
             TestAutoRun, #uncomment to run full test suite (~ 1 min)
             TestTransverseTracking,
             TestCobra,
             TestGPUInterface,
             TestDispatch,
             TestMonitor]


if __name__ == '__main__':
    test_load = unittest.TestLoader()
    case_list = []
    for case in test_list:
        test = test_load.loadTestsFromTestCase(case)
        case_list.append(test)
    test_suite = unittest.TestSuite(case_list)
    print('Running unit tests:')
    pprint(test_list)
    runner = unittest.TextTestRunner()
    ret = not runner.run(test_suite).wasSuccessful()
    sys.exit(ret)
