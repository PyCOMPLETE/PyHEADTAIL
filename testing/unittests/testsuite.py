'''
@date:   11/03/2015
@author: Stefan Hegglin
This script runs all the unit tests specified in test_list
'''

import sys
import unittest

from test_slicing import TestSlicing
from test_particles import TestParticles
from test_generators import TestParticleGenerators
from test_aperture import TestAperture
from test_simple_long_tracking import TestSimpleLongTracking
from test_transverse_tracking import TestTransverseTracking
from test_listproxy import TestListProxy
from test_detuners import TestDetuner
from test_itest_autorun import TestAutoRun

#add your test classes here
test_list = [TestSlicing,
             TestParticles,
             TestParticleGenerators,
             TestAperture,
             TestSimpleLongTracking,
             TestListProxy,
             TestDetuner,
             TestTransverseTracking,
             TestAutoRun]


if __name__ == '__main__':
    test_load = unittest.TestLoader()
    case_list = []
    for case in test_list:
        test = test_load.loadTestsFromTestCase(case)
        case_list.append(test)
    test_suite = unittest.TestSuite(case_list)
    print('Running unit tests ' + str(test_list))
    runner = unittest.TextTestRunner()
    ret = not runner.run(test_suite).wasSuccessful()
    sys.exit(ret)
