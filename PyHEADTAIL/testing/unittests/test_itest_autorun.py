'''
@date:   23/03/2015
@author: Stefan Hegglin
'''

from __future__ import division

import sys, os
BIN = os.path.dirname(__file__) # ./PyHEADTAIL/testing/unittests/
BIN = os.path.abspath( BIN ) # absolute path to unittests
sys.path.append(BIN)
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/testing/
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/
BIN = os.path.dirname( BIN ) # ../ -->  ./
sys.path.append(BIN)
sys.path.append('autoruntests/')
import unittest

import autoruntests.ApertureNLossesTest as at
import autoruntests.DetunersTest as dt
import autoruntests.MonitorTest as mt
import autoruntests.RFQTest as rt
import autoruntests.SlicingTest as st
import autoruntests.TransverseTrackingTest as ttt
import autoruntests.WakeTest as wt

from PyHEADTAIL.general.utils import ListProxy


class TestAutoRun(unittest.TestCase):
    '''Unittest which calls the functions in /autoruntests.
    These tests were converted from the interactive-tests and have
    removed any plotting and time-consuming function calls by limiting the
    number of particles/turns.
    These tests do not test any specific functionality but should simply
    run without throwing any errors. This way one can check whether existing
    code still works properly when introducing new features/interfaces.
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_aperturenlossestest(self):
        '''Runs the autoruntests/ApertureNLossesTest script and fails
        the test if an exception is thrown
        '''
        try:
            at.run()
        except Exception, err:
            self.fail('ApertureNLossesTest threw an exception:\n' + str(err))

    def test_detunerstest(self):
        '''Runs the autoruntests/DetunersTest script and fails
        the test if an exception is thrown
        '''
        try:
            dt.run()
        except Exception, err:
            self.fail('DetunersTest threw an exception:\n' + str(err))

    def test_monitortest(self):
        '''Runs the autoruntests/MonitorTest script and fails
        the test if an exception is thrown
        '''
        try:
            mt.run()
        except Exception, err:
            self.fail('MonitorTest threw an exception:\n' + str(err))

    def test_rfqtest(self):
        '''Runs the autoruntests/RFQTest script and fails the test
        if an exception is thrown
        '''
        try:
            rt.run()
        except Exception, err:
            self.fail('RFQTest threw an exception:\n' + str(err))

    def test_slicingtest(self):
        '''Runs the autoruntests/SlicingTest script and fails the test
        if an exception is thrown
        '''
        try:
            st.run()
        except Exception, err:
            self.fail('SlicingTest threw an exception:\n' + str(err))

    def test_transversetrackingtest(self):
        '''Runs the autoruntests/TransverseTrackingTest script
        and fails the test if an exception is thrown
        '''
        try:
            ttt.run()
        except Exception, err:
            self.fail('TransverseTrackingTest threw an exception:\n' +
                      str(err))

    def test_waketest(self):
        '''Runs the autoruntests/WakeTest script and fails the
        test if an exception is thrown
        '''
        try:
            wt.run()
        except Exception, err:
            self.fail('WakeTest threw an exception:\n' +
                    str(err))

if __name__ == '__main__':
    unittest.main()
