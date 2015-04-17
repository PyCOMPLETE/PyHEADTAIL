'''
@date:   17/03/2015
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

from PyHEADTAIL.general.utils import ListProxy


class TestListProxy(unittest.TestCase):
    xvalue = 20
    yvalue = 10
    def setUp(self):

        class MockClass(object):
            def __init__(self):
                self.x = TestListProxy.xvalue
                self.y = TestListProxy.yvalue

        self.obj1 = MockClass()
        self.obj2 = MockClass()
        self.obj3 = MockClass()
        self.obj4 = MockClass()
        objects = [self.obj1, self.obj2, self.obj3, self.obj4]
        self.proxylist_x = ListProxy(objects, 'x')

    def tearDown(self):
        pass

    def test_listproxy_getitem(self):
        '''Tests whether the ListProxy access via [] -> getitem
        works correctly
        '''
        for i in range(4):
            self.assertEqual(self.proxylist_x[i], TestListProxy.xvalue,
                             'Accessing attributes via ListProxy [] ' +
                             'incorrect')

    def test_listproxy_setitem(self):
        '''Tests whether the ListProxy setitem via [] works correctly
        eg updates the underlying object
        '''
        new_value = 4
        self.proxylist_x[0] = new_value
        for i in range(1,4):
            self.assertEqual(self.proxylist_x[i], TestListProxy.xvalue,
                             'changing one object in a ListProxy ' +
                             'invalidates other objects')
        self.assertEqual(self.proxylist_x[0], new_value,
                         'setting properties via ListProxy [] ' +
                         'does not work correctly')
        self.assertEqual(self.obj1.x, new_value, 'setting a property ' +
                         'via a ListProxy does not change the proxied ' +
                         'object')

    def test_listproxy_length(self):
        '''Tests whether len() returns the correct length of the
        ListProxy, i.e. the number of proxied objects
        '''
        self.assertEqual(len(self.proxylist_x), 4, 'ListProxy has wrong' +
                         'length when accessed via len(listproxy)')

    def test_listproxy_set_multiple_reference(self):
        '''Tests whether modifying two proxied objects behaves
        the same as modyfing two (unproxied) objects
        '''
        self.proxylist_x[0] = -10
        self.proxylist_x[1] = self.proxylist_x[0]
        self.proxylist_x[0] = -20
        self.assertEqual(self.proxylist_x[1], -10, 'Modifying a proxied' +
                         'object isn\'t independent of modifying another' +
                         ' proxied object')


if __name__ == '__main__':
    unittest.main()
