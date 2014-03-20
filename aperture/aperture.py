from __future__ import division
'''
Created on 16.03.2014

@author: Hannes Bartosik
'''


import numpy as np
'''http://docs.scipy.org/doc/numpy/reference/routines.html'''
#~ import cobra_functions.cobra_functions as cp


class Aperture(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def track(self, bunch):
        bunch.identity[self.lost_particles(bunch)] = 0
            

class Rectangular_aperture(Aperture):
    '''
    classdocs
    '''
    def __init__(self, limit_x, limit_y):
        '''
        Constructor
        '''
        self.limit_x = limit_x
        self.limit_y = limit_y

    def lost_particles(self, bunch):        
        return (np.abs(bunch.x) >= self.limit_x) | (np.abs(bunch.y) >= self.limit_y)
        
