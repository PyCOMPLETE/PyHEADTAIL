from __future__ import division
'''
Created on 16.03.2014

@author: Hannes Bartosik
'''


import numpy as np


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
      

class Longitudinal_aperture(Aperture):
    '''
    classdocs
    '''
    def __init__(self, limit_dz):
        '''
        Constructor
        '''
        self.limit_dz_neg = - limit_dz[0]
        self.limit_dz_pos = + limit_dz[-1]

    def lost_particles(self, bunch):        
        return (bunch.dz >= self.limit_dz_pos) | (bunch.dz <= self.limit_dz_neg)
