'''
@author Kevin Li
@date 20/06/2014
@copyright CERN
'''
from __future__ import division


import numpy as np
from scipy.special import k0
from scipy.constants import c, e


class TransverseDamper(object):

    def __init__(self, dampingrate_x, dampingrate_y):

        if dampingrate_x and not dampingrate_y:
            self.gain_x = 2/dampingrate_x
            self.track = self.track_horizontal
            print('Damper in V active')
        elif not dampingrate_x and dampingrate_y:
            self.gain_y = 2/dampingrate_y
            self.track = self.track_vertical
            print('Damper in Y active')
        elif not dampingrate_x and not dampingrate_y:
            print('Dampers not active')
            self.track = lambda x: 0
        else:
            self.gain_x = 2/dampingrate_x
            self.gain_y = 2/dampingrate_y
            self.track = self.track_all
            print('Dampers active')

    def track_horizontal(self, beam):
        beam.xp -= self.gain_x * beam.mean_xp()

    def track_vertical(self, beam):
        beam.yp -= self.gain_y * beam.mean_yp()

    def track_all(self, beam):
        beam.xp -= self.gain_x * beam.mean_xp()
        beam.yp -= self.gain_y * beam.mean_yp()

    @classmethod
    def horizontal(cls, dampingrate_x):
        return cls(dampingrate_x, 0)

    @classmethod
    def vertical(cls, dampingrate_y):
        return cls(0, dampingrate_y)
