'''
@author L. R. Carver, M. Schenk
@date 27/11/2015
@copyright CERN
'''
from __future__ import division

from . import Printing
import numpy as np

class SkewQuadrupole(Printing):

    def __init__(self, k_skew, *args, **kwargs):
        self.k_skew = k_skew

    def track(self, beam):
        beam.xp += self.k_skew * beam.y
        beam.yp += self.k_skew * beam.x
