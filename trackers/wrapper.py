from __future__ import division

from . import Element

import numpy as np

class LongWrapper(Element):
    '''Wrap particles that go out of the z range covering the circumference.'''
    def __init__(self, circumference, z0=0, *args, **kwargs):
        '''Arguments:
            - circumference: the interval length in z in [m]
            - z0: the central value of z
            Particles outside of the interval
            [z0 - circumference / 2, z0 + circumference / 2]
            will be folded back into the interval.
        '''
        self.circumference = circumference
        self.z_min = z0 - circumference / 2
        self.z_max = z0 + circumference / 2

    def track(self, beam):
        beam.z -= (((beam.z - self.z_min) // self.circumference) *
                   self.circumference)

    def track_numpy(self, beam):
        '''Explicitly uses numpy functions on the beam.'''
        too_low = np.where(beam.z < self.z_min)[0]
        while len(too_low) > 0:
            beam.z[too_low] += self.circumference
            too_low = too_low[beam.z[too_low] < self.z_min]

        too_high = np.where(beam.z > self.z_max)[0]
        while len(too_high) > 0:
            beam.z[too_high] -= self.circumference
            too_high = too_high[beam.z[too_high] > self.z_max]
