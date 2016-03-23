'''
Collection of localised thin multipole maps.
For formulae see e.g. SIXTRACK:

SixTrack Physics Manual
R. De. Maria and M. Fjellstrom
February 21, 2014

@authors: Adrian Oeftiger
@date:    23/03/2016
'''

from __future__ import division, print_function

from . import Element

class ThinQuadrupole(Element):
    '''Thin quadrupolar map.'''
    def __init__(self, length, k1, *args, **kwargs):
        '''Arguments:
            - length: interval in s along accelerator over which to integrate
            - k1: normalised strength [1/m] of the quadrupole magnet
        '''
        self.kL = k1 * length

    def track(self, beam):
        beam.xp -= self.kL * beam.x
        beam.yp += self.kL * beam.y

class ThinSextupole(Element):
    '''Thin sextupolar map.'''
    def __init__(self, length, k2, *args, **kwargs):
        '''Arguments:
            - length: interval in s along accelerator over which to integrate
            - k2: normalised strength [1/m^2] of the sextupole magnet
        '''
        self.kL = k2 / 2. * length

    def track(self, beam):
        beam.xp -= self.kL * (beam.x*beam.x - beam.y*beam.y)
        beam.yp += self.kL * beam.x * beam.y

class ThinOctupole(Element):
    '''Thin octupolar map.'''
    def __init__(self, length, k3, *args, **kwargs):
        '''Arguments:
            - length: interval in s along accelerator over which to integrate
            - k3: normalised strength [1/m^3] of the octupole magnet
        '''
        self.kL = k3 / 6. * length

    def track(self, beam):
        beam.xp -= self.kL * (beam.x*beam.x*beam.x - 3*beam.x*beam.y*beam.y)
        beam.yp += self.kL * (beam.y*beam.y*beam.y - 3*beam.x*beam.x*beam.y)
