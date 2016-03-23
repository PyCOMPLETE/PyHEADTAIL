'''
Collection of localised thin multipole maps.
For formulae see e.g. SIXTRACK:

SixTrack Physics Manual
R. De. Maria and M. Fjellstrom
August 18, 2015

or, likewise,

A Symplectic Six-Dimensional Thin-Lens Formalism for Tracking
G. Ripken, F. Schmidt
April 5, 1995

@authors: Adrian Oeftiger
@date:    23/03/2016
'''

from __future__ import division, print_function

from . import Element

class ThinQuadrupole(Element):
    '''Thin quadrupolar map.'''
    def __init__(self, k1l, *args, **kwargs):
        '''Arguments:
            - k1l: normalised strength times the length of the
                   quadrupole magnet [1/m]
        '''
        self.kL = k1l

    def track(self, beam):
        beam.xp -= self.kL * beam.x
        beam.yp += self.kL * beam.y


class SkewThinQuadrupole(Element):
    '''Thin skew quadrupolar map.'''
    def __init__(self, k1sl, *args, **kwargs):
        '''Arguments:
            - k1sl: normalised strength times the length of the
                    skew quadrupole magnet [1/m]
        '''
        self.kL = k1sl

    def track(self, beam):
        beam.xp += self.kL * beam.y
        beam.yp += self.kL * beam.x


class ThinSextupole(Element):
    '''Thin sextupolar map.'''
    def __init__(self, k2l, *args, **kwargs):
        '''Arguments:
            - k2l: normalised strength times the length of the
                   sextupole magnet [1/m^2]
        '''
        self.kL = k2l

    def track(self, beam):
        beam.xp -= 0.5 * self.kL * (beam.x*beam.x - beam.y*beam.y)
        beam.yp += self.kL * beam.x * beam.y


class ThinOctupole(Element):
    '''Thin octupolar map.'''
    def __init__(self, k3l, *args, **kwargs):
        '''Arguments:
            - k3l: normalised strength times the length of the
                   octupole magnet [1/m^3]
        '''
        self.kL = k3l
        self.kL6 = k3l / 6.

    def track(self, beam):
        beam.xp -= self.kL6 * (beam.x*beam.x*beam.x - 3*beam.x*beam.y*beam.y)
        beam.yp -= self.kL6 * (beam.y*beam.y*beam.y - 3*beam.x*beam.x*beam.y)
