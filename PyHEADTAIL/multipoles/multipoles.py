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

from . import Element

from math import factorial


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


class ThinSkewQuadrupole(Element):
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

class ThinMultipole(Element):
    '''Implements the Horner scheme to efficiently calculate the
    polynomials for any order multipole maps.
    '''
    def __init__(self, knl, ksl=[], *args, **kwargs):
        '''MAD style counting of of normal and skew multipole strengths:
        [dipolar, quadrupolar, sextupolar, octupolar, ...] components.
        Arguments:
            - knl: list of normalised normal strengths times the length
                   of the multipole magnet [1/m^order] in ascending
                   order
        Optional arguments:
            - ksl: list of normalised skew strengths times the length
                   of the multipole magnet [1/m^order] in ascending
                   order
        N.B.: If knl and ksl have different lengths, zeros are appended
        until they are equally long.
        '''
        newlen = max(len(knl), len(ksl))
        knl = list(knl) + [0] * (newlen - len(knl))
        ksl = list(ksl) + [0] * (newlen - len(ksl))
        self.knl = knl
        self.ksl = ksl

    def track(self, beam):
        dxp, dyp = self.ctaylor(beam.x, beam.y, self.knl, self.ksl)
        beam.xp -= dxp
        beam.yp += dyp

    @staticmethod
    def ctaylor(x, y, kn, ks):
        '''Efficient Horner scheme.'''
        dpx = kn[-1]
        dpy = ks[-1]
        nn = range(1, len(kn) + 1)
        for n, kkn, kks in zip(nn, kn, ks)[-2::-1]:
            dpxi = (dpx*x - dpy*y) / float(n)
            dpyi = (dpx*y + dpy*x) / float(n)
            dpx = kkn + dpxi
            dpy = kks + dpyi
        return dpx, dpy

    @staticmethod
    def ztaylor(x, y, kn, ks):
        '''Same as ctaylor but using complex numbers, slower but more
        readable -- added for the sake of clarity.
        '''
        z = (x + 1j*y)
        res = 0
        for n, (kkn, kks) in enumerate(zip(kn, ks)):
            res += (kkn + 1j*kks) * z**n / factorial(n)
        return res.real, res.imag
