'''
Copyright CERN 2014
Author: Adrian Oeftiger, oeftiger@cern.ch Adrian Oeftiger, oeftiger@cern.ch

This module provides a two-dimensional Truncated Power Series
up until first order as suited for algebraical Jacobian determination
for two given variables.

The TPS class supports elementary operations such as +, -, /, *
(and true division according to "from __future__ import division").

Functions such as sin, cos, exp, log etc are envisaged
to be implemented in a later version.
'''
from __future__ import division
import numpy as np

class TPS(object):
    '''Truncated Power Series which obeys a TPS Algebra,
    cf. "DIFFERENTIAL ALGEBRAIC DESCRIPTION OF BEAM
    DYNAMICS TO VERY HIGH ORDERS" by M. BERZ,
    Particle Accelerators, 1989. Vol. 24, pp. 109-124.'''
    def __init__(self, vector=np.array([0, 1, 0])):
        if hasattr(vector, "__len__"):
            assert len(vector) is 3
        else:
            vector = [vector, 1, 0]
        self._vector = np.array(vector)

    @classmethod
    def get_instance(cls, vector):
        return cls(vector)

    def __add__(self, other):
        '''this TPS + (other TPS or scalar)'''
        if issubclass(other.__class__, TPS):
            return self.get_instance(vector=self._vector + other._vector)
        else:
            return self.get_instance(vector=self._vector + other)

    def __radd__(self, other):
        '''(other TPS or scalar) + this TPS'''
        return self + other

    def __mul__(self, other):
        '''this TPS * (other TPS or scalar)'''
        v = self._vector
        if issubclass(other.__class__, TPS):
            w = other._vector
            return self.get_instance(vector=[     v[0] * w[0],
                                    v[0] * w[1] + v[1] * w[0],
                                    v[0] * w[2] + v[2] * w[0]
                                ] )
        else:
            return self.get_instance(vector=v * other)

    def __rmul__(self, other):
        '''(other TPS or scalar) * TPS'''
        return self * other

    def __div__(self, other):
        '''this TPS / (other TPS or scalar)'''
        if issubclass(other.__class__, TPS):
            return self * other.invert()
        else:
            return self.get_instance(vector=self._vector / other)

    def __rdiv__(self, other):
        '''(other TPS or scalar) / this TPS'''
        return other * self.invert()

    def __truediv__(self, other):
        '''this TPS / (other TPS or scalar)'''
        return self.__div__(other)

    def __rtruediv__(self, other):
        '''(other TPS or scalar) / this TPS'''
        return self.__rdiv__(other)

    def __sub__(self, other):
        '''this TPS - other TPS'''
        return self + -other

    def __rsub__(self, other):
        '''other TPS - this TPS'''
        return other + -self

    def __eq__(self, other):
        '''this TPS == other TPS or this TPS real value == other scalar.
        '''
        if issubclass(other.__class__, TPS):
        	return self._vector == other._vector
        else:
        	return self.real() == other

    def __ne__(self, other):
        '''this TPS != other TPS'''
        return not self == other

    def __neg__(self):
        '''- (this TPS)'''
        return self.get_instance(vector=-self._vector)

    def invert(self):
        '''1 / (this TPS)'''
        if self.real() == 0:
            raise ZeroDivisionError("TPS real part is zero, cannot be inverted")
        a0i = 1. / self._vector[0]
        a1 = self._vector[1]
        a2 = self._vector[2]
        return self.get_instance(vector=[a0i, -a1 * a0i**2, -a2 * a0i**2])

    @property
    def real(self):
        '''Zero-order entry, the main value'''
        return self._vector[0]

    @property
    def diff(self):
        '''First-order entries, the first differential values'''
        return self._vector[1], self._vector[2]

    def getvector(self):
        '''returns all TPS coefficients in an np.ndarray'''
        return self._vector


class TPS4(TPS):
    '''1D4, 1st order and 4 variables'''
    def __init__(self, vector=np.array([0, 1, 0, 0, 0])):
        if hasattr(vector, "__len__"):
            assert len(vector) is 5
        else:
            vector = [vector, 1, 0, 0, 0]
        self._vector = np.array(vector)

    def __mul__(self, other):
        '''this TPS * (other TPS or scalar)'''
        v = self._vector
        if issubclass(other.__class__, TPS):
            w = other._vector
            return self.get_instance(vector=[     v[0] * w[0],
                                    v[0] * w[1] + v[1] * w[0],
                                    v[0] * w[2] + v[2] * w[0],
                                    v[0] * w[3] + v[3] * w[0],
                                    v[0] * w[4] + v[4] * w[0]
                                ] )
        else:
            return self.get_instance(vector=v * other)

    def invert(self):
        '''1 / (this TPS)'''
        if self.real() == 0:
            raise ZeroDivisionError("TPS real part is zero, cannot be inverted")
        a0i = 1. / self._vector[0]
        a1 = self._vector[1]
        a2 = self._vector[2]
        a3 = self._vector[3]
        a4 = self._vector[4]
        return self.get_instance(
			vector=[a0i, -a1 * a0i**2, -a2 * a0i**2,
			             -a3 * a0i**2, -a4 * a0i**2])

    @property
    def diff(self):
        '''returns the first-order entries, the first differential values'''
        return (self._vector[1], self._vector[2],
                self._vector[3], self._vector[4])
