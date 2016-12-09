from __future__ import division

import numpy as np
from scipy.optimize import brentq


def zero_crossings(f, x):
    """Get root of function f in intervall x"""
    y = f(x)
    zix = np.where(np.abs(np.diff(np.sign(y))) == 2)[0]

    x0 = np.array([brentq(f, x[i], x[i+1]) for i in zix])
    # y0 = np.array([f(i) for i in x0])

    return x0 #, y0 # the function values y0 should be ~0


def extrema(x, y=None):
    """Get extrema of curve x"""
    zix = np.where(np.abs(np.diff(np.sign(np.diff(x)))) == 2)[0]
    return zix
    if not y:
        print zix
