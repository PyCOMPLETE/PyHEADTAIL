'''
@authors: Vadim Gubaidulin, Adrian Oeftiger
@date:    18.02.2020
'''
from __future__ import division

from PyHEADTAIL.general.element import Element
from PyHEADTAIL.particles.slicing import clean_slices

import numpy as np
from scipy.constants import c, epsilon_0, pi, m_e, m_p, e

from scipy.interpolate import splrep, splev
from functools import wraps

from PyHEADTAIL.general import pmath as pm


def _sig_sqrt(sig_x, sig_y):
    return pm.sqrt(2 * (sig_x**2 - sig_y**2))

def _efieldn_mit(x, y, sig_x, sig_y):
    '''The charge-normalised electric field components of a
    two-dimensional Gaussian charge distribution according to
    M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

    Return (E_x / Q, E_y / Q).

    Assumes sig_x > sig_y and mean_x == 0 as well as mean_y == 0.
    For convergence reasons of the erfc, use only x > 0 and y > 0.

    Uses FADDEEVA C++ implementation from MIT (via SciPy >= 0.13.0).
    '''
    sig_sqrt = _sig_sqrt(sig_x, sig_y)
    w1re, w1im = pm.wofz(x / sig_sqrt, y / sig_sqrt)
    ex = pm.exp(-x*x / (2 * sig_x*sig_x) +
                -y*y / (2 * sig_y*sig_y))
    w2re, w2im = pm.wofz(x * sig_y/(sig_x*sig_sqrt),
                         y * sig_x/(sig_y*sig_sqrt))
    denom = 2. * epsilon_0 * np.sqrt(pi) * sig_sqrt
    return (w1im - ex * w2im) / denom, (w1re - ex * w2re) / denom

def _efieldn_mitmod(x, y, sig_x, sig_y):
    '''The charge-normalised electric field components of a
    two-dimensional Gaussian charge distribution according to
    M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

    Return (E_x / Q, E_y / Q).

    Assumes sig_x > sig_y and mean_x == 0 as well as mean_y == 0.
    For convergence reasons of the erfc, use only x > 0 and y > 0.

    Uses erfc C++ implementation from MIT (via SciPy >= 0.13.0)
    and calculates wofz (FADDEEVA function) explicitely.
    '''
    # timing was ~1.01ms for same situation as _efieldn_mit
    sig_sqrt = _sig_sqrt(sig_x, sig_y)
    w1 = pm._errfadd((x + 1j * y) / sig_sqrt)
    ex = pm.exp(-x*x / (2 * sig_x*sig_x) +
                -y*y / (2 * sig_y*sig_y))
    w2 = pm._errfadd(x * sig_y/(sig_x*sig_sqrt) +
                        y * sig_x/(sig_y*sig_sqrt) * 1j)
    val = (w1 - ex * w2) / (2 * epsilon_0 * np.sqrt(pi) * sig_sqrt)
    return val.imag, val.real

def _efieldn_koelbig(x, y, sig_x, sig_y):
    '''The charge-normalised electric field components of a
    two-dimensional Gaussian charge distribution according to
    M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

    Return (E_x / Q, E_y / Q).

    Assumes sig_x > sig_y and mean_x == 0 as well as mean_y == 0.
    For convergence reasons of the erfc, use only x > 0 and y > 0.

    Uses CERN library from K. Koelbig.
    '''
    # timing was ~3.35ms for same situation as _efieldn_mit
    if not pm._errf:
        raise ImportError('errfff cannot be imported for using ' +
                            'TransverseSpaceCharge._efield_koelbig .' +
                            'Did you call make (or f2py general/errfff.f)?')
    sig_sqrt = _sig_sqrt(sig_x, sig_y)
    w1re, w1im = pm._errf(x/sig_sqrt, y/sig_sqrt)
    ex = pm.exp(-x*x / (2 * sig_x*sig_x) +
                -y*y / (2 * sig_y*sig_y))
    w2re, w2im = pm._errf(x * sig_y/(sig_x*sig_sqrt),
                            y * sig_x/(sig_y*sig_sqrt))
    pref = 1. / (2 * epsilon_0 * np.sqrt(pi) * sig_sqrt)
    return pref * (w1im - ex * w2im), pref * (w1re - ex * w2re)

def wfun(z):
    '''FADDEEVA function as implemented in PyECLOUD, vectorised.'''
    x=z.real
    y=z.imag
    if not pm._errf:
        raise ImportError('errfff cannot be imported for using ' +
                            'TransverseSpaceCharge._efield_pyecloud .' +
                            'Did you f2py errfff.f?')
    wx,wy=pm._errf(x,y) # in PyECLOUD only pm._errf_f (not vectorised)
    return wx+1j*wy

def _efieldn_pyecloud(xin, yin, sigmax, sigmay):
    '''The charge-normalised electric field components of a
    two-dimensional Gaussian charge distribution according to
    M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

    Return (E_x / Q, E_y / Q).

    Effective copy of PyECLOUD.BassErsk.BassErsk implementation.
    '''
    x=abs(xin);
    y=abs(yin);
    eps0=8.854187817620e-12;
    if sigmax>sigmay:
        S=np.sqrt(2*(sigmax*sigmax-sigmay*sigmay));
        factBE=1/(2*eps0*np.sqrt(pi)*S);
        etaBE=sigmay/sigmax*x+1j*sigmax/sigmay*y;
        zetaBE=x+1j*y;
        val=factBE*(wfun(zetaBE/S)-
                    np.exp( -x*x/(2*sigmax*sigmax)-y*y/(2*sigmay*sigmay))*
                    wfun(etaBE/S) );
        Ex=abs(val.imag)*np.sign(xin);
        Ey=abs(val.real)*np.sign(yin);
    else:
        S=np.sqrt(2*(sigmay*sigmay-sigmax*sigmax));
        factBE=1/(2*eps0*np.sqrt(pi)*S);
        etaBE=sigmax/sigmay*y+1j*sigmay/sigmax*x;
        yetaBE=y+1j*x;
        val=factBE*(wfun(yetaBE/S)-
                    np.exp( -y*y/(2*sigmay*sigmay)-x*x/(2*sigmax*sigmax))*
                    wfun(etaBE/S) );
        Ey=abs(val.imag)*np.sign(yin);
        Ex=abs(val.real)*np.sign(xin);
    return Ex, Ey

@np.vectorize
def _efieldn_kv_a(x, y, sigma_x, sigma_y):
    '''
        Field of a KV distrubition calculated as in here: https://cds.cern.ch/record/258225/files/P00020427.pdf
    '''
    a = sigma_x*pm.sqrt(2)
    b = sigma_y*pm.sqrt(2)
    if (x/a)**2+(y/b)**2 <= 1:
        efield_x = 4.0/(a+b)*x/a
        efield_y = 4.0/(a+b)*y/b
    else:
        uxy = (x)**2-(y)**2 - (a)**2+(b)**2
        vxy = uxy**2+(2.0*x*y)**2
        efield_x = 4.0/(a**2-b**2)*(x-pm.sign(x) /
                                    pm.sqrt(2.0)*pm.sqrt(uxy+pm.sqrt(vxy)))
        uxy = (y)**2-(x)**2 - (b)**2+(a)**2
        efield_y = 4.0/(b**2-a**2)*(y - pm.sign(y) /
                                    pm.sqrt(2.0)*pm.sqrt(uxy+pm.sqrt(vxy)))
    denom = 4*np.pi*epsilon_0
    return efield_x/denom, efield_y/denom
# vectorize is bad for cuda


@np.vectorize
def _efieldn_kv_b(x, y, sigma_x, sigma_y):
    '''
        Field of a KV distrubition calculated as in here: https://cds.cern.ch/record/258225/files/P00020427.pdf
    '''
    a = sigma_x*pm.sqrt(2)
    b = sigma_y*pm.sqrt(2)
    if x == 0 and y == 0:
        return 0, 0
    elif (x/a)**2+(y/b)**2 <= 1:
        efield_x = 4.0/(a+b)*x/a
        efield_y = 4.0/(a+b)*y/b
    else:
        zbar = x-1j*y
        efield = 4.0/(zbar+pm.sqrt(zbar*zbar-a*a+b*b))
        efield_x = efield.real
        efield_y = -efield.imag
    denom = 4*np.pi*epsilon_0
    return efield_x/denom, efield_y/denom

@np.vectorize
def _efieldn_wb(x, y, sigma_x, sigma_y):
    a = sigma_x*pm.sqrt(3)
    b = sigma_y*pm.sqrt(3)
    zs = x-1j*y
    # if x**2/(a)**2+y**2/(b)**2 <= 1:
    chi = x/a+1j*y/b
    omegs = b*x/a-1j*a*y/b
    efield = 8.0*chi/(a+b) * \
        (1.0-(2.0*zs+omegs)*chi/(3.0*(a+b)))
    # else:
        # zs = pm.abs(x)+1j*pm.abs(y)
        # sqrt_diff = pm.sqrt(zs**2-a**2+b**2)
        # first_term = 2.0*zs/(zs+sqrt_diff)
        # efield = 2.0/zs*first_term*(zs+2.0*sqrt_diff)/(3.0*zs)
        # efield = efield.real*pm.sign(x) - 1.0j*efield.imag*pm.sign(y)
    denom = 4.*np.pi*epsilon_0
    return efield.real/denom, efield.imag/denom


def _efieldn_gauss_round(x, y, sig_x, sig_y):
    '''Return (E_x / Q, E_y / Q) for a round distribution
    with sigma_x == sigma_y == sig_r .
    '''
    r2 = x*x + y*y
    sig_r = sig_x
    amplitude = (1 - pm.exp(-r2/(2*sig_r*sig_r))) / (2*pi*epsilon_0 * r2)
    return x * amplitude, y * amplitude

def _efieldn_linearized(x, y, sig_x, sig_y):
    '''
    Returns linearized field
    '''
    a = pm.sqrt(2)*sig_x
    b = pm.sqrt(2)*sig_y
    amplitude  = 1./(2.*np.pi*epsilon_0*(a+b))
    return x/a * amplitude, y/b * amplitude

def add_sigma_check(efieldn, dist):
    '''Wrapper for a normalised electric field function.

    Adds the following actions before calculating the field:
    - exchange x and y quantities if sigma_x < sigma_y
    - apply round beam field formula when sigma_x close to sigma_y
    '''

    '''Threshold for relative transverse beam size difference
    below which the beam is assumed to be round:
    abs(1 - sig_y / sig_x) < ratio_threshold ==> round beam
    '''
    ratio_threshold = 1e-3

    '''Threshold for absolute transverse beam size difference
    below which the beam is assumed to be round:
    abs(sig_y - sig_x) < absolute_threshold ==> round beam
    '''
    absolute_threshold = 1e-10
    if dist == 'GS':
        efieldn_round = _efieldn_gauss_round
    elif dist == 'KV':
        efieldn_round = _efieldn_linearized
    elif dist == 'LN':
        efieldn_round = _efieldn_linearized

    @wraps(efieldn)
    def efieldn_checked(x, y, sig_x, sig_y, *args, **kwargs):
        tol_kwargs = dict(
            rtol=ratio_threshold,
            atol=absolute_threshold
        )
        if pm.allclose(sig_y, sig_x, **tol_kwargs):
            if pm.almost_zero(sig_y, **tol_kwargs):
                en_x = en_y = pm.zeros(x.shape, dtype=x.dtype)
            else:
                en_x, en_y = efieldn_round(x, y, sig_x, sig_y, *args, **kwargs)
        elif pm.all(sig_x < sig_y):
            en_y, en_x = efieldn(y, x, sig_y, sig_x, *args, **kwargs)
        else:
            en_x, en_y = efieldn(x, y, sig_x, sig_y, *args, **kwargs)
        return en_x, en_y
    return efieldn_checked
