'''
@author Kevin Li, Adrian Oeftiger
@date 21.06.2017
@brief 2D distribution integration methods for y(x) parametrised domains
'''
from scipy.integrate import quad, dblquad, cumtrapz, romb

import numpy as np


def quad2d(f, ylimits, xmin, xmax):
    '''Integrate 2D function f=f(x,y) over interval [xmin,xmax] and
    between contours [-ylimits(x), ylimits(x)] using the numerical
    scipy.integrate.dblquad integration method.
    '''
    Q, error = dblquad(lambda y, x: f(x, y), xmin, xmax,
                       lambda x: -ylimits(x), lambda x: ylimits(x))

    return Q

### scipy.integrate.dblquad moment integration:

def compute_zero_quad(psi, ylimit_min, ylimit_max, xmin, xmax):
    '''Compute the zeroth moment of the distribution function psi from
    xmin to xmax between the contours ylimit_min and ylimit_max
    (e.g. an RFBucket separatrix) using the numerical
    scipy.integrate.dblquad integration method.

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimit_min, ylimit_max: contour functions yielding the lower
          and the upper y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
    '''
    Q, error = dblquad(lambda y, x: psi(x, y), xmin, xmax,
                       ylimit_min, ylimit_max)

    return Q

def compute_mean_quad(psi, ylimit_min, ylimit_max, xmin, xmax, direction='x'):
    '''Compute the first moment of the distribution function psi from
    xmin to xmax between the contours ylimit_min and ylimit_max
    (e.g. an RFBucket separatrix) using the numerical
    scipy.integrate.dblquad integration method.

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimit_min, ylimit_max: contour functions yielding the lower
          and the upper y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
        - direction: 'x' or 'y' for calculating the mean in the first
          or second argument of psi, respectively (default: 'x')
    '''

    Q = compute_zero_quad(psi, ylimit_min, ylimit_max, xmin, xmax)
    if direction is 'x':
        f = lambda y, x: x * psi(x, y)
    elif direction is 'y':
        f = lambda y, x: y * psi(x, y)
    else:
        raise ValueError('direction needs to be either "x" or "y".')

    M, error = dblquad(f, xmin, xmax, ylimit_min, ylimit_max)

    return M/Q

def compute_var_quad(psi, ylimit_min, ylimit_max, xmin, xmax, direction='x'):
    '''Compute the second moment (variance) with respect to the x or
    y direction of the distribution function psi from xmin to xmax
    between the contours ylimit_min and ylimit_max (e.g. an RFBucket
    separatrix) using the numerical scipy.integrate.dblquad integration
    method.

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimit_min, ylimit_max: contour functions yielding the lower
          and the upper y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
        - direction: 'x' or 'y' for calculating the standard deviation
          in the first or second argument of psi, respectively
          (default: 'x')
    '''

    Q = compute_zero_quad(psi, ylimit_min, ylimit_max, xmin, xmax)
    M = compute_mean_quad(psi, ylimit_min, ylimit_max, xmin, xmax, direction)
    if direction is 'x':
        f = lambda y, x: (x - M)**2 * psi(x, y)
    elif direction is 'y':
        f = lambda y, x: (y - M)**2 * psi(x, y)
    else:
        raise ValueError('direction needs to be either "x" or "y".')

    V, error = dblquad(f, xmin, xmax, ylimit_min, ylimit_max)

    return V/Q

def compute_cov_quad(psi, ylimit_min, ylimit_max, xmin, xmax):
    '''Compute the second moments (covariance matrix entries) of the
    distribution function psi from xmin to xmax between the contours
    ylimit_min and ylimit_max (e.g. an RFBucket separatrix) using the
    numerical scipy.integrate.dblquad integration method.
    For x the first and y the second argument of psi, return the tuple
    (variance(x), covariance(x,y), variance(y)).

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimits: contour function yielding the y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
    '''

    Q = compute_zero_quad(psi, ylimit_min, ylimit_max, xmin, xmax)
    M_x = compute_mean_quad(psi, ylimit_min, ylimit_max, xmin, xmax, 'x')
    M_y = compute_mean_quad(psi, ylimit_min, ylimit_max, xmin, xmax, 'y')

    V_x = compute_var_quad(psi, ylimit_min, ylimit_max, xmin, xmax, 'x')
    V_y = compute_var_quad(psi, ylimit_min, ylimit_max, xmin, xmax, 'y')

    f = lambda y, x: (x - M_x) * (y - M_y) * psi(x, y)
    C_xy, error = dblquad(f, xmin, xmax, ylimit_min, ylimit_max)

    C_xy /= Q

    return V_x, C_xy, V_y

### scipy.integrate.cumtrapz moment integration:

def compute_zero_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax,
                          n_samples=513):
    '''Compute the zeroth moment of the distribution function psi from
    xmin to xmax between the contours ylimit_min and ylimit_max
    (e.g. an RFBucket separatrix) using the numerical
    scipy.integrate.cumtrapz integration method.

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimit_min, ylimit_max: contour functions yielding the lower
          and the upper y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
        - n_samples: integer number of sampling points for integration
    '''

    x_arr = np.linspace(xmin, xmax, num=n_samples)
    dx = x_arr[1] - x_arr[0]

    Q = 0
    for x in x_arr:
        y = np.linspace(ylimit_min(x), ylimit_max(x), num=n_samples)
        z = psi(x, y)
        Q += cumtrapz(z, y)[-1]
    Q *= dx

    return Q

def compute_mean_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax,
                          direction='x', n_samples=513):
    '''Compute the first moment of the distribution function psi from
    xmin to xmax between the contours ylimit_min and ylimit_max
    (e.g. an RFBucket separatrix) using the numerical
    scipy.integrate.cumtrapz integration method.

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimit_min, ylimit_max: contour functions yielding the lower
          and the upper y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
        - direction: 'x' or 'y' for calculating the mean in the first
          or second argument of psi, respectively (default: 'x')
        - n_samples: integer number of sampling points for integration
    '''

    Q = compute_zero_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax,
                              n_samples)

    x_arr = np.linspace(xmin, xmax, num=n_samples)
    dx = x_arr[1] - x_arr[0]

    if direction is 'x':
        f = lambda x, y: x * psi(x, y)
    elif direction is 'y':
        f = lambda x, y: y * psi(x, y)
    else:
        raise ValueError('direction needs to be either "x" or "y".')

    M = 0
    for x in x_arr:
        y = np.linspace(ylimit_min(x), ylimit_max(x), num=n_samples)
        z = f(x, y)
        M += cumtrapz(z, y)[-1]
    M *= dx

    return M/Q

def compute_var_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax,
                         direction='x', n_samples=513):
    '''Compute the second moment (variance) with respect to the x or
    y direction of the distribution function psi from xmin to xmax
    between the contours ylimit_min and ylimit_max
    (e.g. an RFBucket separatrix) using the numerical
    scipy.integrate.cumtrapz integration method.

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimit_min, ylimit_max: contour functions yielding the lower
          and the upper y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
        - direction: 'x' or 'y' for calculating the mean in the first
          or second argument of psi, respectively (default: 'x')
        - n_samples: integer number of sampling points for integration
    '''

    Q = compute_zero_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax,
                              n_samples)
    M = compute_mean_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax,
                              direction, n_samples)

    x_arr = np.linspace(xmin, xmax, num=n_samples)
    dx = x_arr[1] - x_arr[0]

    if direction is 'x':
        f = lambda x, y: (x - M)**2 * psi(x, y)
    elif direction is 'y':
        f = lambda x, y: (y - M)**2 * psi(x, y)
    else:
        raise ValueError('direction needs to be either "x" or "y".')

    V = 0
    for x in x_arr:
        y = np.linspace(ylimit_min(x), ylimit_max(x), num=n_samples)
        z = f(x, y)
        V += cumtrapz(z, y)[-1]
    V *= dx

    return V/Q

def compute_cov_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax,
                         n_samples=513):
    '''Compute the second moments (covariance matrix entries) of the
    distribution function psi from xmin to xmax between the contours
    ylimit_min and ylimit_max (e.g. an RFBucket separatrix) using the
    numerical scipy.integrate.cumtrapz integration method.
    For x the first and y the second argument of psi, return the tuple
    (variance(x), covariance(x,y), variance(y)).

    Arguments:
        - psi: 2D distribution function with two arguments x and y
        - ylimit_min, ylimit_max: contour functions yielding the lower
          and the upper y(x) limit for given x
        - xmin, xmax: lower and upper limit in the first argument of psi
        - n_samples: integer number of sampling points for integration
    '''


    Q = compute_zero_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax)
    M_x = compute_mean_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax, 'x')
    M_y = compute_mean_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax, 'y')

    V_x = compute_var_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax, 'x')
    V_y = compute_var_cumtrapz(psi, ylimit_min, ylimit_max, xmin, xmax, 'y')

    x_arr = np.linspace(xmin, xmax, num=n_samples)
    dx = x_arr[1] - x_arr[0]

    C_xy = 0
    for x in x_arr:
        y = np.linspace(ylimit_min(x), ylimit_max(x), num=n_samples)
        z = (x - M_x) * (y - M_y) * psi(x, y)
        C_xy += cumtrapz(z, y)[-1]
    C_xy *= dx

    C_xy /= Q

    return V_x, C_xy, V_y


### scipy.integrate.romberg standard deviation integration:
### ==> not yet adapted to above quad and cumtrapz approaches,
###     works as Kevin had previously defined it in the RFBucketMatcher
###     (slightly corrected to make this function work here)

def compute_std_romberg(psi, ylimits, xmin, xmax, n_samples=513):
    '''
    Compute the standard deviation of the distribution function psi
    from xmin to xmax along the contours ylimits using numerical
    integration methods.
    '''

    x_arr = np.linspace(xmin, xmax, num=n_samples)
    dx = x_arr[1] - x_arr[0]

    Q, V = 0, 0
    for x in x_arr:
        y = np.linspace(0, ylimits(x), num=n_samples)
        dy = y[1] - y[0]
        z = psi(x, y)
        Q += romb(z, dy)
        z = x**2 * psi(x, y)
        V += romb(z, dy)
    Q *= dx
    V *= dx

    return np.sqrt(V/Q)
