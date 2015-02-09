from scipy.integrate import quad, fixed_quad, dblquad, cumtrapz, romb


def quad2d(f, ylimits, xmin, xmax):
    Q, error = dblquad(lambda y, x: f(x, y), xmin, xmax,
                       lambda x: -ylimits(x), lambda x: ylimits(x))

    return Q


def _compute_zero_quad(self, psi, p_sep, xmin, xmax):
    '''
    Compute the variance of the distribution function psi from xmin
    to xmax along the contours p_sep using numerical integration
    methods.
    '''

    Q, error = dblquad(lambda y, x: psi(x, y), xmin, xmax,
                lambda x: 0, lambda x: p_sep(x))

    return Q

def _compute_mean_quad(self, psi, p_sep, xmin, xmax):
    '''
    Compute the variance of the distribution function psi from xmin
    to xmax along the contours p_sep using numerical integration
    methods.
    '''

    Q = self._compute_zero_quad(psi, p_sep, xmin, xmax)
    M, error = dblquad(lambda y, x: x * psi(x, y), xmin, xmax,
                lambda x: 0, lambda x: p_sep(x))

    return M/Q

def _compute_std_quad(self, psi, p_sep, xmin, xmax):
    '''
    Compute the variance of the distribution function psi from xmin
    to xmax along the contours p_sep using numerical integration
    methods.
    '''

    Q = self._compute_zero_quad(psi, p_sep, xmin, xmax)
    M = self._compute_mean_quad(psi, p_sep, xmin, xmax)
    V, error = dblquad(lambda y, x: (x-M) ** 2 * psi(x, y), xmin, xmax,
                       lambda x: 0, lambda x: p_sep(x))

    return np.sqrt(V/Q)

def _compute_zero_cumtrapz(self, psi, p_sep, xmin, xmax):

    x_arr = np.linspace(xmin, xmax, 257)
    dx = x_arr[1] - x_arr[0]

    Q = 0
    for x in x_arr:
        y = np.linspace(0, p_sep(x), 257)
        z = psi(x, y)
        Q += cumtrapz(z, y)[-1]
    Q *= dx

    return Q

def _compute_mean_cumtrapz(self, psi, p_sep, xmin, xmax):

    Q = self._compute_zero_cumtrapz(psi, p_sep, xmin, xmax)

    x_arr = np.linspace(xmin, xmax, 257)
    dx = x_arr[1] - x_arr[0]

    M = 0
    for x in x_arr:
        y = np.linspace(0, p_sep(x), 257)
        z = x * psi(x, y)
        M += cumtrapz(z, y)[-1]
    M *= dx

    return M/Q

def _compute_std_cumtrapz(self, psi, p_sep, xmin, xmax):
    '''
    Compute the variance of the distribution function psi from xmin
    to xmax along the contours p_sep using numerical integration
    methods.
    '''

    Q = self._compute_zero_cumtrapz(psi, p_sep, xmin, xmax)
    M = self._compute_mean_cumtrapz(psi, p_sep, xmin, xmax)

    x_arr = np.linspace(xmin, xmax, 257)
    dx = x_arr[1] - x_arr[0]

    V = 0
    for x in x_arr:
        y = np.linspace(0, p_sep(x), 257)
        z = (x-M)**2 * psi(x, y)
        V += cumtrapz(z, y)[-1]
    V *= dx

    return np.sqrt(V/Q)

def _compute_std_romberg(self, psi, p_sep, xmin, xmax):
    '''
    Compute the variance of the distribution function psi from xmin
    to xmax along the contours p_sep using numerical integration
    methods.
    '''

    x_arr = np.linspace(xmin, xmax, 257)
    dx = x_arr[1] - x_arr[0]

    Q, V = 0, 0
    for x in x_arr:
        y = np.linspace(0, p_sep(x), 257)
        dy = y[1] - y[0]
        z = psi(x, y)
        Q += romb(z, dy)
        z = x**2 * psi(x, y)
        V += romb(z, dy)
    Q *= dx
    V *= dx

    return np.sqrt(V/Q)
