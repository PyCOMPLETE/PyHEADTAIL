'''
@file matching
@author Kevin Li, Adrian Oeftiger
@date February 2014
@brief Module for matching transverse and longitudinal distributions
@copyright CERN
'''
from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import RandomState

from scipy.constants import c, e
from scipy.optimize import newton
from scipy.interpolate import interp2d
from scipy.integrate import quad, dblquad, cumtrapz, romb

import pylab as plt


class PhaseSpace(object):
    """Knows how to distribute particle coordinates for a beam
    according to certain distribution functions.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate(self, beam):
        """Creates the beam macroparticles according to a
        distribution function (depends on the implementing class).
        """
        pass


class GaussianX(PhaseSpace):
    """Horizontal Gaussian particle phase space distribution."""

    def __init__(self, sigma_x, sigma_xp, generator_seed=None):
        """Initiates the horizontal beam coordinates
        to the given Gaussian shape.
        """
        self.sigma_x  = sigma_x
        self.sigma_xp = sigma_xp

        self.random_state = RandomState()
        self.random_state.seed(generator_seed)

    @classmethod
    def from_optics(cls, alpha_x, beta_x, epsn_x, betagamma, generator_seed=None):
        """Initialise GaussianX from the given optics functions.
        beta_x is given in meters and epsn_x in micrometers.
        """

        sigma_x  = np.sqrt(beta_x * epsn_x * 1e-6 / betagamma)
        sigma_xp = sigma_x / beta_x

        return cls(sigma_x, sigma_xp, generator_seed)

    def generate(self, beam):
        beam.x = self.sigma_x * self.random_state.randn(beam.n_macroparticles)
        beam.xp = self.sigma_xp * self.random_state.randn(beam.n_macroparticles)


class GaussianY(PhaseSpace):
    """Vertical Gaussian particle phase space distribution."""

    def __init__(self, sigma_y, sigma_yp, generator_seed=None):
        """Initiates the vertical beam coordinates
        to the given Gaussian shape.
        """
        self.sigma_y  = sigma_y
        self.sigma_yp = sigma_yp

        self.random_state = RandomState()
        self.random_state.seed(generator_seed)

    @classmethod
    def from_optics(cls, alpha_y, beta_y, epsn_y, betagamma, generator_seed=None):
        """Initialise GaussianY from the given optics functions.
        beta_y is given in meters and epsn_y in micrometers.
        """

        sigma_y  = np.sqrt(beta_y * epsn_y * 1e-6 / betagamma)
        sigma_yp = sigma_y / beta_y

        return cls(sigma_y, sigma_yp, generator_seed)

    def generate(self, beam):
        beam.y = self.sigma_y * self.random_state.randn(beam.n_macroparticles)
        beam.yp = self.sigma_yp * self.random_state.randn(beam.n_macroparticles)


class GaussianZ(PhaseSpace):
    """Longitudinal Gaussian particle phase space distribution."""

    def __init__(self, sigma_z, sigma_dp, is_accepted=None, generator_seed=None):
        """Initiates the longitudinal beam coordinates to a given
        Gaussian shape. If the argument is_accepted is set to
        the is_in_separatrix(z, dp, beam) method of a RFSystems
        object (or similar), macroparticles will be initialised
        until is_accepted returns True.
        """
        self.sigma_z = sigma_z
        self.sigma_dp = sigma_dp
        self.is_accepted = is_accepted

        self.random_state = RandomState()
        self.random_state.seed(generator_seed)

    @classmethod
    def from_optics(cls, beta_z, epsn_z, p0, is_accepted=None,
                    generator_seed=None):
        """Initialise GaussianZ from the given optics functions.
        For the argument is_accepted see __init__.
        """

        sigma_z = np.sqrt(beta_z*epsn_z/(4*np.pi) * e/p0)
        sigma_dp = sigma_z / beta_z

        return cls(sigma_z, sigma_dp, is_accepted, generator_seed)

    def generate(self, beam):
        beam.z = self.sigma_z * self.random_state.randn(beam.n_macroparticles)
        beam.dp = self.sigma_dp * self.random_state.randn(beam.n_macroparticles)
        if self.is_accepted:
            self._redistribute(beam)

    def _redistribute(self, beam):
        n = beam.n_macroparticles
        z = beam.z.copy()
        dp = beam.dp.copy()

        mask_out = ~self.is_accepted(z, dp)
        while mask_out.any():
            n_gen = np.sum(mask_out)
            z[mask_out] = self.sigma_z * self.random_state.randn(n_gen)
            dp[mask_out] = self.sigma_dp * self.random_state.randn(n_gen)
            mask_out = ~self.is_accepted(z, dp)
            print 'Reiterate on non-accepted particles'

        # for i in xrange(n):
        #     while not self.is_accepted(z[i], dp[i]):
        #         z[i]  = self.sigma_z * self.random_state.randn()
        #         dp[i] = self.sigma_dp * self.random_state.randn()

        beam.z = z
        beam.dp = dp


class GaussianTheta(PhaseSpace):
    """Longitudinal Gaussian particle phase space distribution."""

    def __init__(self, sigma_theta, sigma_dE, is_accepted=None, generator_seed=None):

        self.sigma_theta = sigma_theta
        self.sigma_dE = sigma_dE
        self.is_accepted = is_accepted

        self.random_state = RandomState()
        self.random_state.seed(generator_seed)

    def generate(self, beam):
        beam.theta = self.sigma_theta * self.random_state.randn(beam.n_macroparticles)
        beam.delta_E = self.sigma_dE * self.random_state.randn(beam.n_macroparticles)
        if self.is_accepted:
            self._redistribute(beam)

    def _redistribute(self, beam):
        n = beam.n_macroparticles
        theta = beam.theta.copy()
        delta_E = beam.delta_E.copy()
        for i in xrange(n):
            while not self.is_accepted(theta[i], delta_E[i]):
                theta[i]  = self.sigma_theta * self.random_state.randn()
                delta_E[i] = self.sigma_dE * self.random_state.randn()
        beam.theta = theta
        beam.delta_E = delta_E


class RFBucket(PhaseSpace):

    def __init__(self, psi, rfsystem, sigma_z=None, epsn_z=None):

        self.psi = psi
        self.H = rfsystem
        self.sigma_z = sigma_z

        self._compute_std = self._compute_std_quad

        if sigma_z and not epsn_z:
            self.variable = sigma_z
            self.psi_for_variable = self.psi_for_bunchlength_newton_method
        elif not sigma_z and epsn_z:
            self.variable = epsn_z
            self.psi_for_variable = self.psi_for_emittance_newton_method
        else:
            raise ValueError("Can not generate mismatched matched distribution!")

        # self.generate = self.dontgenerate

    def psi_for_emittance(self, epsn_z):

        H = self.H
        psi_c =  self.psi(H.hamiltonian, H.Hmax)
        psi = psi_c.function

        # Maximum emittance
        epsn_max = self._compute_mean_quad(lambda y, x: 1, H.separatrix, H.zleft, H.zright) * 2*self.p0/e
        # print 'Maximum emittance', epsn_max
        if epsn_z > epsn_max:
            print '\n*** Emittance larger than bucket; using full bucket emittance', epsn_max, ' [eV s].\n'
            epsn_z = epsn_max*0.99
        print '\n*** Maximum emittance', epsn_z, 'm.\n'

        # Cut on z-axis
        zz = np.linspace(H.zs + np.abs(H.zs)*0.01, H.zright - np.abs(H.zright)*0.01, 10)
        A = []
        for i, zc in enumerate(zz):
            try:
                zleft, zright = self.H.get_z_left_right(zc)
                A.append( self._compute_mean_quad(lambda y, x: 1, self.H.equihamiltonian(zc), zleft, zright) * 2*self.p0/e )
            except IndexError:
                print '\n*** z cut', zc, 'too tight; skipping value.\n'
                zz = np.delete(zz, i)
        A = np.array(A)

        ix = np.where(np.diff(np.sign(A-epsn_z)))[0]
        m = (A[ix+1] - A[ix])/(zz[ix+1] - zz[ix])
        dy = epsn_z - A[ix]
        zc_emittance = zz[ix] + dy/m
        try:
            zc_emittance[0]
        except IndexError:
            raise RuntimeError("\n*** Emittance", epsn_z, "not found in range. Increase range or resolution.")

        # Width for cut on z-axis
        fw = self.H.zright-self.H.zs
        vv = np.linspace(fw*0.01, fw*0.99, 10)
        L = []
        for vc in vv:
            psi_c.H0 = H.H0(vc)
            L.append( H._get_zero_crossings(lambda x: psi(x, 0)-0.0044318484119380075)[-1] )
        L = np.array(L)

        # TODO: catch if it is empty
        ix = np.where(np.diff(np.sign(L-zc_emittance)))[0]
        m = (L[ix+1] - L[ix])/(vv[ix+1] - vv[ix])
        dy = zc_emittance - L[ix]
        zc_bunchlength = vv[ix] + dy/m
        try:
            zc_bunchlength[0]
        except IndexError:
            raise RuntimeError("\n*** RMS length not found in range. Increase range or resolution.")

        psi_c.H0 = H.H0(zc_bunchlength)
        sigma = self._compute_std(psi, H.separatrix, H.zleft, H.zright)
        # print epsn_z, sigma

        # xx, pp = np.linspace(H.zleft, H.zright, 200), np.linspace(-H.p_max(H.zright), H.p_max(H.zright), 200)
        # XX, PP = np.meshgrid(xx, pp)
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        # ax4 = fig.add_subplot(224, projection='3d')
        # ax1.plot(zz, A)
        # ax1.axhline(a, c='r', lw=2)
        # ax1.plot(zc_emittance, a, '+', ms=12, mew=4)
        # ax1.grid()
        # ax2.plot(vv, L)
        # ax2.axhline(zc_emittance, c='r', lw=2)
        # ax2.plot(zc_bunchlength, zc_emittance, '+', ms=12, mew=4)
        # ax2.grid()
        # ax3.plot(xx, psi(xx, 0))
        # ax3.axvline(sigma, c='y', lw=2)
        # ax3.axvline(zc_emittance, c='r', lw=2)
        # ax4.plot_surface(XX, PP, psi(XX, PP), cmap=plt.cm.jet)
        # plt.show()

        return psi#, epsn_z, sigma

    def psi_for_bunchlength(self, sigma):

        H = self.H
        psi_c =  self.psi(H.hamiltonian, H.Hmax)
        psi = psi_c.function

        # Maximum bunch length
        psi_c.H0 = self.H.H0(self.circumference)
        sigma_max = self._compute_std(psi, self.H.separatrix, self.H.zleft, self.H.zright)
        if sigma > sigma_max:
            print "\n*** RMS bunch larger than bucket; using full bucket rms length", sigma_max, " m."
            sigma = sigma_max*0.99

        # Width for bunch length
        fw = self.H.zright-self.H.zs
        zz = np.linspace(fw*0.05, fw*0.95, 20)
        L = []
        for i, zc in enumerate(zz):
            psi_c.H0 = self.H.H0(zc)
            print i+1, psi_c.H0
            L.append( self._compute_std(psi, H.separatrix, H.zleft, H.zright) )
        L = np.array(L)

        ix = np.where(np.diff(np.sign(L-sigma)))[0]
        m = (L[ix+1] - L[ix])/(zz[ix+1] - zz[ix])
        dy = sigma - L[ix]
        k = zz[ix] + dy/m
        psi_c.H0 = self.H.H0(k)

        for zc in [zz[ix], k, zz[ix+1]]:
            psi_c.H0 = self.H.H0(zc)
            print zc, self._compute_std(psi, H.separatrix, H.zleft, H.zright)

        xx, pp = np.linspace(H.zleft, H.zright, 200), np.linspace(-H.p_max(H.zright), H.p_max(H.zright), 200)
        XX, PP = np.meshgrid(xx, pp)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        ax3 = fig.add_subplot(133, projection='3d')
        ax1.plot(zz, L, '-*')
        ax1.axhline(sigma, c='r', lw=2)
        ax1.plot(k, sigma, '+', ms=12, mew=4)
        ax1.grid()
        ax2.plot(xx, psi(xx, 0))
        ax2.axvline(sigma, c='y', lw=2)
        ax3.plot_surface(XX, PP, psi(XX, PP), cmap=plt.cm.jet)
        plt.show()

        exit(-1)

        return psi

    def psi_for_emittance_newton_method(self, epsn_z):

        H = self.H
        psi_c =  self.psi(H.hamiltonian, H.Hmax)
        psi = psi_c.function

        sigma = np.sqrt(epsn_z/(4*np.pi) * H.beta_z * e/H.p0_reference)

        # Maximum emittance
        epsn_max = self._compute_zero_quad(lambda y, x: 1, H.separatrix, H.zleft, H.zright) * 2*H.p0_reference/e
        if epsn_z > epsn_max:
            print '\n*** Emittance larger than bucket; using full bucket emittance', epsn_max, ' [eV s].'
            epsn_z = epsn_max*0.99
        print '\n*** Maximum emittance', epsn_max, 'eV s.'

        def get_zcut_for_epsn(zcut):
            if zcut > H.zright:
                zcut = H.zright*0.95
            if zcut < H.zleft:
                zcut = H.zleft*0.95

            zleft, zright = H.get_z_left_right(zcut)
            eqh = H.equihamiltonian(zcut)
            emittance = self._compute_zero_quad(lambda y, x: 1, eqh, zleft, zright) * 2*H.p0_reference/e
            if np.isnan(emittance):
                raise ValueError

            return emittance-epsn_z

        # Width for emittance
        def get_zc_for_zcut(zc):
            psi_c.H0 = H.H0(zc)
            zleft, zright = H._get_zero_crossings(lambda x: psi(x, 0)-0.0044318484119380075)
            eqh = H.equihamiltonian(zcut_bar)
            if np.isnan(zright):
                raise ValueError

            if zcut_bar > 0:
                return zright-zcut_bar
            else:
                return zleft-zcut_bar

        zcut_bar = newton(get_zcut_for_epsn, sigma)
        zc_bar = newton(get_zc_for_zcut, sigma)

        zleft, zright = self.H.get_z_left_right(zcut_bar)
        eqh = H.equihamiltonian(zcut_bar)
        print '\n--> Emittance:', self._compute_zero_quad(lambda y, x: 1, eqh, zleft, zright) * 2*H.p0_reference/e
        psi_c.H0 = H.H0(zc_bar)
        sigma = self._compute_std(psi, H.separatrix, H.zleft, H.zright)
        print '--> Bunch length:', sigma

#         xx, pp = np.linspace(H.zleft, H.zright, 200), np.linspace(-H.p_max(H.zright), H.p_max(H.zright), 200)
#         XX, PP = np.meshgrid(xx, pp)
#         fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,12), sharex=True)
#         ax3 = fig.add_subplot(313, projection='3d')
#         ax1.contour(XX, PP, H.hamiltonian(XX, PP))
#         ax1.plot(xx, eqh(xx), lw=2, c='r')
#         ax1.plot(xx, -eqh(xx), lw=2, c='r')
#         ax1.axvspan(zleft, zright, color='orange', alpha=0.2)
#         ax1.axvline(zcut_bar, c='r', lw=2)
#         ax2.plot(xx, psi(xx, 0), '-')
#         ax2.plot(xx, psi(xx, 0), '-')
#         ax2.axvspan(zleft, zright, color='orange', alpha=0.2)
#         ax3.plot_surface(XX, PP, psi(XX, PP), cmap=plt.get_cmap('jet'))
#         plt.show()

        return psi

    def psi_for_bunchlength_newton_method(self, sigma):

        H = self.H
        psi_c =  self.psi(H.hamiltonian, H.Hmax)
        psi = psi_c.function

        # Maximum bunch length
        psi_c.H0 = self.H.H0(self.H.circumference)
        sigma_max = self._compute_std(psi, self.H.separatrix, self.H.zleft, self.H.zright)
        if sigma > sigma_max:
            print "\n*** RMS bunch larger than bucket; using full bucket rms length", sigma_max, " m."
            sigma = sigma_max*0.99
        print '\n*** Maximum RMS bunch length', sigma_max, 'm.'

        # Width for bunch length
        def sigma_for_zc(zc):
            psi_c.H0 = self.H.H0(zc)
            length = self._compute_std(psi, self.H.separatrix, self.H.zleft, self.H.zright)
            if np.isnan(length):
                raise ValueError
            return length-sigma

        zc_bar = newton(sigma_for_zc, sigma)

        psi_c.H0 = self.H.H0(zc_bar)
        sigma = self._compute_std(psi, H.separatrix, H.zleft, H.zright)
        print '\n--> Bunch length:', sigma
        zleft, zright = H._get_zero_crossings(lambda x: psi(x, 0)-0.0044318484119380075)
        zcut_bar = zright
        eqh = H.equihamiltonian(zcut_bar)
        print '--> Emittance:', self._compute_zero_quad(lambda y, x: 1, eqh, zleft, zright) * 2*H.p0_reference/e

#         xx, pp = np.linspace(H.zleft, H.zright, 200), np.linspace(-H.p_max(H.zright), H.p_max(H.zright), 200)
#         XX, PP = np.meshgrid(xx, pp)
#         fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8,12), sharex=True)
#         ax3 = fig.add_subplot(133, projection='3d')
#         ax1.contour(XX, PP, H.hamiltonian(XX, PP))
#         ax1.plot(xx, eqh(xx), lw=2, c='r')
#         ax1.plot(xx, -eqh(xx), lw=2, c='r')
#         ax1.axvspan(zleft, zright, color='orange', alpha=0.2)
#         ax1.axvline(zcut_bar, c='r', lw=2)
#         ax2.plot(xx, psi(xx, 0), '-')
#         ax2.plot(xx, psi(xx, 0), '-')
#         ax2.axvspan(zleft, zright, color='orange', alpha=0.2)
#         ax3.plot_surface(XX, PP, psi(XX, PP), cmap=plt.get_cmap('jet'))
#         plt.show()

        return psi

    def generate(self, particles):
        '''
        Generate a 2d phase space of n_particles particles randomly distributed
        according to the particle distribution function psi within the region
        [xmin, xmax, ymin, ymax].
        '''
        psi = self.psi_for_variable(self.variable)
        # print self.variable
        # print self._compute_std(psi, self.H.separatrix, self.H.zleft, self.H.zright)

        # x = np.zeros(particles.n_macroparticles)
        # y = np.zeros(particles.n_macroparticles)

        # Bin
        i, j = 0, 0
        nx, ny = 128, 128
        xmin, xmax = self.H.zleft, self.H.zright
        ymin, ymax = -self.H.p_max(self.H.zright), self.H.p_max(self.H.zright)
        lx = (xmax - xmin)
        ly = (ymax - ymin)

        xx = np.linspace(xmin, xmax, nx + 1)
        yy = np.linspace(ymin, ymax, ny + 1)
        XX, YY = np.meshgrid(xx, yy)
        HH = psi(XX, YY)
        psi_interp = interp2d(xx, yy, HH)

        # ================================================================
        # mask_out = ~self.is_accepted(z, dp)
        # while mask_out.any():
        #     n_gen = np.sum(mask_out)
        #     z[mask_out] = self.sigma_z * self.random_state.randn(n_gen)
        #     dp[mask_out] = self.sigma_dp * self.random_state.randn(n_gen)
        #     mask_out = ~self.is_accepted(z, dp)
        #     print 'Reiterate on non-accepted particles'

        # for i in xrange(n):
        #     while not self.is_accepted(z[i], dp[i]):
        #         z[i]  = self.sigma_z * self.random_state.randn()
        #         dp[i] = self.sigma_dp * self.random_state.randn()
        # ================================================================

        n_gen = particles.n_macroparticles
        u = xmin + lx * np.random.random(n_gen)
        v = ymin + ly * np.random.random(n_gen)
        s = np.random.random(n_gen)
        mask_out = ~(s<psi(u, v))
        while mask_out.any():
            n_gen = np.sum(mask_out)
            u[mask_out] = xmin + lx * np.random.random(n_gen)
            v[mask_out] = ymin + ly * np.random.random(n_gen)
            s[mask_out] = np.random.random(n_gen)
            mask_out = ~(s<psi(u, v))
            # print 'Reiterate on non-accepted particles.'
            # print n_gen, '\n'

        # while j < particles.n_macroparticles:
        #     u = xmin + lx * np.random.random()
        #     v = ymin + ly * np.random.random()

        #     s = np.random.random()

        #     i += 1
        #     if s < psi_interp(u, v):
        #         x[j] = u
        #         y[j] = v
        #         # TODO: check if this does not cause problems! Setter for item does not work - not implemented!
        #         # particles.dp[j] = v
        #         j += 1

        particles.z = u
        particles.dp = v
        # particles.psi = psi
        # return x, y, j / i * dx * dy, psi

    def _compute_zero_quad(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin to xmax
        along the contours p_sep using numerical integration methods.
        '''

        Q, error = dblquad(lambda y, x: psi(x, y), xmin, xmax,
                    lambda x: 0, lambda x: p_sep(x))

        return Q

    def _compute_mean_quad(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin to xmax
        along the contours p_sep using numerical integration methods.
        '''

        Q = self._compute_zero_quad(psi, p_sep, xmin, xmax)
        M, error = dblquad(lambda y, x: x * psi(x, y), xmin, xmax,
                    lambda x: 0, lambda x: p_sep(x))

        return M/Q

    def _compute_std_quad(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin to xmax
        along the contours p_sep using numerical integration methods.
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
        Compute the variance of the distribution function psi from xmin to xmax
        along the contours p_sep using numerical integration methods.
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
        Compute the variance of the distribution function psi from xmin to xmax
        along the contours p_sep using numerical integration methods.
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


class UniformX(PhaseSpace):
    """
    Horizontal uniform particle phase space distribution.
    """

    def __init__(self, x_min, x_max):

        self.x_min, self.x_max = x_min, x_max

    def generate(self, particles):
        dx = self.x_max - self.x_min
        particles.x = self.x_min + np.random.rand(particles.n_macroparticles) * dx
        particles.xp = 0. * particles.x


class UniformY(PhaseSpace):
    """
    Vertical uniform particle phase space distribution.
    """

    def __init__(self, y_min, y_max):

        self.y_min, self.y_max = y_min, y_max

    def generate(self, particles):
        dy = self.y_max - self.y_min
        particles.y = self.y_min + np.random.rand(particles.n_macroparticles) * dy
        particles.yp = 0. * particles.y


class UniformZ(PhaseSpace):
    """
    Longitudinal uniform particle phase space distribution.
    """

    def __init__(self, z_min, z_max):

        self.z_min, self.z_max = z_min, z_max

    def generate(self, particles):
        dz = self.z_max - self.z_min
        particles.z = self.z_min + np.random.rand(particles.n_macroparticles) * dz
        particles.dp = 0. * particles.z


class ImportX(PhaseSpace):

    def __init__(self, x, xp):

        self.x = np.array(x)
        self.xp = np.array(xp)

    def generate(self, particles):

        # x = np.zeros(particles.n_macroparticles)
        # xp = np.zeros(particles.n_macroparticles)
        # particles.x = self.x[:particles.n_macroparticles]
        # particles.xp = self.xp[:particles.n_macroparticles]

        assert(particles.n_macroparticles == len(self.x) == len(self.xp))
        particles.x = self.x.copy()
        particles.xp = self.xp.copy()


class ImportY(PhaseSpace):

    def __init__(self, y, yp):

        self.y = np.array(y)
        self.yp = np.array(yp)

    def generate(self, particles):

        assert(particles.n_macroparticles == len(self.y) == len(self.yp))
        particles.y = self.y.copy()
        particles.yp = self.yp.copy()


class ImportZ(PhaseSpace):

    def __init__(self, z, dp):

        self.z = np.array(z)
        self.dp = np.array(dp)

    def generate(self, particles):

        assert(particles.n_macroparticles == len(self.z) == len(self.dp))
        particles.z = self.z.copy()
        particles.dp = self.dp.copy()


class StationaryExponential(object):

    def __init__(self, H, Hmax=None, width=1000, Hcut=0):
        self.H = H
        self.H0 = 1
        if not Hmax:
            self.Hmax = H(0, 0)
        else:
            self.Hmax = Hmax
        self.Hcut = Hcut
        self.width = width

    def function(self, z, dp):
        # psi = np.exp((self.H(z, dp)) / (self.width*self.Hmax)) - 1
        # psi_offset = np.exp(self.Hcut / (self.width*self.Hmax)) - 1
        # psi_norm = (np.exp(1/self.width) - 1) - psi_offset
        # return ( (psi-psi_offset) / psi_norm ).clip(min=0)

        # psi = np.exp( (self.H(z, dp)-self.Hcut).clip(min=0) / (self.width*self.Hmax)) - 1
        # psi_norm = np.exp( (self.Hmax-0*self.Hcut) / (self.width*self.Hmax) ) - 1
        # psi = np.exp( -self.H(z, dp).clip(min=0)/(self.width*self.Hmax) ) - 1
        # psi_norm = np.exp( -self.Hmax/(self.width*self.Hmax) ) - 1

        psi = np.exp(self.H(z, dp).clip(min=0)/self.H0) - 1
        psi_norm = np.exp(self.Hmax/self.H0) - 1
        return psi/psi_norm
