'''
@author Kevin Li, Adrian Oeftiger, Stefan Hegglin
@date 16.06.2017
@brief module for matching longitudinal particle distributions to an RFBucket instance
'''

from __future__ import division, print_function

import numpy as np
from scipy.optimize import brentq, newton
from scipy.integrate import fixed_quad
from scipy.constants import e, c

from ..cobra_functions import pdf_integrators_2d as integr

from . import Printing

from functools import partial

from abc import abstractmethod

class RFBucketMatcher(Printing):

    integrationmethod = ['quad', 'cumtrapz'][0]
    def get_moment_integrators(self):
        '''Return moment integrators from
        cobra_functions.pdf_integrators_2d according to the chosen
        self.integrationmethod. Allows to change integration method
        for RFBucket matching.
        '''
        zero = getattr(integr, 'compute_zero_' + self.integrationmethod)
        mean = getattr(integr, 'compute_mean_' + self.integrationmethod)
        var = getattr(integr, 'compute_var_' + self.integrationmethod)
        cov = getattr(integr, 'compute_cov_' + self.integrationmethod)
        return zero, mean, var, cov

    def __init__(self, rfbucket, distribution_type=None, sigma_z=None,
                 epsn_z=None, verbose_regeneration=False, psi=None,
                 *args, **kwargs):

        if psi is not None:
            self.warns('\n\n*** DEPRECATED: keyword argument "psi" will '
                       'be removed in a future PyHEADTAIL release!')
            if distribution_type is not None:
                raise TypeError('RFBucketMatcher accepts either '
                                'distribution_type or psi as argument. ')
        else:
            if distribution_type is None:
                raise TypeError(
                    '__init__() takes at least 2 arguments (1 given)\n'
                    '--> kwarg psi has been renamed to distribution_type.')

        self.rfbucket = rfbucket
        hamiltonian = partial(rfbucket.hamiltonian, make_convex=True)
        hmax = rfbucket.h_sfp(make_convex=True)

        self.psi_object = distribution_type(hamiltonian, hmax)
        self.psi = self.psi_object.function

        self.verbose_regeneration = verbose_regeneration

        if sigma_z and not epsn_z:
            self.variable = sigma_z
            self.psi_for_variable = self.psi_for_bunchlength_newton_method
        elif not sigma_z and epsn_z:
            self.variable = epsn_z
            self.psi_for_variable = self.psi_for_emittance_newton_method
        else:
            raise ValueError("Can not generate mismatched matched "
                             "distribution! (Don't provide both sigma_z "
                             "and epsn_z!)")

    ### analytic matching methods:

    def psi_for_emittance_newton_method(self, epsn_z):
        # Maximum emittance
        self.psi_object.H0 = self.rfbucket.guess_H0(
            self.rfbucket.circumference, from_variable='sigma')
        epsn_max = self._compute_emittance(self.rfbucket, self.psi)
        if epsn_z > epsn_max:
            self.warns('Given RMS emittance does not fit into bucket. '
                       'Using (maximum) full bucket emittance ' +
                       str(epsn_max*0.99) + 'eV s instead.')
            epsn_z = epsn_max*0.99
        self.prints('*** Maximum RMS emittance ' + str(epsn_max) + 'eV s.')

        def error_from_target_epsn(ec):
            self.psi_object.H0 = self.rfbucket.guess_H0(
                ec, from_variable='epsn')
            emittance = self._compute_emittance(self.rfbucket, self.psi)

            if np.isnan(emittance): raise ValueError

            self.prints('... distance to target emittance: ' +
                        '{:.2e}'.format(emittance-epsn_z))

            return emittance-epsn_z

        try:
            ec_bar = brentq(error_from_target_epsn, epsn_z/100, 2*epsn_max,
                            rtol=1e-5)
        except ValueError:
            self.warns(
                'RFBucketMatcher: failed to converge with Brent method, '
                'continuing with Newton-Raphson method.')
            ec_bar = newton(error_from_target_epsn, epsn_z, tol=1e-5)

        self.psi_object.H0 = self.rfbucket.guess_H0(
            ec_bar, from_variable='epsn')
        emittance = self._compute_emittance(self.rfbucket, self.psi)
        self.prints('--> Emittance: ' + str(emittance))
        sigma = self._compute_sigma(self.rfbucket, self.psi)
        self.prints('--> Bunch length: ' + str(sigma))

    def psi_for_bunchlength_newton_method(self, sigma):
        # Maximum bunch length
        self.psi_object.H0 = self.rfbucket.guess_H0(
            self.rfbucket.circumference, from_variable='sigma')
        sigma_max = self._compute_sigma(self.rfbucket, self.psi)
        if sigma > sigma_max:
            self.warns('Given RMS bunch length does not fit into bucket. '
                       'Using (maximum) full bucket RMS bunch length ' +
                       str(sigma_max*0.99) + 'm instead.')
            sigma = sigma_max*0.99
        self.prints('*** Maximum RMS bunch length ' + str(sigma_max) + 'm.')

        def error_from_target_sigma(sc):
            '''Width for bunch length'''
            self.psi_object.H0 = self.rfbucket.guess_H0(
                sc, from_variable='sigma')
            length = self._compute_sigma(self.rfbucket, self.psi)

            if np.isnan(length): raise ValueError

            self.prints('... distance to target bunch length: ' +
                        '{:.4e}'.format(length-sigma))

            return length-sigma

        try:
            sc_bar = brentq(error_from_target_sigma, sigma/100, 2*sigma_max,
                            rtol=1e-5)
        except ValueError:
            self.warns(
                'RFBucketMatcher: failed to converge with Brent method, '
                'continuing with Newton-Raphson method.')
            sc_bar = newton(error_from_target_sigma, sigma, tol=1e-5)

        self.psi_object.H0 = self.rfbucket.guess_H0(
            sc_bar, from_variable='sigma')
        sigma = self._compute_sigma(self.rfbucket, self.psi)
        self.prints('--> Bunch length: ' + str(sigma))
        emittance = self._compute_emittance(self.rfbucket, self.psi)
        self.prints('--> Emittance: ' + str(emittance))

    def linedensity(self, xx, quad_type=fixed_quad):
        L = []
        try:
            L = np.array([quad_type(lambda y: self.psi(x, y), 0,
                                    self.rfbucket.separatrix(x))[0]
                          for x in xx])
        except TypeError:
            L = quad_type(lambda y: self.psi(xx, y), 0,
                          self.rfbucket.separatrix(xx))[0]
        L = np.array(L)

        return 2*L

    def generate(self, macroparticlenumber, cutting_margin=0):
        '''Generate a 2d phase space of n_particles particles randomly distributed
        according to the particle distribution function psi within the region
        [xmin, xmax, ymin, ymax].
        '''
        self.psi_for_variable(self.variable)

        xmin, xmax = self.rfbucket.z_left, self.rfbucket.z_right
        ymin = -self.rfbucket.dp_max(self.rfbucket.z_right)
        ymax = -ymin

        # rejection sampling
        uniform = np.random.uniform
        n_gen = macroparticlenumber
        u = uniform(low=xmin, high=xmax, size=n_gen)
        v = uniform(low=ymin, high=ymax, size=n_gen)
        s = uniform(size=n_gen)

        def mask_out(s, u, v):
            return s >= self.psi(u, v)

        if cutting_margin:
            mask_out_nocut = mask_out

            def mask_out(s, u, v):
                return np.logical_or(
                    mask_out_nocut(s, u, v),
                    ~self.rfbucket.is_in_separatrix(u, v, cutting_margin))

        # masked_out = ~(s<self.psi(u, v))
        masked_out = mask_out(s, u, v)
        while np.any(masked_out):
            masked_ids = np.where(masked_out)[0]
            n_gen = len(masked_ids)
            u[masked_out] = uniform(low=xmin, high=xmax, size=n_gen)
            v[masked_out] = uniform(low=ymin, high=ymax, size=n_gen)
            s[masked_out] = uniform(size=n_gen)
            # masked_out = ~(s<self.psi(u, v))
            masked_out[masked_ids] = mask_out(
                s[masked_out], u[masked_out], v[masked_out]
            )
            if self.verbose_regeneration:
                self.prints(
                    'Thou shalt not give up! :-) '
                    'Regenerating {0} macro-particles...'.format(n_gen))

        return u, v, self.psi, self.linedensity

    def _compute_sigma(self, rfbucket, psi):
        z_left = rfbucket.z_left
        z_right = rfbucket.z_right
        zero, mean, var, cov = self.get_moment_integrators()

        var_x = var(self.psi, lambda x: -rfbucket.separatrix(x),
                    rfbucket.separatrix, z_left, z_right,
                    direction='x') # x means z direction

        return np.sqrt(var_x)

    def _compute_emittance(self, rfbucket, psi):
        z_left = rfbucket.z_left
        z_right = rfbucket.z_right
        zero, mean, var, cov = self.get_moment_integrators()

        var_x, cov_xy, var_y = cov(
            self.psi, lambda x: -rfbucket.separatrix(x),
            rfbucket.separatrix, z_left, z_right)

        return (np.sqrt(var_x*var_y - cov_xy**2) *
                4*np.pi*rfbucket.p0/np.abs(rfbucket.charge))


class StationaryDistribution(object):
    def __init__(self, H, Hmax=None, Hcut=0, H0=1):
        self.H = H
        self.H0 = H0
        if not Hmax:
            self.Hmax = H(0, 0)
        else:
            self.Hmax = Hmax
        self.Hcut = Hcut

    @abstractmethod
    def _psi(self, H):
        '''Define the distribution value for the given H, the output
        lies in the interval [0,1]. This is the central function to
        be implemented by stationary distributions.
        '''
        pass

    def function(self, z, dp):
        psi = self._psi(self.H(z, dp).clip(min=self.Hcut))
        norm = self._psi(self.Hmax)
        return psi / norm

class ThermalDistribution(StationaryDistribution):
    '''Thermal Boltzmann distribution \psi ~ \exp(-H/H0).
    For a quadratic harmonic oscillator Hamiltonian this gives the
    bi-Gaussian phase space distribution.
    '''
    def _psi(self, H):
        # convert from convex Hamiltonian
        # (SFP being the maximum and the separatrix having zero value)
        # to conventional literature scale (zero-valued minimum at SFP)
        Hsep = self.Hcut + self.Hmax
        Hn = Hsep - H
        # f(Hn) - f(Hsep)
        return np.exp(-Hn / self.H0) - np.exp(-Hsep / self.H0)

class QGaussianDistribution(StationaryDistribution):
    '''Specific Tsallis q-Gaussian distribution for q=3/5 for now,
    leading to \psi ~ (1 - H/H0)^2, this may be generalised.
    '''
    n = 2

    def _psi(self, H):
        # convert from convex Hamiltonian
        # (SFP being the maximum and the separatrix having zero value)
        # to conventional scale (zero-valued minimum at SFP)
        Hsep = self.Hcut + self.Hmax
        Hn = Hsep - H
        dist = (1 - Hn / self.H0).clip(min=self.Hcut)**self.n
        return dist.clip(min=self.Hcut)

class ParabolicDistribution(StationaryDistribution):
    '''The parabolic profile distribution is a specific case of the
    present implementation of the q-Gaussian distribution for n = 1/2,
    \psi ~ \sqrt(1 - H/H0).
    For a quadratic harmonic oscillator Hamiltonian this distribution
    provides a parabolic line density.
    '''
    def _psi(self, H):
        # convert from convex Hamiltonian
        # (SFP being the maximum and the separatrix having zero value)
        # to conventional scale (zero-valued minimum at SFP)
        Hsep = self.Hcut + self.Hmax
        Hn = Hsep - H
        return np.sqrt((1 - Hn / self.H0).clip(min=self.Hcut))

class WaterbagDistribution(StationaryDistribution):
    '''The waterbag distribution has a constant Hamiltonian distribution
    until a cutoff, \psi ~ \Theta(H - H0) with \Theta the Heaviside
    step function.
    '''
    def _psi(self, H):
        # convert from convex Hamiltonian
        # (SFP being the maximum and the separatrix having zero value)
        # to conventional scale (zero-valued minimum at SFP)
        Hsep = self.Hcut + self.Hmax
        Hn = Hsep - H
        return np.piecewise(Hn, [Hn <= self.H0, Hn > self.H0], [1./self.H0, 0])
