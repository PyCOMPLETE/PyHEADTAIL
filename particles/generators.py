from __future__ import division
'''
@file matching
@author Kevin Li, Adrian Oeftiger
@date February 2014
@brief Module for matching transverse and longitudinal distributions
@copyright CERN
'''


import numpy as np
from numpy.random import RandomState

from abc import ABCMeta, abstractmethod

from scipy.constants import c, e
from scipy.integrate import quad, dblquad


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

    def __init__(self, sigma_z, sigma_dp, is_accepted = None, generator_seed=None):
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
    def from_optics(cls, beta_z, epsn_z, p0, is_accepted = None,
                    generator_seed=None):
        """Initialise GaussianZ from the given optics functions.
        For the argument is_accepted see __init__.
        """
        sigma_z = np.sqrt(beta_z*epsn_z/(4*np.pi) * e/p0)
        # print sigma_z
        # exit(-1)
        sigma_dp = sigma_z / beta_z
        return cls(sigma_z, sigma_dp, is_accepted, generator_seed)

    def generate(self, beam):
        beam.z = self.sigma_z * self.random_state.randn(beam.n_macroparticles)
        beam.dp = self.sigma_dp * self.random_state.randn(beam.n_macroparticles)
        if self.is_accepted:
            self._redistribute(beam)

    def _redistribute(self, beam):
        n = beam.n_macroparticles
        for i in xrange(n):
            while not self.is_accepted(beam.z[i], beam.dp[i], beam):
                beam.z[i]  = self.sigma_z * self.random_state.randn(n)
                beam.dp[i] = self.sigma_dp * self.random_state.randn(n)

class UniformX(PhaseSpace):
    """Horizontal Uniform particle phase space distribution."""

    def __init__(self, x_min, x_max):
        """Initiates the horizontal beam coordinates
        to the given Uniform shape.
        """
        
        self.x_max = x_max
        self.x_min = x_min

    def generate(self, particles):
        particles.x = (self.x_max-self.x_min)*np.random.rand(particles.n_macroparticles)+self.x_min
        particles.xp = 0.*particles.x
        
        
class UniformY(PhaseSpace):
    """Horizontal Uniform particle phase space distribution."""

    def __init__(self, y_min, y_max):
        """Initiates the horizontal beam coordinates
        to the given Uniform shape.
        """
        
        self.y_max = y_max
        self.y_min = y_min

    def generate(self, particles):
        particles.y = (self.y_max-self.y_min)*np.random.rand(particles.n_macroparticles)+self.y_min
        particles.yp = 0.*particles.y
        
