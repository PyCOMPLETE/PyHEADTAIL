'''
Created on 06.01.2014

@author: Kevin Li, Adrian Oeftiger
'''


import numpy as np
import exceptions
from generators import GaussianX, GaussianY, GaussianZ

from scipy.constants import c, e, m_p


# re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
# rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p

class ArgumentError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Particles(object):
    """
    Single bunch with 6D coordinates
    (x, x', y, y', z, \delta p) where \delta p = (p - p0) / p0.
    These translate to
    self.x
    self.xp
    self.y
    self.yp
    self.z
    self.dp
    """

    def __init__(self, charge, gamma, intensity, mass, *phase_space_generators):
        """
        Initialises the bunch and distributes its particles via the
        given PhaseSpace generator instances (minimum 1) for both the
        transverse and longitudinal plane.
        """
        if len(phase_space_generators) < 1:
            raise ArgumentError("Particles needs at least one entry " +
                                "for phase_space_generators.")

        self.charge = charge
        self.mass = mass
        self.intensity = intensity
        self.gamma = gamma

        for phase_space in phase_space_generators:
            phase_space.generate(self)

        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)

    @classmethod
    def as_gaussian(cls, n_macroparticles, charge, gamma, intensity, mass,
                    alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                    sigma_z, sigma_dp, is_accepted = None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """
        betagamma = np.sqrt(gamma ** 2 - 1)
        #p0 = betagamma * mass * c
        gaussianx = GaussianX.from_optics(
                        n_macroparticles, alpha_x, beta_x, epsn_x, betagamma)
        gaussiany = GaussianY.from_optics(
                        n_macroparticles, alpha_y, beta_y, epsn_y, betagamma)
        gaussianz = GaussianZ(n_macroparticles, sigma_z, sigma_dp, is_accepted)
        return cls(charge, gamma, intensity, mass,
                   gaussianx, gaussiany, gaussianz)

    @property
    def n_macroparticles(self):
        return len(self.z)

    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self._gamma**-2)
        self._betagamma = np.sqrt(self._gamma**2 - 1)
        self._p0 = self._betagamma * self.mass * c

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1 - value ** 2)

    @property
    def betagamma(self):
        return self._betagamma
    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value ** 2 + 1)

    @property
    def p0(self):
        return self._p0
    @p0.setter
    def p0(self, value):
        self.gamma = value / (self.mass * self.beta * c)

    def sort_particles(self):
        # update the number of lost particles
        self.n_macroparticles_lost = (self.n_macroparticles -
                                      np.count_nonzero(self.id))

        # sort particles according to z (this is needed for correct
        # functioning of bunch.compute_statistics)
        if self.n_macroparticles_lost:
            # place lost particles at the end of the array
            z_argsorted = np.lexsort((self.z, -np.sign(self.id)))
        else:
            z_argsorted = np.argsort(self.z)

        self.x  = self.x.take(z_argsorted)
        self.xp = self.xp.take(z_argsorted)
        self.y  = self.y.take(z_argsorted)
        self.yp = self.yp.take(z_argsorted)
        self.z  = self.z.take(z_argsorted)
        self.dp = self.dp.take(z_argsorted)
        self.id = self.id.take(z_argsorted)
