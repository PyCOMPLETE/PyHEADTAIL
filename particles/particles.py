'''
Created on 06.01.2014

@author: Kevin Li, Adrian Oeftiger
'''


import sys
import numpy as np
from scipy.constants import c, e, m_e, m_p

from cobra_functions import stats
from generators import *


class Particles(object):
    """Single bunch with 6D coordinates
    (x, x', y, y', z, \Delta p) where \Delta p = p - p0.
    These translate to
    self.x
    self.xp
    self.y
    self.yp
    self.z
    self.delta_p
    """

    def __init__(self, n_macroparticles, charge, gamma, mass, n_particles_per_mp, phase_space_generators):
        """
        Initialises the bunch and distributes its particles via the
        given PhaseSpace generator instances for all planes.
        """

        self.charge = charge
        self.gamma = gamma
        self.mass = mass

        self.n_macroparticles = n_macroparticles

        self.n_particles_per_mp = n_particles_per_mp

        for phase_space in phase_space_generators:
            phase_space.generate(self)

        self.same_size_for_all_MPs = True
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)

    @classmethod
    def as_gaussian(cls, n_macroparticles, charge, gamma, intensity, mass,
                    alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                    beta_z, epsn_z, is_accepted = None, generator_seed=None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """

        n_particles_per_mp = intensity/n_macroparticles

        betagamma = np.sqrt(gamma ** 2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX,Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussianx = GaussianX.from_optics(alpha_x, beta_x, epsn_x, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        gaussiany = GaussianY.from_optics(alpha_y, beta_y, epsn_y, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        gaussianz = GaussianZ.from_optics(beta_z, epsn_z, p0, is_accepted,
                                          generator_seed=random_state.randint(sys.maxint))

        return cls(n_macroparticles, charge, gamma, mass, n_particles_per_mp,
                   [gaussianx, gaussiany, gaussianz])

    @classmethod
    def as_gaussian_in_bucket(cls, n_macroparticles, charge, gamma, intensity, mass,
                              alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                              sigma_z, rfsystem, generator_seed=None):

        n_particles_per_mp = intensity/n_macroparticles

        betagamma = np.sqrt(gamma ** 2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX,Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussianx = GaussianX.from_optics(alpha_x, beta_x, epsn_x, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        gaussiany = GaussianY.from_optics(alpha_y, beta_y, epsn_y, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        rfbucket = GaussianZ.from_optics(beta_z, epsn_z, p0, is_accepted,
                                          generator_seed=random_state.randint(sys.maxint))

        return cls(n_macroparticles, charge, gamma, mass, n_particles_per_mp,
                   [gaussianx, gaussiany, rfbucket])

    @classmethod
    def as_uniformXY(cls, n_macroparticles, charge, gamma, intensity, mass,
                     xmin, xmax, ymin, ymax, zmin=0, zmax=0):

        n_particles_per_mp = intensity/n_macroparticles

        betagamma = np.sqrt(gamma ** 2 - 1)
        p0 = betagamma * mass * c

        uniformx = UniformX(xmin, xmax)
        uniformy = UniformY(xmin, xmax)
        uniformz = UniformZ(zmin, zmax)

        return cls(n_macroparticles, charge, gamma, mass, n_particles_per_mp,
                   [uniformx, uniformy, uniformz])

    @classmethod
    def as_uniformXYzeroZp(cls, n_macroparticles, charge, gamma, intensity, mass,
                            x_min, x_max, y_min, y_max):

        particles = cls.as_uniformXY(n_macroparticles, charge, gamma, intensity, mass,
                                     x_min, x_max, y_min, y_max)
        particles.zp = 0.*particles.x

        return particles

    @property
    def intensity(self):
        if self.same_size_for_all_MPs:
            return self.n_particles_per_mp*self.n_macroparticles
        else:
            return  np.sum(self.n_particles_per_mp)

    @property
    def beta(self):
        return np.sqrt(1 - 1. / self.gamma ** 2)
    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1 - value ** 2)

    @property
    def betagamma(self):
        return np.sqrt(self.gamma ** 2 - 1)
    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value ** 2 + 1)

    @property
    def p0(self):
        return self.mass * self.gamma * self.beta * c
    @p0.setter
    def p0(self, value):
        self.gamma = value / (self.mass * self.beta * c)

    @property
    def dp(self):
        return self.delta_p / self.p0
    @dp.setter
    def dp(self, value):
        self.delta_p = value * self.p0

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

    def compute_statistics(self):
        self.mean_x  = stats.mean(self.x)
        self.mean_xp = stats.mean(self.xp)
        self.mean_y  = stats.mean(self.y)
        self.mean_yp = stats.mean(self.yp)
        self.mean_z  = stats.mean(self.z)
        self.mean_dp = stats.mean(self.dp)

        self.sigma_x  = stats.std(self.x)
        self.sigma_y  = stats.std(self.y)
        self.sigma_z  = stats.std(self.z)
        self.sigma_dp = stats.std(self.dp)

        self.epsn_x = stats.emittance(self.x, self.xp) * self.gamma * self.beta * 1e6
        self.epsn_y = stats.emittance(self.y, self.yp) * self.gamma * self.beta * 1e6
        self.epsn_z = 4 * np.pi * self.sigma_z * self.sigma_dp * self.p0 / self.charge
