'''
Created on 06.01.2014

@author: Kevin Li, Adrian Oeftiger
'''


import sys
import numpy as np
from scipy.constants import c, e, m_e, m_p

import cobra_functions.stats as cp
from generators import *


class Particles(object):

    # (n_macroparticles, n_particles_per_mp, charge, mass, gamma_reference, ring_radius, phase_space_generators)
    def __init__(self, n_macroparticles, charge, mass, gamma, n_particles_per_mp, phase_space_generators):
        """
        Initialises the bunch and distributes its particles via the
        given PhaseSpace generator instances for all planes.
        """

        self.n_macroparticles = n_macroparticles
        self.n_particles_per_mp = n_particles_per_mp
        self.same_size_for_all_MPs = True

        self.charge = charge
        self.mass = mass

        self.gamma = gamma # gamma_reference
        self.ring_radius = 0

        self.phase_space_coordinates_list = []
        for phase_space in phase_space_generators:
            phase_space.generate(self)
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)

    @classmethod
    def as_gaussian(cls, n_macroparticles, charge, mass, gamma, intensity,
                    alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                    beta_z, epsn_z, is_accepted=None, generator_seed=None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """

        n_particles_per_mp = intensity/n_macroparticles

        betagamma = np.sqrt(gamma**2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX, Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussianx = GaussianX.from_optics(alpha_x, beta_x, epsn_x, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        gaussiany = GaussianY.from_optics(alpha_y, beta_y, epsn_y, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        gaussianz = GaussianZ.from_optics(beta_z, epsn_z, p0, is_accepted,
                                          generator_seed=random_state.randint(sys.maxint))

        return cls(n_macroparticles, charge, mass, gamma, n_particles_per_mp,
                   (gaussianx, gaussiany, gaussianz))

    @classmethod
    def as_gaussian_in_bucket(cls, n_macroparticles, charge, gamma, intensity, mass,
                              alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                              sigma_z=None, epsn_z=None, rfsystem=None, generator_seed=None):

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
        rfbucket = RFBucket(StationaryExponential, rfsystem, sigma_z, epsn_z)

        return cls(n_macroparticles, charge, mass, gamma, n_particles_per_mp,
                   (gaussianx, gaussiany, rfbucket))

    @classmethod
    def as_gaussian_z(cls, n_macroparticles, charge, mass, gamma, intensity,
                      beta_z, epsn_z, is_accepted=None, generator_seed=None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """

        n_particles_per_mp = intensity/n_macroparticles

        betagamma = np.sqrt(gamma**2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX, Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussianz = GaussianZ.from_optics(beta_z, epsn_z, p0, is_accepted,
                                          generator_seed=random_state.randint(sys.maxint))

        return cls(n_macroparticles, charge, mass, gamma, n_particles_per_mp,
                   (gaussianz,))


    @classmethod
    def as_gaussian_theta(cls, n_macroparticles, charge, mass, gamma, intensity,
                          sigma_theta, sigma_dE, is_accepted=None, generator_seed=None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """

        n_particles_per_mp = intensity/n_macroparticles

        betagamma = np.sqrt(gamma**2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX, Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussiantheta = GaussianTheta(sigma_theta, sigma_dE, is_accepted,
                                      generator_seed=random_state.randint(sys.maxint))

        return cls(n_macroparticles, charge, mass, gamma, n_particles_per_mp,
                   (gaussiantheta,))


    @classmethod
    def as_uniform(cls, n_macroparticles, charge, gamma, intensity, mass,
                   xmin, xmax, ymin, ymax, zmin=0, zmax=0):

        n_particles_per_mp = intensity/n_macroparticles

        betagamma = np.sqrt(gamma ** 2 - 1)
        p0 = betagamma * mass * c

        uniformx = UniformX(xmin, xmax)
        uniformy = UniformY(xmin, xmax)
        uniformz = UniformZ(zmin, zmax)

        return cls(n_macroparticles, charge, mass, gamma, n_particles_per_mp,
                   [uniformx, uniformy, uniformz])

    @classmethod
    def as_import(cls, n_macroparticles, charge, mass, gamma, intensity,
                  x, xp, y, yp, z, dp):

        n_particles_per_mp = intensity/n_macroparticles

        importx = ImportX(x, xp)
        importy = ImportY(y, yp)
        importz = ImportZ(z, dp)

        return cls(n_macroparticles, charge, mass, gamma, n_particles_per_mp,
                   [importx, importy, importz])

    @property
    def intensity(self):
        if self.same_size_for_all_MPs:
            return self.n_particles_per_mp*self.n_macroparticles
        else:
            return  np.sum(self.n_particles_per_mp)

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


    @property
    def theta(self):
        return self.z/self.ring_radius
    @theta.setter
    def theta(self, value):
        self.z = value*self.ring_radius

    @property
    def delta_E(self):
        return self.dp * self.beta*c*self.p0
    @delta_E.setter
    def delta_E(self, value):
        self.dp = value / (self.beta*c*self.p0)


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


    '''
    Stats.
    '''
    def mean_x(self):
        return cp.mean(self.x)

    def mean_xp(self):
        return cp.mean(self.xp)
    
    def mean_y(self):
        return cp.mean(self.y)

    def mean_yp(self):
        return cp.mean(self.yp)
    
    def mean_z(self):
        return cp.mean(self.z)

    def mean_dp(self):
        return cp.mean(self.dp)

    def sigma_x(self):
        return cp.std(self.x)

    def sigma_y(self):
        return cp.std(self.y)

    def sigma_z(self):
        return cp.std(self.z)

    def sigma_dp(self):
        return cp.std(self.dp)

    def epsn_x(self):
        return cp.emittance(self.x, self.xp) * self.gamma * self.beta * 1e6

    def epsn_y(self):
        return cp.emittance(self.y, self.yp) * self.gamma * self.beta * 1e6

    def epsn_z(self):
        return (4 * np.pi * self.sigma_z() * self.sigma_dp() * self.p0 / self.charge)
