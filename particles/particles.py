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

        assert(len(phase_space_generators) < 4)

        self.n_macroparticles = n_macroparticles

        self.charge = charge
        self.mass = mass
        self.gamma = gamma

        self.n_particles_per_mp = n_particles_per_mp

        for phase_space in phase_space_generators:
            phase_space.generate(self)

        self.same_size_for_all_MPs = True
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)
        # self._set_energies()

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
                   (gaussianx, gaussiany, gaussianz))

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
        rfbucket = RFBucket(sigma_z, rfsystem, )

        return cls(n_macroparticles, charge, gamma, mass, n_particles_per_mp,
                   (gaussianx, gaussiany, rfbucket))

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


    @classmethod
    def from_ASCII_file(cls, filename, n_macroparticles, charge, gamma, intensity, mass):
        """
        Load initial distribution from text file with 6 columns in order
        x, xp, y, yp, z, dp.
        """        
        data = np.loadtxt(filename)
        assert(n_macroparticles == data.shape[0])

        n_particles_per_mp = intensity/n_macroparticles

        from_file_x = ImportX(data[:,0], data[:,1])
        from_file_y = ImportY(data[:,2], data[:,3])
        from_file_z = ImportZ(data[:,4], data[:,5])

        return cls(n_macroparticles, charge, gamma, mass, n_particles_per_mp,
                   (from_file_x, from_file_y, from_file_z))        


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
