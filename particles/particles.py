'''
Created on 06.01.2014

@author: Kevin Li, Adrian Oeftiger
'''


import sys
import numpy as np
from scipy.constants import c, e, m_e, m_p

import cobra_functions.stats as cp
from trackers.rf_bucket import RFBucket
from generators import *


class Particles(object):

    '''Dictionary of SliceSet objects which are retrieved via
    self.get_slices(slicer) by a client. Each SliceSet is recorded
    only once for a specific longitudinal state of Particles.
    Any longitudinal trackers (or otherwise modifying elements)
    should clean the saved SliceSet dictionary via self.clean_slices().
    '''
    _slice_sets = {}

    def __init__(self, macroparticlenumber, particlenumber_per_mp, charge,
                 mass, circumference, gamma_reference,
                 phase_space_coordinates_dict):

        self.macroparticlenumber = macroparticlenumber
        self.particlenumber_per_mp = particlenumber_per_mp

        self.charge = charge
        self.mass = mass

        self.circumference = circumference
        self.gamma_reference = gamma_reference

        for k, v in phase_space_coordinates_dict.items():
            setattr(self, k, v)
        self.phase_space_coordinates_list = phase_space_coordinates_dict.keys()
        self.id = np.arange(1, self.macroparticlenumber+1, dtype=int)

        # Compatibility
        self.n_macroparticles = self.macroparticlenumber
        self.n_macroparticles_lost = 0
        self.n_particles_per_mp = self.particlenumber_per_mp
        self.same_size_for_all_MPs = True
        self.gamma = self.gamma_reference

        assert( all([len(v) == self.macroparticlenumber for v in phase_space_coordinates_dict.values()]) )


    def __init__2(self, macroparticlenumber, particlenumber_per_mp, charge, mass, circumference, gamma_reference, phase_space_coordinates_dict):

        self.macroparticlenumber = macroparticlenumber
        self.particlenumber_per_mp = particlenumber_per_mp

        self.charge = charge
        self.mass = mass

        self.ring_radius = circumference
        self.gamma = gamma_reference

        self.phase_space_coordinates_list = phase_space_coordinates_dict.keys()

        self.id = np.arange(1, self.macroparticlenumber+1, dtype=int)

    @classmethod
    def as_gaussian(cls, macroparticlenumber, charge, mass, gamma_reference, intensity,
                    alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                    beta_z, epsn_z, is_accepted=None, generator_seed=None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference**2 - 1)
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

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, 0, gamma_reference,
                   gaussianx, gaussiany, gaussianz)


    @classmethod
    def as_gaussian_bucket(cls, macroparticlenumber, intensity, charge, mass, ring_radius, gamma_reference,
                           alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                           alpha, p_increment, harmonic_list, voltage_list, phi_offset_list,
                           sigma_z=None, epsn_z=None, generator_seed=None):

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference ** 2 - 1)
        p0 = betagamma * mass * c

        rfbucket = RFBucket(2*np.pi*ring_radius, gamma_reference, alpha, p_increment, harmonic_list, voltage_list, phi_offset_list)

        # Generate seeds for GaussianX,Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussianx = GaussianX.from_optics(alpha_x, beta_x, epsn_x, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        gaussiany = GaussianY.from_optics(alpha_y, beta_y, epsn_y, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        rfbucket = RFBucketMatcher(StationaryExponential, rfbucket, sigma_z, epsn_z)

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, ring_radius, gamma_reference,
                   gaussianx, gaussiany, rfbucket)


    @classmethod
    def as_gaussian_bucket_match(cls, macroparticlenumber, intensity, charge, mass, ring_radius, gamma_reference,
                                 alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, rfbucket,
                                 sigma_z=None, epsn_z=None, generator_seed=None):

        self = cls.as_gaussian_bucket(macroparticlenumber, intensity, charge, mass, ring_radius, gamma_reference,
                                      alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                                      rfbucket.alpha0, rfbucket.p_increment, rfbucket.h, rfbucket.V, rfbucket.dphi,
                                      sigma_z, epsn_z, generator_seed)
        rfbucket.gamma_reference = self.get_gamma

        return self


    @classmethod
    def as_gaussian_in_bucket(cls, macroparticlenumber, charge, gamma_reference, intensity, mass,
                              alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                              sigma_z=None, epsn_z=None, rfbucket=None, generator_seed=None):

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference ** 2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX,Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussianx = GaussianX.from_optics(alpha_x, beta_x, epsn_x, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        gaussiany = GaussianY.from_optics(alpha_y, beta_y, epsn_y, betagamma,
                                          generator_seed=random_state.randint(sys.maxint))
        rfbucket = RFBucketMatcher(StationaryExponential, rfbucket, sigma_z, epsn_z)

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, 0, gamma_reference,
                   gaussianx, gaussiany, rfbucket)


    # @classmethod
    # def as_gaussian_linear_match(cls, macroparticlenumber, intensity, charge, mass, ring_radius, gamma_reference,
    #                              epsn_x, epsn_y, epsn_z, transverse_map, longitudinal_map, generator_seed=None):
    #     self = cls.as_gaussian(cls, macroparticlenumber, charge, mass, gamma_reference, intensity,
    #                            transverse_map.alpha_x, transverse_map.beta_x, epsn_x,
    #                            transverse_map.alpha_y, transverse_map.beta_y, epsn_y,
    #                            longitudinal_map.beta_z, epsn_z, is_accepted=None, generator_seed=None)

    #     return self


    @classmethod
    def as_gaussian_z(cls, macroparticlenumber, charge, mass, gamma_reference, intensity,
                      beta_z, epsn_z, is_accepted=None, generator_seed=None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference**2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX, Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussianz = GaussianZ.from_optics(beta_z, epsn_z, p0, is_accepted,
                                          generator_seed=random_state.randint(sys.maxint))

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, 0, gamma_reference,
                   gaussianz)


    @classmethod
    def as_gaussian_theta(cls, macroparticlenumber, charge, mass, gamma_reference, intensity,
                          sigma_theta, sigma_dE, is_accepted=None, generator_seed=None):
        """Initialises a Gaussian bunch from the given optics functions.
        For the argument is_accepted cf. generators.Gaussian_Z .
        """

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference**2 - 1)
        p0 = betagamma * mass * c

        # Generate seeds for GaussianX, Y and Z.
        random_state = RandomState()
        random_state.seed(generator_seed)

        gaussiantheta = GaussianTheta(sigma_theta, sigma_dE, is_accepted,
                                      generator_seed=random_state.randint(sys.maxint))

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, 0, gamma_reference,
                   gaussiantheta)


    @classmethod
    def as_uniform(cls, macroparticlenumber, charge, gamma_reference, intensity, mass,
                   xmin, xmax, ymin, ymax, zmin=0, zmax=0):

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference ** 2 - 1)
        p0 = betagamma * mass * c

        uniformx = UniformX(xmin, xmax)
        uniformy = UniformY(xmin, xmax)
        uniformz = UniformZ(zmin, zmax)

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, 0, gamma_reference,
                   uniformx, uniformy, uniformz)


    @classmethod
    def as_import(cls, macroparticlenumber, charge, mass, gamma_reference, intensity,
                  x, xp, y, yp, z, dp):

        particlenumber_per_mp = intensity/macroparticlenumber

        importx = ImportX(x, xp)
        importy = ImportY(y, yp)
        importz = ImportZ(z, dp)

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, 0, gamma_reference,
                   importx, importy, importz)


    @property
    def intensity(self):
        if self.same_size_for_all_MPs:
            return self.n_particles_per_mp*self.n_macroparticles
        else:
            return  np.sum(self.n_particles_per_mp)

    def get_ring_radius(self): return self.ring_radius

    def get_gamma(self): return self.gamma

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


    def get_slices(self, slicer):
        '''For the given Slicer, the last SliceSet is returned.
        If there is no SliceSet recorded (i.e. the longitudinal
        state has changed), a new SliceSet is requested from the Slicer
        via Slicer.slice(self) and stored for future reference.
        '''
        if slicer not in self._slice_sets:
            self._slice_sets[slicer] = slicer.slice(self)
        return self._slice_sets[slicer]

    def clean_slices(self):
        '''Erases the SliceSet records of this Particles instance.
        Any longitudinal trackers (or otherwise modifying elements)
        should use this method to clean the recorded SliceSet objects.
        '''
        del self._slice_sets

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
        return cp.emittance(self.x, self.xp) * self.betagamma

    def epsn_y(self):
        return cp.emittance(self.y, self.yp) * self.betagamma

    def epsn_z(self):
        return (4 * np.pi * self.sigma_z() * self.sigma_dp() * self.p0 / self.charge)
