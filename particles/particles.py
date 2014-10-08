'''
Created on 04.09.2014
@author: Kevin Li, Adrian Oeftiger
'''


import sys
import numpy as np
from numpy.random import normal, uniform, RandomState
from scipy.constants import c, e, m_e, m_p

from ..cobra_functions import stats as cp
from ..trackers.rf_bucket import RFBucket
from generators import RFBucketMatcher


class Particles(object):

    def __init__(self, macroparticlenumber, particlenumber_per_mp, charge,
                 mass, circumference, gamma_reference,
                 phase_space_coordinates_dict):

        self.macroparticlenumber = macroparticlenumber
        self.particlenumber_per_mp = particlenumber_per_mp

        self.charge = charge
        self.mass = mass

        self.circumference = circumference
        self.gamma_reference = gamma_reference

        '''Dictionary of SliceSet objects which are retrieved via
        self.get_slices(slicer) by a client. Each SliceSet is recorded
        only once for a specific longitudinal state of Particles.
        Any longitudinal trackers (or otherwise modifying elements)
        should clean the saved SliceSet dictionary via
        self.clean_slices().
        '''
        self._slice_sets = {}

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


    @classmethod
    def as_gaussian(cls, macroparticlenumber, intensity, charge, mass,
                    circumference, gamma_reference, sigma_x, sigma_xp,
                    sigma_y, sigma_yp, sigma_z, sigma_dp, generator_seed=None):

        particlenumber_per_mp = intensity/macroparticlenumber

        if generator_seed:
            random_state = RandomState()
            random_state.seed(generator_seed)

        x  = normal(0, sigma_x, macroparticlenumber)
        xp = normal(0, sigma_xp, macroparticlenumber)
        y  = normal(0, sigma_y, macroparticlenumber)
        yp = normal(0, sigma_yp, macroparticlenumber)
        z  = normal(0, sigma_z, macroparticlenumber)
        dp = normal(0, sigma_dp, macroparticlenumber)

        phase_space_coordinates_dict = {'x': x, 'xp': xp, 'y': y, 'yp': yp, 'z': z, 'dp': dp}

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, circumference, gamma_reference,
                   phase_space_coordinates_dict)


    @classmethod
    def as_gaussian_linear(cls, macroparticlenumber, intensity, charge, mass,
                           circumference, gamma_reference,
                           alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                           beta_z, epsn_z,
                           is_accepted=None, generator_seed=None):

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference**2 - 1)
        p0 = betagamma * mass * c

        if generator_seed:
            random_state = RandomState()
            random_state.seed(generator_seed)

        x  = normal(0, np.sqrt(epsn_x/betagamma * beta_x), macroparticlenumber)
        xp = normal(0, np.sqrt(epsn_x/betagamma / beta_x), macroparticlenumber)
        y  = normal(0, np.sqrt(epsn_y/betagamma * beta_y), macroparticlenumber)
        yp = normal(0, np.sqrt(epsn_y/betagamma / beta_y), macroparticlenumber)
        if not is_accepted:
            z  = normal(0, np.sqrt(epsn_z*e/p0 * beta_z), macroparticlenumber)
            dp = normal(0, np.sqrt(epsn_z*e/p0 / beta_z), macroparticlenumber)

        phase_space_coordinates_dict = {'x': x, 'xp': xp, 'y': y, 'yp': yp, 'z': z, 'dp': dp}

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, circumference, gamma_reference,
                   phase_space_coordinates_dict)


    @classmethod
    def as_gaussian_linear_matched(cls, macroparticlenumber, intensity, charge, mass, circumference, gamma_reference,
                                   transverse_map=None, longitudinal_map=None,
                                   epsn_x=None, epsn_y=None, sigma_z=None, epsn_z=None, generator_seed=None):

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference**2 - 1)

        if generator_seed:
            random_state = RandomState()
            random_state.seed(generator_seed)

        phase_space_coordinates_dict = {}
        if transverse_map:
            x  = normal(0, np.sqrt(epsn_x/betagamma * transverse_map.beta_x), macroparticlenumber)
            xp = normal(0, np.sqrt(epsn_x/betagamma / transverse_map.beta_x), macroparticlenumber)
            y  = normal(0, np.sqrt(epsn_y/betagamma * transverse_map.beta_y), macroparticlenumber)
            yp = normal(0, np.sqrt(epsn_y/betagamma / transverse_map.beta_y), macroparticlenumber)
            phase_space_coordinates_dict['x']  = x
            phase_space_coordinates_dict['xp'] = xp
            phase_space_coordinates_dict['y']  = y
            phase_space_coordinates_dict['yp'] = yp
        if longitudinal_map:
            beta_z = np.abs(longitudinal_map.eta)*circumference/(2*np.pi)/longitudinal_map.Qs
            if sigma_z and not epsn_z:
                sigma_z = sigma_z
            elif not sigma_z and epsn_z:
                sigma_z = np.sqrt(epsn_z*beta_z/(4*np.pi) * e/p0)
            else:
                raise TypeError('***ERROR: at least and at most one of sigma_z and epsn_z to be given!')
            z  = normal(0, sigma_z, macroparticlenumber)
            dp = normal(0, sigma_z/beta_z, macroparticlenumber)
            phase_space_coordinates_dict['z']  = z
            phase_space_coordinates_dict['dp'] = dp

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, circumference, gamma_reference,
                   phase_space_coordinates_dict)


    @classmethod
    def as_gaussian_bucket_matched(cls, macroparticlenumber, intensity, charge, mass, circumference, gamma_reference,
                                   transverse_map=None, longitudinal_map=None,
                                   epsn_x=None, epsn_y=None, sigma_z=None, epsn_z=None, generator_seed=None):

        particlenumber_per_mp = intensity/macroparticlenumber

        betagamma = np.sqrt(gamma_reference**2 - 1)

        if generator_seed:
            random_state = RandomState()
            random_state.seed(generator_seed)

        phase_space_coordinates_dict = {}
        if transverse_map:
            x  = normal(0, np.sqrt(epsn_x/betagamma * transverse_map.beta_x), macroparticlenumber)
            xp = normal(0, np.sqrt(epsn_x/betagamma / transverse_map.beta_x), macroparticlenumber)
            y  = normal(0, np.sqrt(epsn_y/betagamma * transverse_map.beta_y), macroparticlenumber)
            yp = normal(0, np.sqrt(epsn_y/betagamma / transverse_map.beta_y), macroparticlenumber)
            phase_space_coordinates_dict['x']  = x
            phase_space_coordinates_dict['xp'] = xp
            phase_space_coordinates_dict['y']  = y
            phase_space_coordinates_dict['yp'] = yp
        if longitudinal_map:
            z, dp, psi, linedensity = RFBucketMatcher(StationaryExponential, longitudinal_map, sigma_z, epsn_z).generate(macroparticlenumber)
            phase_space_coordinates_dict['z']  = z
            phase_space_coordinates_dict['dp'] = dp

        self = cls(macroparticlenumber, particlenumber_per_mp, charge, mass, circumference, gamma_reference,
                   phase_space_coordinates_dict)
        longitudinal_map.circumference = self.get_circumference
        longitudinal_map.gamma_reference = self.get_gamma_reference

        self.psi = psi
        self.linedensity = linedensity

        return self


    @classmethod
    def as_uniform(cls, macroparticlenumber, intensity, charge, mass, circumference, gamma_reference,
                   xextent, yextent, zextent):

        particlenumber_per_mp = intensity/macroparticlenumber

        x  = uniform(-xextent, xextent, macroparticlenumber)
        xp = np.zeros(macroparticlenumber)
        y  = normal(-yextent, yextent, macroparticlenumber)
        yp = np.zeros(macroparticlenumber)
        z  = uniform(-zextent, zextent, macroparticlenumber)
        dp = np.zeros(macroparticlenumber)

        phase_space_coordinates_dict = {'x': x, 'xp': xp, 'y': y, 'yp': yp, 'z': z, 'dp': dp}

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, circumference, gamma_reference,
                   phase_space_coordinates_dict)


    @classmethod
    def as_import(cls, macroparticlenumber, intensity, charge, mass, circumference, gamma_reference,
                  phase_space_coordinates_dict):

        particlenumber_per_mp = intensity/macroparticlenumber

        return cls(macroparticlenumber, particlenumber_per_mp, charge, mass, circumference, gamma_reference,
                   phase_space_coordinates_dict)


    @property
    def intensity(self):
        if self.same_size_for_all_MPs:
            return self.n_particles_per_mp*self.n_macroparticles
        else:
            return  np.sum(self.n_particles_per_mp)

    def get_circumference(self): return self.circumference

    def get_gamma_reference(self): return self.gamma

    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma**-2)
        self._betagamma = np.sqrt(self.gamma**2 - 1)
        self._p0 = self.betagamma * self.mass * c

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
        self._slice_sets = {}

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
