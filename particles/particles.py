'''
Created on 17.10.2014
@author: Kevin Li, Michael Schenk, Adrian Oeftiger
'''

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.constants import c, e

from ..cobra_functions import stats as cp

class Particles(object):
    '''Contains the basic properties of a particle ensemble with
    their coordinate and conjugate momentum arrays, energy and the like.
    Designed to describe beams, electron clouds, ... '''

    def __init__(self, macroparticlenumber, particlenumber_per_mp, charge,
                 mass, circumference, gamma_reference,
                 coords_n_momenta_dict={}):
        '''The dictionary coords_n_momenta_dict contains the coordinate
        and conjugate momenta names and assigns to each the
        corresponding array.
        e.g.: coords_n_momenta_dict = {'x': array(..), 'xp': array(..)}
        '''
        self.macroparticlenumber = macroparticlenumber
        self.particlenumber_per_mp = particlenumber_per_mp

        self.charge = charge
        if not np.allclose(self.charge, e, atol=1e-24):
            raise NotImplementedError('PyHEADTAIL currently features many "e" '
                                      + 'all over the place, these need to be '
                                      + 'consistently replaced by '
                                      + '"self.charge"!')
        self.mass = mass

        self.circumference = circumference
        self.gamma = gamma_reference

        '''Dictionary of SliceSet objects which are retrieved via
        self.get_slices(slicer) by a client. Each SliceSet is recorded
        only once for a specific longitudinal state of Particles.
        Any longitudinal trackers (or otherwise modifying elements)
        should clean the saved SliceSet dictionary via
        self.clean_slices().
        '''
        self._slice_sets = {}

        '''Set of coordinate and momentum attributes of this Particles
        instance.
        '''
        self.coords_n_momenta = set()

        '''ID of particles in order to keep track of single entries
        in the coordinate and momentum arrays.
        '''
        self.id = np.arange(1, self.macroparticlenumber+1, dtype=np.int32)

        self.update(coords_n_momenta_dict)

    @property
    def intensity(self):
        return self.particlenumber_per_mp * self.macroparticlenumber
    @intensity.setter
    def intensity(self, value):
        self.particlenumber_per_mp = value / float(self.macroparticlenumber)

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
        self.gamma = np.sqrt(value**2 + 1)

    @property
    def p0(self):
        return self._p0
    @p0.setter
    def p0(self, value):
        self.gamma = value / (self.mass * self.beta * c)

    # @property
    # def theta(self):
    #     return self.z/self.ring_radius
    # @theta.setter
    # def theta(self, value):
    #     self.z = value*self.ring_radius

    # @property
    # def delta_E(self):
    #     return self.dp * self.beta*c*self.p0
    # @delta_E.setter
    # def delta_E(self, value):
    #     self.dp = value / (self.beta*c*self.p0)

    def get_coords_n_momenta_dict(self):
        '''Return a dictionary containing the coordinate and conjugate
        momentum arrays.
        '''
        return {coord: getattr(self, coord) for coord in self.coords_n_momenta}

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

    def update(self, coords_n_momenta_dict):
        '''Assigns the keys of the dictionary coords_n_momenta_dict as
        attributes to this Particles instance and puts the corresponding
        values. Pretty much the same as dict.update({...}) .
        Attention: overwrites existing coordinate / momentum attributes.
        '''
        if any(len(v) != self.macroparticlenumber for v in
               coords_n_momenta_dict.values()):
            raise ValueError("lengths of given phase space coordinate arrays" +
                             " do not coincide with self.macroparticlenumber.")
        for coord, array in coords_n_momenta_dict.items():
            setattr(self, coord, array.copy())
        self.coords_n_momenta.update(coords_n_momenta_dict.keys())

    def add(self, coords_n_momenta_dict):
        '''Add the coordinates and momenta with their according arrays
        to the attributes of the Particles instance (via
        self.update(coords_n_momenta_dict)). Does not allow existing
        coordinate or momentum attributes to be overwritten.
        '''
        if any(s in self.coords_n_momenta
               for s in coords_n_momenta_dict.keys()):
            raise ValueError("One or more of the specified coordinates or" +
                             " momenta already exist and cannot be added." +
                             " Use self.update(...) for this purpose.")
        self.update(coords_n_momenta_dict)

    # Statistics methods

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
        return (4*np.pi * cp.emittance(self.z, self.dp) * self.p0/e)
