'''
Created on 17.10.2014
@author: Kevin Li, Michael Schenk, Adrian Oeftiger
'''

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.constants import c, e, m_p

from ..cobra_functions import stats as cp
from . import Printing

arange = np.arange
mean = np.mean
std = cp.std

class Particles(Printing):
    '''Contains the basic properties of a particle ensemble with
    their coordinate and conjugate momentum arrays, energy and the like.
    Designed to describe beams, electron clouds, ... '''

    def __init__(self, macroparticlenumber, particlenumber_per_mp, charge,
                 mass, circumference, gamma, coords_n_momenta_dict={}):
        '''The dictionary coords_n_momenta_dict contains the coordinate
        and conjugate momenta names and assigns to each the
        corresponding array.
        e.g.: coords_n_momenta_dict = {'x': array(..), 'xp': array(..)}
        '''
        self.macroparticlenumber = macroparticlenumber
        self.particlenumber_per_mp = particlenumber_per_mp

        self.charge = charge
        if not np.allclose(self.charge, e): #, atol=1e-24):
            self.warns('PyHEADTAIL currently features many "e" ' +
                       'in the various modules, these need to be ' +
                       'consistently replaced by "beam.charge"!')
        self.charge_per_mp = particlenumber_per_mp * charge
        self.mass = mass
        if not np.allclose(self.charge, m_p): #, atol=1e-24):
            self.warns('PyHEADTAIL currently features many "m_p" ' +
                       'in the various modules, these need to be ' +
                       'consistently replaced by "beam.mass"!')

        self.circumference = circumference
        self.gamma = gamma

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
        self.id = arange(1, self.macroparticlenumber + 1, dtype=np.int32)

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

    @property
    def z_beamframe(self):
        return self.z * self.gamma


    def get_coords_n_momenta_dict(self):
        '''Return a dictionary containing the coordinate and conjugate
        momentum arrays.
        '''
        return {coord: getattr(self, coord) for coord in self.coords_n_momenta}

    def get_slices(self, slicer, *args, **kwargs):
        '''For the given Slicer, the last SliceSet is returned.
        If there is no SliceSet recorded (i.e. the longitudinal
        state has changed), a new SliceSet is requested from the Slicer
        via Slicer.slice(self) and stored for future reference.

        Arguments:
        - statistics=True attaches mean values, standard deviations
        and emittances to the SliceSet for all planes.
        - statistics=['mean_x', 'sigma_dp', 'epsn_z'] only adds the
        listed statistics values (can be used to save time).
        Valid list entries are all statistics functions of Particles.

        Note: Requesting statistics after calling get_slices w/o
        the statistics keyword results in creating a new SliceSet!
        '''
        if slicer not in self._slice_sets or kwargs.get('statistics'):
            self._slice_sets[slicer] = slicer.slice(self, *args, **kwargs)
        # # try to save time by allowing longitudinal statistics to
        # # simply be added to the existing SliceSet:
        # # (transverse statistics may change even if longitudinal stays
        # # the same between two SliceSet requesting elements)
        # elif 'statistics' in kwargs:
        #     if (any([ '_x' in stats for stats in kwargs['statistics']]) or
        #             any([ '_y' in stats for stats in kwargs['statistics']])):
        #         self._slice_sets[slicer] = slicer.slice(self, *args, **kwargs)
        #     else:
        #         slicer.add_statistics(self._slice_sets[slicer], self,
        #                               kwargs['statistics'])
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

    def sort_for(self, attr):
        '''Sort the named particle attribute (coordinate / momentum)
        array and reorder all particles accordingly.
        '''
        permutation = np.argsort(getattr(self, attr))
        self.reorder(permutation)

    def reorder(self, permutation, except_for_attrs=[]):
        '''Reorder all particle coordinate and momentum arrays
        (in self.coords_n_momenta) and ids except for except_for_attrs
        according to the given index array permutation.
        '''
        to_be_reordered = ['id'] + list(self.coords_n_momenta)
        for attr in to_be_reordered:
            if attr in except_for_attrs:
                continue
            reordered = getattr(self, attr)[permutation]
            setattr(self, attr, reordered)
            
    def __add__(self, other):
        '''Merges two beams.
		'''
        #print 'Checks still to be added!!!!!!'

        self_coords_n_momenta_dict = self.get_coords_n_momenta_dict()
        other_coords_n_momenta_dict = other.get_coords_n_momenta_dict()
        
        result = Particles(macroparticlenumber=self.macroparticlenumber+other.macroparticlenumber, 
                    particlenumber_per_mp=self.particlenumber_per_mp, charge=self.charge,
					mass=self.mass, circumference=self.circumference, gamma=self.gamma, coords_n_momenta_dict={})
        
        
        for coord in self_coords_n_momenta_dict.keys():
            #setattr(result, coord, np.concatenate((self_coords_n_momenta_dict[coord].copy(), other_coords_n_momenta_dict[coord].copy())))
            result.update({coord: np.concatenate((self_coords_n_momenta_dict[coord].copy(), other_coords_n_momenta_dict[coord].copy()))})
            
        return result 
    
    def __radd__(self, other):
        if other==0:
            self_coords_n_momenta_dict = self.get_coords_n_momenta_dict()
            result = Particles(macroparticlenumber=self.macroparticlenumber, 
                    particlenumber_per_mp=self.particlenumber_per_mp, charge=self.charge,
                    mass=self.mass, circumference=self.circumference, gamma=self.gamma, coords_n_momenta_dict={})
        
            for coord in self_coords_n_momenta_dict.keys():
                #setattr(result, coord, np.concatenate((self_coords_n_momenta_dict[coord].copy(), other_coords_n_momenta_dict[coord].copy())))
                result.update({coord: self_coords_n_momenta_dict[coord].copy()})
        else:
            result = self.__add__(other)
        
        return result      
  

    # Statistics methods

    def mean_x(self):
        return mean(self.x)

    def mean_xp(self):
        return mean(self.xp)

    def mean_y(self):
        return mean(self.y)

    def mean_yp(self):
        return mean(self.yp)

    def mean_z(self):
        return mean(self.z)

    def mean_dp(self):
        return mean(self.dp)

    def sigma_x(self):
        return std(self.x)

    def sigma_y(self):
        return std(self.y)

    def sigma_z(self):
        return std(self.z)

    def sigma_xp(self):
        return std(self.xp)

    def sigma_yp(self):
        return std(self.yp)

    def sigma_dp(self):
        return std(self.dp)

    def effective_normalized_emittance_x(self):
        return cp.emittance(self.x, self.xp, None) * self.betagamma

    def effective_normalized_emittance_y(self):
        return cp.emittance(self.y, self.yp, None) * self.betagamma

    def effective_normalized_emittance_z(self):
        return(4*np.pi * cp.emittance(self.z, self.dp, None) * self.p0/e)

    def epsn_x(self):
        return (cp.emittance(self.x, self.xp, getattr(self, 'dp', None))
               * self.betagamma)

    def epsn_y(self):
        return (cp.emittance(self.y, self.yp, getattr(self, 'dp', None))
               * self.betagamma)

    def epsn_z(self):
        # always use the effective emittance
        return self.effective_normalized_emittance_z()

    def dispersion_x(self):
        return cp.dispersion(self.x, self.dp)

    def dispersion_y(self):
        return cp.dispersion(self.y, self.dp)

    def alpha_Twiss_x(self):
        return cp.get_alpha(self.x, self.xp, getattr(self, 'dp', None))

    def alpha_Twiss_y(self):
        return cp.get_alpha(self.y, self.yp, getattr(self, 'dp', None))

    def beta_Twiss_x(self):
        return cp.get_beta(self.x, self.xp, getattr(self, 'dp', None))

    def beta_Twiss_y(self):
        return cp.get_beta(self.y, self.yp, getattr(self, 'dp', None))

    def gamma_Twiss_x(self):
        return cp.get_gamma(self.x, self.xp, getattr(self, 'dp', None))

    def gamma_Twiss_y(self):
        return cp.get_gamma(self.y, self.yp, getattr(self, 'dp', None))


