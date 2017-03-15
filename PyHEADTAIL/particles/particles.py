'''
Created on 17.10.2014
@author: Kevin Li, Michael Schenk, Adrian Oeftiger
@copyright CERN
'''
import numpy as np
from scipy.constants import c, e  # unit e needed for long. emittance

from ..cobra_functions import stats as cp
from ..general import pmath as pm
from . import Printing

arange = np.arange
mean = np.mean
std = cp.std


class Particles(Printing):
    '''Contains the basic properties of a particle ensemble with
    their coordinate and conjugate momentum arrays, energy and the like.
    Designed to describe beams, electron clouds, ... '''

    def __init__(self, macroparticlenumber, particlenumber_per_mp,
                 charge, mass, circumference, gamma, coords_n_momenta_dict={},
                 *args, **kwargs):
        '''The dictionary coords_n_momenta_dict contains the coordinate
        and conjugate momenta names and assigns to each the
        corresponding array.
        e.g.: coords_n_momenta_dict = {'x': array(..), 'xp': array(..)}
        '''
        self.macroparticlenumber = macroparticlenumber
        self.particlenumber_per_mp = particlenumber_per_mp

        self.charge = charge
        self.mass = mass

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
    def charge_per_mp(self):
        return self.particlenumber_per_mp * self.charge
    @charge_per_mp.setter
    def charge_per_mp(self, value):
        self.particlenumber_per_mp = value / self.charge

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
        self.gamma = np.sqrt(1 + (value / (self.mass * c))**2)

    @property
    def z_beamframe(self):
        return self.z * self.gamma
    @z_beamframe.setter
    def z_beamframe(self, value):
        self.z = value / self.gamma

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


    def extract_slices(self, slicer, include_non_sliced='if_any', *args, **kwargs):
        '''Return a list Particles object with the different slices.
        The last element of the list contains particles not assigned to any slice.

        include_non_sliced : {'always', 'never', 'if_any'}, optional
        'always':
          extra element in the list with particles not belonging to any slice
          is always inserted (it can be empty).
        'never':
          extra element in the list with particles not belonging to any slice
          is never inserted.
        'if_any':
          extra element in the list with particles not belonging to any slice
          is inserted only if such particles exist.
        '''

        if include_non_sliced not in ['if_any', 'always', 'never']:
        	raise ValueError("include_non_sliced=%s is not valid!\n" % include_non_sliced +
        					 "Possible values are {'always', 'never', 'if_any'}" )

        slices = self.get_slices(slicer, *args, **kwargs)
        self_coords_n_momenta_dict = self.get_coords_n_momenta_dict()
        slice_object_list = []

        for i_sl in xrange(slices.n_slices):

            ix = slices.particle_indices_of_slice(i_sl)
            macroparticlenumber = len(ix)

            slice_object = Particles(macroparticlenumber=macroparticlenumber,
                particlenumber_per_mp=self.particlenumber_per_mp, charge=self.charge,
                mass=self.mass, circumference=self.circumference, gamma=self.gamma, coords_n_momenta_dict={})

            for coord in self_coords_n_momenta_dict.keys():
                slice_object.update({coord: self_coords_n_momenta_dict[coord][ix]})

            slice_object.id[:] = self.id[ix]

            slice_object.slice_info = {\
                    'z_bin_center': slices.z_centers[i_sl],\
                    'z_bin_right':slices.z_bins[i_sl+1],\
                    'z_bin_left':slices.z_bins[i_sl]}

            slice_object_list.append(slice_object)

        # handle unsliced
        if include_non_sliced is not 'never':
            ix = slices.particles_outside_cuts
            if len(ix)>0 or include_non_sliced is 'always':
                slice_object = Particles(macroparticlenumber=len(ix),
                    particlenumber_per_mp=self.particlenumber_per_mp, charge=self.charge,
                    mass=self.mass, circumference=self.circumference, gamma=self.gamma, coords_n_momenta_dict={})
                for coord in self_coords_n_momenta_dict.keys():
                    slice_object.update({coord: self_coords_n_momenta_dict[coord][ix]})
                slice_object.id[:] = self.id[ix]
                slice_object.slice_info = 'unsliced'
                slice_object_list.append(slice_object)

        return slice_object_list

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
        permutation = pm.argsort(getattr(self, attr))
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
            reordered = pm.apply_permutation(getattr(self, attr), permutation)
            setattr(self, attr, reordered)
        self.clean_slices()

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

        result.id = np.concatenate((self.id.copy(), other.id.copy()))

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
            result.id = self.id.copy()
        else:
            result = self.__add__(other)

        return result


    # Statistics methods
    # kwargs are for passing stream=... in the gpu case

    def mean_x(self, **kwargs):
        #return np.float(pm.mean(self.x))
        return pm.mean(self.x, **kwargs)

    def mean_xp(self, **kwargs):
        #return np.float(pm.mean(self.xp))
        return pm.mean(self.xp, **kwargs)

    def mean_y(self, **kwargs):
        #return np.float(pm.mean(self.y))
        return pm.mean(self.y, **kwargs)

    def mean_yp(self, **kwargs):
        #return np.float(pm.mean(self.yp))
        return pm.mean(self.yp, **kwargs)

    def mean_z(self, **kwargs):
        #return np.float(pm.mean(self.z))
        return pm.mean(self.z, **kwargs)

    def mean_dp(self, **kwargs):
        #return np.float(pm.mean(self.dp))
        return pm.mean(self.dp, **kwargs)

    def sigma_x(self, **kwargs):
        return pm.std(self.x, **kwargs)

    def sigma_y(self, **kwargs):
        #return np.float(pm.std(self.y))
        return pm.std(self.y, **kwargs)

    def sigma_z(self, **kwargs):
        #return np.float(pm.std(self.z))
        return pm.std(self.z, **kwargs)

    def sigma_xp(self, **kwargs):
        #return np.float(pm.std(self.xp))
        return pm.std(self.xp, **kwargs)

    def sigma_yp(self, **kwargs):
        #return np.float(pm.std(self.yp))
        return pm.std(self.yp, **kwargs)

    def sigma_dp(self, **kwargs):
        #return np.float(pm.std(self.dp))
        return pm.std(self.dp, **kwargs)

    def effective_normalized_emittance_x(self, **kwargs):
        return pm.emittance(self.x, self.xp, None, **kwargs) * self.betagamma

    def effective_normalized_emittance_y(self, **kwargs):
        return pm.emittance(self.y, self.yp, None, **kwargs) * self.betagamma

    def effective_normalized_emittance_z(self, **kwargs):
        return(4*np.pi * pm.emittance(self.z, self.dp, None, **kwargs) * self.p0/e)

    def epsn_x(self, **kwargs):
        return (pm.emittance(self.x, self.xp, getattr(self, 'dp', None), **kwargs)
               * self.betagamma)

    def epsn_y(self, **kwargs):
        return (pm.emittance(self.y, self.yp, getattr(self, 'dp', None), **kwargs)
               * self.betagamma)

    def epsn_z(self, **kwargs):
        # always use the effective emittance
        return self.effective_normalized_emittance_z(**kwargs)

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
