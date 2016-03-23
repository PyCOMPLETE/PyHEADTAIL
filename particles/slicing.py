'''
@authors: Hannes Bartosik,
          Stefan Hegglin,
          Giovanni Iadarola,
          Kevin Li,
          Adrian Oeftiger,
          Michael Schenk
@date:    01/10/2014
'''
from __future__ import division

import numpy as np
import scipy.ndimage as ndimage
from scipy.constants import c, e
from random import sample

from abc import ABCMeta, abstractmethod
from functools import partial, wraps

from ..cobra_functions import stats as cp
# from ..general.decorators import memoize
from . import Printing

from scipy import interpolate


floor = np.floor
empty_like = np.empty_like
min_ = np.min
max_ = np.max
arange = np.arange
diff = np.diff
def make_int32(array):
    # return np.array(array, dtype=np.int32)
    return array.astype(np.int32)


class ModeIsUniformCharge(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

# @clean_slices needs to be attached to the track methods that change
# the longitudinal phase space. It could not be handled automatically
# inside the Particles class because e.g. the beam.z and beam.dp would
# have to be exposed as properties. This would introduce function
# overhead to a central data element and thus slow down PyHEADTAIL
# considerably.
def clean_slices(long_track_method):
    '''Adds the beam.clean_slices() to any track(beam) method of
    longitudinal elements (elements that change beam.z, the
    longitudinal position of any particles).
    '''
    @wraps(long_track_method)
    def cleaned_long_track_method(long_track_element, beam, *args, **kwargs):
        res = long_track_method(long_track_element, beam, *args, **kwargs)
        beam.clean_slices()
        return res
    return cleaned_long_track_method


class SliceSet(Printing):
    '''Defines a set of longitudinal slices. It's a blueprint or photo
    of a beam's longitudinal profile. It knows where the slices are
    located, how many and which particles there are in which slice. All
    its attributes refer to the state of the beam at creation time of
    the SliceSet. Hence, it must never be updated with new
    distributions, rather, a new SliceSet needs to be created.
    '''

    def __init__(self, z_bins, slice_index_of_particle, mode,
                 n_macroparticles_per_slice=None,
                 beam_parameters={}):
        '''Is intended to be created by the Slicer factory method.
        A SliceSet is given a set of intervals defining the slicing
        region and the histogram over the thereby defined slices.

        beam_parameters is a dictionary containing certain beam
        parameters to be recorded with this SliceSet.
        (e.g. beta being saved via beam_parameters['beta'] = beam.beta)
        '''

        '''Array of z values of each bin, goes from the left bin edge
        of the first bin to the right bin edge of the last bin.
        '''
        self.z_bins = z_bins

        '''Array of slice indices for each particle, positions (indices)
        are the same as in beam.z .
        '''
        self.slice_index_of_particle = slice_index_of_particle

        '''How is the slicing done? For the moment it is either
        'uniform_charge' or 'uniform_bin'.
        '''
        self.mode = mode

        '''Numpy array containing the number of macroparticles in each
        slice.
        '''
        self._n_macroparticles_per_slice = n_macroparticles_per_slice

        for p_name, p_value in beam_parameters.iteritems():
            if hasattr(self, p_name):
                raise ValueError('SliceSet.' + p_name + ' already exists!' +
                                 'Do not overwrite existing SliceSet ' +
                                 'attributes via the beam_parameters ' +
                                 'keyword entries! ')
            setattr(self, p_name, p_value)

    @property
    def z_cut_head(self):
        return self.z_bins[-1]

    @property
    def z_cut_tail(self):
        return self.z_bins[0]

    @property
    def z_centers(self):
        return self.z_bins[:-1] + 0.5 * (self.z_bins[1:] - self.z_bins[:-1])

    @property
    def n_slices(self):
        return len(self.z_bins) - 1

    @property
    def smoothing_sigma(self):
        return 0.02 * self.n_slices

    @property
    def slice_widths(self):
        '''Array of the widths of the slices.'''
        return diff(self.z_bins)

    @property
    def slice_positions(self):
        '''Position of the respective slice start within the array
        self.particle_indices_by_slice .
        '''
        slice_positions_ = np.zeros(self.n_slices + 1, dtype=np.int32)
        slice_positions_[1:] = (
                np.cumsum(self.n_macroparticles_per_slice).astype(np.int32))
        return slice_positions_

    @property
    def n_macroparticles_per_slice(self):
        '''Slice distribution, i.e. number of macroparticles in each
        slice.
        '''
        if self._n_macroparticles_per_slice is None:
            self._n_macroparticles_per_slice = np.zeros(
                self.n_slices, dtype=np.int32)
            cp.count_macroparticles_per_slice(self.slice_index_of_particle,
                                              self.particles_within_cuts,
                                              self._n_macroparticles_per_slice)
        return self._n_macroparticles_per_slice

    @property
    def charge_per_slice(self):
        '''Array of slice charges, i.e. summing up all the
        particle charges for each slice.
        '''
        return self.charge_per_mp * self.n_macroparticles_per_slice

    @property
    # @memoize
    def particles_within_cuts(self):
        '''All particle indices which are situated within the slicing
        region defined by [z_cut_tail, z_cut_head).'''
        particles_within_cuts_ = make_int32(np.where(
                (self.slice_index_of_particle > -1) &
                (self.slice_index_of_particle < self.n_slices)
            )[0])
        return particles_within_cuts_
    
    @property
    # @memoize
    def particles_outside_cuts(self):
        '''All particle indices which are situated outside the slicing
        region defined by [z_cut_tail, z_cut_head).'''
        particles_ouside_cuts_ = make_int32(np.where(np.logical_not(
                (self.slice_index_of_particle > -1) &
                (self.slice_index_of_particle < self.n_slices))
            )[0])
        return particles_ouside_cuts_

    @property
    # @memoize
    def particle_indices_by_slice(self):
        '''Array of particle indices arranged / sorted according to
        their slice affiliation.
        '''
        particle_indices_by_slice = np.zeros(len(self.particles_within_cuts),
                                             dtype=np.int32)
        cp.sort_particle_indices_by_slice(
            self.slice_index_of_particle, self.particles_within_cuts,
            self.slice_positions, particle_indices_by_slice)
        return particle_indices_by_slice

    def lambda_bins(self, sigma=None, smoothen=True):
        '''Line charge density with respect to bins along the slices.'''
        if sigma is None:
            sigma = self.smoothing_sigma
        lambda_of_bins = self.n_macroparticles_per_slice * self.charge_per_mp
        if smoothen:
            lambda_of_bins = ndimage.gaussian_filter1d(
                lambda_of_bins, sigma=sigma, mode='nearest')
        return lambda_of_bins

    def lambda_prime_bins(self, sigma=None, smoothen_before=True,
                                smoothen_after=True):
        '''Return array of length (n_slices - 1) containing
        the derivative of the line charge density \lambda
        w.r.t. the slice bins while smoothing via a Gaussian filter.
        (i.e. the smoothened derivative of the n_macroparticles array
        times the macroparticle charge.)
        '''
        if self.mode is 'uniform_charge':
            # self.warns('The line charge density derivative is zero up to ' +
            #            'numerical fluctuations w.r.t. bins because the ' +
            #            'charges have been distributed uniformly across ' +
            #            'the slices.')
            raise ModeIsUniformCharge('The derivative is zero up to ' +
                                      'numerical fluctuations because the ' +
                                      'charges have been distributed ' +
                                      'uniformly across the slices.')
        if sigma is None:
            sigma = self.smoothing_sigma
        smoothen = partial(ndimage.gaussian_filter1d,
                           sigma=sigma, mode='nearest')
        line_density = self.n_macroparticles_per_slice
        if smoothen_before:
            line_density = smoothen(line_density)
        # not compatible with uniform_charge:
        # (perhaps use gaussian_filter1d for derivative!)
        mp_density_derivative = np.gradient(line_density, self.slice_widths[0])
        if smoothen_after:
            mp_density_derivative = smoothen(mp_density_derivative)
        return mp_density_derivative * self.charge_per_mp

    def lambda_z(self, z, sigma=None, smoothen=True):
        '''Line charge density with respect to z along the slices.'''
        lambda_along_bins = (self.lambda_bins(sigma, smoothen)
                             / self.slice_widths)
        tck = interpolate.splrep(self.z_centers, lambda_along_bins, s=0)
        l_of_z = interpolate.splev(z, tck, der=0, ext=1)
        return l_of_z

    def lambda_prime_z(self, z, sigma=None, smoothen_before=True,
                       smoothen_after=True):
        '''Line charge density derivative with respect to z along
        the slices.
        '''
        lp_along_bins = self.lambda_prime_bins(
            sigma, smoothen_before, smoothen_after) / self.slice_widths
        tck = interpolate.splrep(self.z_centers, lp_along_bins, s=0)
        lp_of_z = interpolate.splev(z, tck, der=0, ext=1)
        return lp_of_z

    def particle_indices_of_slice(self, slice_index):
        '''Return an array of particle indices which are located in the
        slice defined by the given slice_index.
        '''
        pos      = self.slice_positions[slice_index]
        next_pos = self.slice_positions[slice_index + 1]

        return self.particle_indices_by_slice[pos:next_pos]

    def convert_to_time(self, z):
        '''Convert longitudinal quantity from length to time units using
        the relativistic beta saved at creation time of the SliceSet.
        '''
        return z / (self.beta * c)

    def convert_to_particles(self, slice_array, empty_particles=None):
        '''Convert slice_array with entries for each slice to a
        particle array with the respective entry of each particle
        given by its slice_array value via the slice that the
        particle belongs to.
        '''
        if empty_particles == None:
            empty_particles = empty_like(self.slice_index_of_particle, dtype=np.float)
        particle_array = empty_particles
        p_id = self.particles_within_cuts
        s_id = self.slice_index_of_particle.take(p_id)
        particle_array[p_id] = slice_array.take(s_id)
        return particle_array


class Slicer(Printing):
    '''Slicer class that controls longitudinal binning of a beam.
    Factory for SliceSet objects.
    '''
    __metaclass__ = ABCMeta

    @property
    def config(self):
        return (self.mode, self.n_slices, self.n_sigma_z, self.z_cuts)
    @config.setter
    def config(self, value):
        self.mode = value[0]
        self.n_slices = value[1]
        self.n_sigma_z = value[2]
        self.z_cuts = value[3]
        if(self.z_cuts != None and self.z_cuts[0] >= self.z_cuts[1]):
            self.warns('Slicer.config: z_cut_tail >= z_cut_head,'+
                       ' this leads to negative ' +
                       'bin sizes and particle indices starting at the head')

    @abstractmethod
    def compute_sliceset_kwargs(self, beam):
        '''Return argument dictionary to create a new
        SliceSet object according to the saved configuration.
        This method defines the slicing behaviour of inheriting
        Slicer implementations.
        '''
        pass

    def slice(self, beam, *args, **kwargs):
        '''Return a SliceSet object according to the saved
        configuration. Generate it using the keywords of the
        self.compute_sliceset_kwargs(beam) method.
        Defines interface to create SliceSet instances
        (factory method).

        Arguments:
        - statistics=True attaches mean values, standard deviations
        and emittances to the SliceSet for all planes.
        - statistics=['mean_x', 'sigma_dp', 'epsn_z'] only adds the
        listed statistics values (can be used to save time).
        Valid list entries are all statistics functions of Particles.
        '''
        sliceset_kwargs = self.compute_sliceset_kwargs(beam)
        sliceset_kwargs['beam_parameters'] = (
            self.extract_beam_parameters(beam))
        sliceset = SliceSet(**sliceset_kwargs)
        if 'statistics' in kwargs:
            self.add_statistics(sliceset, beam, kwargs['statistics'])
        return sliceset

    # generalise to extract all relevant parameters automatically?
    @staticmethod
    def extract_beam_parameters(beam):
        '''Return a dictionary of beam parameters to be stored
        in a SliceSet instance. (such as beam.beta etc.)
        '''
        return dict(beta=beam.beta, gamma=beam.gamma, p0=beam.p0,
                    particlenumber_per_mp=beam.particlenumber_per_mp,
                    charge=beam.charge, charge_per_mp=beam.charge_per_mp,
                    mass=beam.mass, intensity=beam.intensity
                    )

    def get_long_cuts(self, beam):
        '''Return boundaries of slicing region,
        (z_cut_tail, z_cut_head). If they have been set at
        instantiation, self.z_cuts is returned.
        If n_sigma_z is given, a cut of
        n_sigma_z * beam.sigma_z to the left and to the right
        respectively is applied, otherwise the longitudinally first and
        last particle define the full region.
        '''
        if self.z_cuts is not None:
            return self.z_cuts
        elif self.n_sigma_z:
            z_cut_tail = beam.mean_z() - self.n_sigma_z * beam.sigma_z()
            z_cut_head = beam.mean_z() + self.n_sigma_z * beam.sigma_z()
        else:
            z_cut_tail = min_(beam.z)
            z_cut_head = max_(beam.z)
            z_cut_head += abs(z_cut_head) * 1e-15
        return z_cut_tail, z_cut_head

    def __hash__(self):
        '''Identifies different instantiations of Slicer objects via
        their configuration (instead of their instance ID).
        '''
        return hash(self.config)

    def __eq__(self, other):
        return self.config == other.config

    def __ne__(self, other):
        return not self.__eq__(other)

    # for notifying users of previous versions to use the right new methods
    def update_slices(self, beam):
        '''non-existent anymore!'''
        raise RuntimeError('update_slices(beam) no longer exists. ' +
                           'Instead, remove all previously recorded SliceSet' +
                           'objects in the beam via beam.clean_slices() ' +
                           'when the longitudinal state of the beam is ' +
                           'changed. Concretely: replace ' +
                           'slices.update_slices(beam) by ' +
                           'beam.clean_slices().' +
                           'The SliceSet objects should be ' +
                           'retrieved via beam.get_slices(Slicer) *only*. ' +
                           'In this way the beam can memorize previously ' +
                           'created slice snapshots. This minimises ' +
                           'computation time.')

    def add_statistics(self, sliceset, beam, statistics):
        '''Calculate all the statistics quantities (strings) that are
        named in the list 'statistics' and add a corresponding
        attribute to the SliceSet instance. The nomenclature must be
        followed. The following names are possible:
            mean_x, mean_y, mean_z, mean_xp, mean_yp, mean_dp, sigma_x,
            sigma_y, sigma_z, sigma_dp, epsn_x, epsn_y, epsn_z.

        statistics=True adds all at once.
        '''
        if statistics is True:
            statistics = ['mean_x', 'mean_y', 'mean_z',
                          'mean_xp', 'mean_yp', 'mean_dp',
                          'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp',
                          'epsn_x', 'epsn_y', 'epsn_z',
                          'eff_epsn_x', 'eff_epsn_y']
        for stat in statistics:
            if not hasattr(sliceset, stat):
                stat_caller = getattr(self, '_' + stat)
                values = stat_caller(sliceset, beam)
                setattr(sliceset, stat, values)

    def _mean_x(self, sliceset, beam):
        return self._mean(sliceset, beam.x)

    def _mean_xp(self, sliceset, beam):
        return self._mean(sliceset, beam.xp)

    def _mean_y(self, sliceset, beam):
        return self._mean(sliceset, beam.y)

    def _mean_yp(self, sliceset, beam):
        return self._mean(sliceset, beam.yp)

    def _mean_z(self, sliceset, beam):
        return self._mean(sliceset, beam.z)

    def _mean_dp(self, sliceset, beam):
        return self._mean(sliceset, beam.dp)

    def _sigma_x(self, sliceset, beam):
        return self._sigma(sliceset, beam.x)

    def _sigma_y(self, sliceset, beam):
        return self._sigma(sliceset, beam.y)

    def _sigma_z(self, sliceset, beam):
        return self._sigma(sliceset, beam.z)

    def _sigma_dp(self, sliceset, beam):
        return self._sigma(sliceset, beam.dp)

    def _epsn_x(self, sliceset, beam): # dp will always be present in a sliced beam
        return self._epsn(sliceset, beam.x, beam.xp, beam.dp) * beam.betagamma

    def _eff_epsn_x(self, sliceset, beam):
        return self._epsn(sliceset, beam.x, beam.xp, None) * beam.betagamma

    def _epsn_y(self, sliceset, beam):
        return self._epsn(sliceset, beam.y, beam.yp, beam.dp) * beam.betagamma

    def _eff_epsn_y(self, sliceset, beam):
        return self._epsn(sliceset, beam.y, beam.yp, None) * beam.betagamma


    def _epsn_z(self, sliceset, beam):
        # Always use the effective emittance --> pass None as second dp param
        return (4. * np.pi * self._epsn(sliceset, beam.z, beam.dp, None) *
                beam.p0 / e)

    # Statistics helper functions.

    def _mean(self, sliceset, u):
        mean_u = np.zeros(sliceset.n_slices)
        cp.mean_per_slice(sliceset.slice_index_of_particle,
                          sliceset.particles_within_cuts,
                          sliceset.n_macroparticles_per_slice,
                          u, mean_u)
        return mean_u

    def _sigma(self, sliceset, u):
        sigma_u = np.zeros(sliceset.n_slices)
        cp.std_per_slice(sliceset.slice_index_of_particle,
                         sliceset.particles_within_cuts,
                         sliceset.n_macroparticles_per_slice,
                         u, sigma_u)
        return sigma_u

    def _epsn(self, sliceset, u, up, dp):
        epsn_u = np.zeros(sliceset.n_slices)
        cp.emittance_per_slice(sliceset.slice_index_of_particle,
                               sliceset.particles_within_cuts,
                               sliceset.n_macroparticles_per_slice,
                               u, up, dp, epsn_u)
        return epsn_u


class UniformBinSlicer(Slicer):
    '''Slices with respect to uniform bins along the slicing region.'''

    def __init__(self, n_slices, n_sigma_z=None, z_cuts=None,
                 z_sample_points=None, *args, **kwargs):
        '''
        Return a UniformBinSlicer object. Set and store the
        corresponding slicing configuration in self.config.
        Note that either n_sigma_z or z_cuts and/or z_sampling_point
        can be set. If both are given, a ValueError will be raised.
        '''
        if n_sigma_z and (z_cuts or z_sample_points):
            raise ValueError("Argument n_sigma_z is incompatible with" +
                             " either of z_cuts or z_sampling_points." +
                             " Choose either n_sigma_z or a combination" +
                             " of z_cuts and z_sampling_points.")
        mode = 'uniform_bin'
        if z_sample_points is not None:
            self.warns("n_slices will be overridden to match given" +
                       " combination of z_cuts and z_sampling_points.")
            n_slices, z_cuts = self._get_slicing_from_z_sample_points(
                z_sample_points, z_cuts)
        self.config = (mode, n_slices, n_sigma_z, z_cuts)

    def _get_slicing_from_z_sample_points(self, z_sample_points, z_cuts=None):
        '''
        Alternative slicing function for UniformBinSlicer. The function
        takes a given array of sampling points and ensures that the
        slice centers lie at those sampling points. If z_cuts is
        given and is beyond the sampling points, it furthermore extends
        the given sampling points at the same sampling frequency to
        include the range given by z_cuts. Very useful if one wants
        to ensure that certain points or regions of a wakefield are
        included or correctl sampled.
        '''
        dz = np.diff(z_sample_points)[0]
        if not np.allclose(np.diff(z_sample_points)-dz, 1e-15):
            raise TypeError("Irregular sampling of wakes incompatible with" +
                            " UniformBinSlicer. Check the sampling points.")

        # Get edges
        n_slices = len(z_sample_points)
        aa, bb = z_sample_points[0]-dz/2., z_sample_points[-1]+dz/2.

        # Extend/compress edges
        if z_cuts:
            if z_cuts[0]<aa:
                while z_cuts[0]<aa:
                    aa -= dz
                    n_slices += 1
            elif z_cuts[0]>aa:
                while z_cuts[0]>aa:
                    aa += dz
                    n_slices -= 1

            if z_cuts[1]<bb:
                while z_cuts[1]<bb:
                    bb -= dz
                    n_slices -=1
            elif z_cuts[1]>bb:
                while z_cuts[1]>bb:
                    bb += dz
                    n_slices +=1
        z_cuts = (aa, bb)

        return n_slices, z_cuts

    def compute_sliceset_kwargs(self, beam):
        '''Return argument dictionary to create a new SliceSet
        according to the saved configuration for
        uniformly binned SliceSet objects.
        '''
        z_cut_tail, z_cut_head = self.get_long_cuts(beam)
        slice_width = (z_cut_head - z_cut_tail) / float(self.n_slices)

        z_bins = arange(z_cut_tail, z_cut_head + 1e-7*slice_width,
                        slice_width, dtype=np.float64)
        slice_index_of_particle = make_int32(floor(
                (beam.z - z_cut_tail) / slice_width
            ))

        return dict(z_bins=z_bins,
                    slice_index_of_particle=slice_index_of_particle,
                    mode='uniform_bin')


class UniformChargeSlicer(Slicer):
    '''Slices with respect to uniform charge for each bin along the
    slicing region.
    '''

    def __init__(self, n_slices, n_sigma_z=None, z_cuts=None, *args, **kwargs):
        '''
        Return a UniformChargeSlicer object. Set and store the
        corresponding slicing configuration in self.config .
        Note that either n_sigma_z or z_cuts can be set. If both are
        given, a ValueError will be raised.
        '''
        if n_sigma_z and z_cuts:
            raise ValueError("Both arguments n_sigma_z and z_cuts are" +
                             " given while only one is accepted!")
        mode = 'uniform_charge'
        self.config = (mode, n_slices, n_sigma_z, z_cuts)

    def compute_sliceset_kwargs(self, beam):
        '''Return argument dictionary to create a new SliceSet
        according to the saved configuration for a uniform charge
        distribution along the bins.
        '''
        z_cut_tail, z_cut_head = self.get_long_cuts(beam)
        n_part = len(beam.z)

        # Sort particles according to z
        id_new = np.argsort(beam.z)
        id_old = np.arange(n_part).take(id_new)
        z_sorted = beam.z.take(id_new)

        n_cut_tail = np.searchsorted(z_sorted, z_cut_tail)
        n_cut_head = n_part - np.searchsorted(z_sorted, z_cut_head)

        # 1. n_part_per_slice - distribute macroparticles uniformly along
        # slices. Must be integer. Distribute remaining particles randomly
        # among slices with indices 'rand_slice_i'.
        n_part_within_cuts = n_part - n_cut_tail - n_cut_head
        rand_slice_i = sample(range(self.n_slices),
                              n_part_within_cuts % self.n_slices)

        n_part_per_slice = ((n_part_within_cuts // self.n_slices)
                            * np.ones(self.n_slices, dtype=np.int32))
        n_part_per_slice[rand_slice_i] += 1

        # 2. z-bins
        # Get indices of the particles defining the bin edges
        n_macroparticles_all = np.hstack(
            (n_cut_tail, n_part_per_slice, n_cut_head))
        first_indices = np.cumsum(n_macroparticles_all, dtype=np.int32)[:-1]

        z_bins = np.empty(self.n_slices + 1)
        z_bins[1:-1] = ((z_sorted[first_indices[1:-1]-1] +
            z_sorted[first_indices[1:-1]]) / 2)
        z_bins[0], z_bins[-1] = z_cut_tail, z_cut_head

        slice_index_of_particle_sorted = -np.ones(n_part, dtype=np.int32)
        for i in xrange(self.n_slices):
            start, end = first_indices[i], first_indices[i+1]
            slice_index_of_particle_sorted[start:end] = i

        # Sort back, including vector 'slice_index_of_particle'.
        slice_index_of_particle = np.empty(n_part, dtype=np.int32)
        np.put(slice_index_of_particle, id_old, slice_index_of_particle_sorted)

        return dict(z_bins=z_bins,
                    slice_index_of_particle=slice_index_of_particle,
                    mode='uniform_charge',
                    n_macroparticles_per_slice=n_part_per_slice)
