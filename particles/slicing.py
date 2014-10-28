'''
@authors: Hannes Bartosik,
          Kevin Li,
          Michael Schenk,
          Giovanni Iadarola,
          Adrian Oeftiger
@date:    01/10/2014
'''
from __future__ import division

import numpy as np
import scipy.ndimage as ndimage
from random import sample

from abc import ABCMeta, abstractmethod
from functools import partial

from ..cobra_functions import stats as cp
from ..general.decorators import memoize

class ModeIsNotUniformBin(Exception):
    def __str__(self):
        return "This SliceSet has self.mode not set to 'uniform_bin'!"

class ModeIsUniformCharge(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class SliceSet(object):
    '''Defines a set of longitudinal slices. It's a blueprint or photo
    of a beam's longitudinal profile. It knows where the slices are
    located, how many and which particles there are in which slice.
    It should not be updated with new distributions, rather create a
    new SliceSet.
    '''

    def __init__(self, z_bins, slice_index_of_particle, mode,
                 n_macroparticles_per_slice=None):
        '''Is intended to be created by the Slicer factory method.
        A SliceSet is given a set of intervals defining the slicing
        region and the histogram over the thereby defined slices.
        '''

        '''Array of z values of each bin, goes from the left
        bin edge of the first bin to the right bin edge of the last bin.
        '''
        self.z_bins = z_bins

        '''Array of slice indices for each particle, positions (indices)
        are the same as in beam.z .'''
        self.slice_index_of_particle = slice_index_of_particle

        '''How is the slicing done? for the moment it is either
        'uniform_charge' or 'uniform_bin'.
        '''
        self.mode = mode
        self._n_macroparticles_per_slice = n_macroparticles_per_slice

    @property
    def z_cut_head(self):
        return self.z_bins[-1]

    @property
    def z_cut_tail(self):
        return self.z_bins[0]

    @property
    def z_centers(self):
        return self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

    @property
    def n_slices(self):
        return len(self.z_bins) - 1

    @property
    def slice_widths(self):
        '''Array of the widths of the slices.'''
        return np.diff(self.z_bins)

    @property
    def slice_positions(self):
        '''Position of the respective slice start within the
        array self.particle_indices_per_slice .
        '''
        slice_positions_ = np.zeros(self.n_slices + 1, dtype=np.int32)
        slice_positions_[1:] = (
                np.cumsum(self.n_macroparticles_per_slice).astype(np.int32))
        return slice_positions_

    @property
    def n_macroparticles_per_slice(self):
        '''Slice distribution, i.e. number of macroparticles for each
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
    # @memoize
    def particles_within_cuts(self):
        '''All particle indices which are situated within the
        slicing region defined by [z_cut_tail, z_cut_head).'''
        particles_within_cuts_ = np.where(
                (self.slice_index_of_particle > -1) &
                (self.slice_index_of_particle < self.n_slices)
            )[0].astype(np.int32)
        return particles_within_cuts_

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

    def line_density_derivative(
            self, n_macroparticles=None):
        '''Array of length (n_slices - 1) containing
        the derivative of the n_macroparticles array.
        '''
        if self.mode is 'uniform_charge':
            raise ModeIsUniformCharge('The derivative is zero up to ' +
                                      'numerical fluctuations because the ' +
                                      'charges have been distributed ' +
                                      'uniformly across the slices.')
        if n_macroparticles is None:
            n_macroparticles = self.n_macroparticles_per_slice
        return np.gradient(n_macroparticles, self.slice_widths[0])

    def line_density_derivative_gauss(self, sigma=None, smoothen_before=True,
                                      smoothen_after=True):
        '''
        Calculate the derivative of the slice charge density while
        smoothing the line density via a Gaussian filter.
        Return list with entries of density derivative of length
        n_slices.
        '''
        if self.mode is not 'uniform_bin':
            raise ModeIsNotUniformBin()
        if sigma is None:
            sigma = 0.02 * self.n_slices
        smoothen = partial(ndimage.gaussian_filter1d,
                           sigma=sigma, mode='wrap')
        line_density = self.n_macroparticles_per_slice
        if smoothen_before:
            line_density = smoothen(line_density)
        derivative = self.line_density_derivative(line_density)
        if smoothen_after:
            derivative = smoothen(derivative)
        return derivative

    def particle_indices_of_slice(self, slice_index):
        '''Return an array of particle indices which are located in the
        slice defined by the given slice_index.
        '''
        pos      = self.slice_positions[slice_index]
        next_pos = self.slice_positions[slice_index + 1]

        return self.particle_indices_by_slice[pos:next_pos]

    # Statistics

    def mean_x(self, beam):
        return self._mean(beam.x)

    def mean_xp(self, beam):
        return self._mean(beam.xp)

    def mean_y(self, beam):
        return self._mean(beam.y)

    def mean_yp(self, beam):
        return self._mean(beam.yp)

    def mean_z(self, beam):
        return self._mean(beam.z)

    def mean_dp(self, beam):
        return self._mean(beam.dp)

    def sigma_x(self, beam):
        return self._sigma(beam.x)

    def sigma_y(self, beam):
        return self._sigma(beam.y)

    def sigma_z(self, beam):
        return self._sigma(beam.z)

    def sigma_dp(self, beam):
        return self._sigma(beam.dp)

    def epsn_x(self, beam):
        return self._epsn(beam.x, beam.xp) * beam.betagamma

    def epsn_y(self, beam):
        return self._epsn(beam.y, beam.yp) * beam.betagamma

    def epsn_z(self, beam):
        '''
        Approximate epsn_z. Correct for Gaussian beam.
        '''
        return (4. * np.pi * self.sigma_z(beam) * self.sigma_dp(beam) *
                beam.p0 / beam.charge)


    # Statistics helper functions.

    def _mean(self, u):
        mean_u = np.zeros(self.n_slices)
        cp.mean_per_slice(self.slice_index_of_particle,
                          self.particles_within_cuts,
                          self.n_macroparticles_per_slice,
                          u, mean_u)
        return mean_u

    def _sigma(self, u):
        sigma_u = np.zeros(self.n_slices)
        cp.std_per_slice(self.slice_index_of_particle,
                         self.particles_within_cuts,
                         self.n_macroparticles_per_slice,
                         u, sigma_u)
        return sigma_u

    def _epsn(self, u, up):
        epsn_u = np.zeros(self.n_slices)
        cp.emittance_per_slice(self.slice_index_of_particle,
                               self.particles_within_cuts,
                               self.n_macroparticles_per_slice,
                               u, up, epsn_u)
        return epsn_u


class Slicer(object):
    '''
    Slicer class that controls longitudinal discretization of a beam.
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

    @abstractmethod
    def slice(self, beam):
        '''Return a SliceSet object according to the saved
        configuration. Factory method.
        '''
        pass

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
            z_cut_tail = np.min(beam.z)
            z_cut_head = np.max(beam.z)
            z_cut_head += abs(z_cut_head) * 1e-15
        return z_cut_tail, z_cut_head

    def __hash__(self):
        return hash(self.config)

    def __eq__(self, other):
        return self.config == other.config

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

class UniformBinSlicer(Slicer):
    '''Slices with respect to uniform bins along the slicing region.'''

    def __init__(self, n_slices, n_sigma_z=None, z_cuts=None):
        '''
        Return a UniformBinSlicer object. Set and store the
        corresponding slicing configuration in self.config .
        Note that either n_sigma_z or z_cuts can be set. If both are
        given, a ValueError will be raised.
        '''
        if n_sigma_z and z_cuts:
            raise ValueError("Both arguments n_sigma_z and z_cuts are" +
                             " given while only one is accepted!")
        mode = 'uniform_bin'
        self.config = (mode, n_slices, n_sigma_z, z_cuts)

    def slice(self, beam):
        '''Return a SliceSet according to the saved configuration.
        Factory method for uniformly binned SliceSet objects.
        '''
        z_cut_tail, z_cut_head = self.get_long_cuts(beam)
        slice_width = (z_cut_head - z_cut_tail) / float(self.n_slices)

        z_bins = np.linspace(z_cut_tail, z_cut_head, self.n_slices + 1)
        slice_index_of_particle = np.floor(
                (beam.z - z_cut_tail) / slice_width
            ).astype(np.int32)

        return SliceSet(z_bins, slice_index_of_particle, 'uniform_bin')


class UniformChargeSlicer(Slicer):
    '''Slices with respect to uniform charge for each bin along the
    slicing region.
    '''

    def __init__(self, n_slices, n_sigma_z=None, z_cuts=None):
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

    def slice(self, beam):
        '''Return a SliceSet according to the saved configuration.
        Factory method for SliceSet objects with a uniform charge
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

        return SliceSet(z_bins, slice_index_of_particle, 'uniform_charge',
                        n_macroparticles_per_slice=n_part_per_slice)
