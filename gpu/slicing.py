'''
@authors: Adrian Oeftiger
@date:    30/07/2015
'''
from __future__ import division

import os

where = os.path.dirname(os.path.abspath(__file__)) + '/'

import numpy as np

# default classes imports from modules as assigned in gpu/__init__.py
from . import def_slicing

from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda import cumath

import thrust_interface
thrust = thrust_interface.compiled_module

get_sort_perm_int = thrust.get_sort_perm_int
lower_bound_int = thrust.lower_bound_int
upper_bound_int = thrust.upper_bound_int

# load kernels
with open(where + 'stats.cu') as stream:
    source = stream.read()
stats_kernels = SourceModule(source)

sorted_mean_per_slice = stats_kernels.get_function('sorted_mean_per_slice')
sorted_cov_per_slice = stats_kernels.get_function('sorted_cov_per_slice')

# prepare calls to kernels!!!
sorted_mean_per_slice.prepare('PPPIP')
sorted_cov_per_slice.prepare('PPPIP')

class PyPICSliceSet(def_slicing.SliceSet):
    '''Defines a set of longitudinal slices with PyPIC as the algorithm,
    this allows for the use of the GPU.
    '''

    def __init__(self, z_bins, slice_index_of_particle, mode, pypic_algorithm,
                 n_macroparticles_per_slice=None, beam_parameters={}):
        '''Is intended to be created by the SlicerGPU factory method.
        A SliceSet is given a set of intervals defining the slicing
        region and the histogram over the thereby defined slices.

        beam_parameters is a dictionary containing certain beam
        parameters to be recorded with this SliceSet.
        (e.g. beta being saved via beam_parameters['beta'] = beam.beta)
        beam_parameters contains lower_bounds and upper_bounds (see SlicerGPU).
        '''
        super(PyPICSliceSet, self).__init__(
            z_bins, slice_index_of_particle, mode,
            n_macroparticles_per_slice, beam_parameters
        )
        self.pypic_algorithm = pypic_algorithm

    @property
    def z_cut_head(self):
        return self.z_bins[-1].get()

    @property
    def z_cut_tail(self):
        return self.z_bins[0].get()

    @property
    def z_centers(self):
        # @TODO: current point
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
        return np.diff(self.z_bins)

    @property
    def slice_positions(self):
        '''Position of the respective slice start within the array
        self.particle_indices_per_slice .
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
                lambda_of_bins, sigma=sigma, mode='wrap')
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
                           sigma=sigma, mode='wrap')
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


class SlicerGPU(def_slicing.Slicer):
    '''Implementation of the Slicer with statistics functions on the
    GPU.
    '''

    def slice(self, beam, *args, **kwargs):
        '''Return a SliceSet object according to the saved
        configuration. Generate it using the keywords of the
        self.compute_sliceset_kwargs(beam) method.
        Defines interface to create SliceSet instances
        (factory method).

        Sort beam attributes by slice indices.

        Arguments:
        - statistics=True attaches mean values, standard deviations
        and emittances to the SliceSet for all planes.
        - statistics=['mean_x', 'sigma_dp', 'epsn_z'] only adds the
        listed statistics values (can be used to save time).
        Valid list entries are all statistics functions of Particles.
        '''
        sliceset_kwargs = self.compute_sliceset_kwargs(beam)
        slice_index_of_particle = sliceset_kwargs['slice_index_of_particle']

        sorting_permutation = gpuarray.zeros(n_particles, dtype=np.int32)
        # also resorts slice_index_of_particle:
        get_sort_perm_int(slice_index_of_particle, sorting_permutation)
        beam.reorder(sorting_permutation)
        del sorting_permutation
        lower_bounds = gpuarray.empty(mesh.n_nodes, dtype=np.int32)
        upper_bounds = gpuarray.empty(mesh.n_nodes, dtype=np.int32)
        seq = gpuarray.arange(self.n_slices, dtype=np.int32)
        lower_bound_int(slice_index_of_particle, seq, lower_bounds)
        upper_bound_int(slice_index_of_particle, seq, upper_bounds)
        del seq

        sliceset_kwargs['beam_parameters'] = (
            self.extract_beam_parameters(beam)
        )
        sliceset_kwargs['beam_parameters'].update(
            {'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds}
        )

        sliceset = SliceSet(**sliceset_kwargs)

        if 'statistics' in kwargs:
            self.add_statistics(sliceset, beam, kwargs['statistics'],
                                lower_bounds, upper_bounds)

        return sliceset

    def add_statistics(self, sliceset, beam, statistics,
                       lower_bounds, upper_bounds):
        '''Calculate all the statistics quantities (strings) that are
        named in the list 'statistics' and add a corresponding
        attribute to the SliceSet instance. The nomenclature must be
        followed. The following names are possible:
            mean_x, mean_y, mean_z, mean_xp, mean_yp, mean_dp, sigma_x,
            sigma_y, sigma_z, sigma_dp, epsn_x, epsn_y, epsn_z.

        statistics=True adds all at once.

        Assumes beam attributes to be sorted by slice indices.
        (The statistics kernels require arrays ordered by slice.)
        The index arrays lower_bounds and upper_bounds indicate the
        start and end indices within the sorted particle arrays for
        each slice. The respective slice id is identical to the indexing
        within lower_bounds and upper_bounds.
        '''
        if statistics is True:
            statistics = ['mean_x', 'mean_y', 'mean_z',
                          'mean_xp', 'mean_yp', 'mean_dp',
                          'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp',
                          # 'epsn_x', 'epsn_y', 'epsn_z',
                          # 'eff_epsn_x', 'eff_epsn_y'
                          ]
        for stat in statistics:
            if not hasattr(sliceset, stat):
                stat_caller = getattr(self, '_' + stat)
                values = stat_caller(sliceset, beam, lower_bounds, upper_bounds)
                setattr(sliceset, stat, values)

    def _mean_x(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._mean(sliceset, beam.x,
                          lower_bounds, upper_bounds)

    def _mean_xp(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._mean(sliceset, beam.xp,
                          lower_bounds, upper_bounds)

    def _mean_y(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._mean(sliceset, beam.y,
                          lower_bounds, upper_bounds)

    def _mean_yp(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._mean(sliceset, beam.yp,
                          lower_bounds, upper_bounds)

    def _mean_z(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._mean(sliceset, beam.z,
                          lower_bounds, upper_bounds)

    def _mean_dp(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._mean(sliceset, beam.dp,
                          lower_bounds, upper_bounds)

    def _sigma_x(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._sigma(sliceset, beam.x,
                           lower_bounds, upper_bounds)

    def _sigma_y(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._sigma(sliceset, beam.y,
                           lower_bounds, upper_bounds)

    def _sigma_z(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._sigma(sliceset, beam.z,
                           lower_bounds, upper_bounds)

    def _sigma_dp(self, sliceset, beam, lower_bounds, upper_bounds):
        return self._sigma(sliceset, beam.dp,
                           lower_bounds, upper_bounds)

    # def _epsn_x(self, sliceset, beam): # dp will always be present in a sliced beam
    #     return self._epsn(sliceset, beam.x, beam.xp, beam.dp) * beam.betagamma

    # def _eff_epsn_x(self, sliceset, beam):
    #     return self._epsn(sliceset, beam.x, beam.xp, None) * beam.betagamma

    # def _epsn_y(self, sliceset, beam):
    #     return self._epsn(sliceset, beam.y, beam.yp, beam.dp) * beam.betagamma

    # def _eff_epsn_y(self, sliceset, beam):
    #     return self._epsn(sliceset, beam.y, beam.yp, None) * beam.betagamma


    # def _epsn_z(self, sliceset, beam):
    #     # Always use the effective emittance --> pass None as second dp param
    #     return (4. * np.pi * self._epsn(sliceset, beam.z, beam.dp, None) *
    #             beam.p0 / beam.charge)

    # Statistics helper functions.

    def _mean(self, sliceset, u, lower_bounds, upper_bounds):
        block = (256, 1, 1)
        grid = (min(sliceset.n_slices // block[0], 1), 1, 1)
        mean_u = gpuarray.zeros(sliceset.n_slices, dtype=np.float64)
        sorted_mean_per_slice(lower_bounds.gpudata,
                              upper_bounds.gpudata,
                              u.gpudata,
                              self.n_slices,
                              mean_u.gpudata)
        return mean_u

    def _sigma(self, sliceset, u, lower_bounds, upper_bounds):
        cov_u = np.zeros(sliceset.n_slices)
        sorted_cov_per_slice(lower_bounds.gpudata,
                             upper_bounds.gpudata,
                             u.gpudata,
                             self.n_slices,
                             cov_u.gpudata)
        return cumath.sqrt(cov_u)

    # def _epsn(self, sliceset, u, up, dp):
    #     epsn_u = np.zeros(sliceset.n_slices)
    #     cp.emittance_per_slice(sliceset.slice_index_of_particle,
    #                            sliceset.particles_within_cuts,
    #                            sliceset.n_macroparticles_per_slice,
    #                            u, up, dp, epsn_u)
    #     return epsn_u


class PyPICSlicer(SlicerGPU):
    '''Slices with respect to the mesh of the PyPIC algorithm.
    Uses the PyPIC methods for the SliceSet which allows for GPU use.
    '''

    def __init__(self, pypic_algorithm, *args, **kwargs):
        '''Set up a Slicer with the mesh of pypic_algorithm.
        z_cuts are set according to the left and right node of the mesh.

        The returned Slicer will use a 1D rectangular mesh from PyPIC.
        '''
        if isinstance(pypic_algorithm.mesh, UniformMesh1D):
            mode = 'uniform_bin'
        else:
            raise NotImplementedError(
                'beam.z is hard-coded in the PyPICSlicer, other slicing '
                'schemes than UniformMesh1D have to be implemented first.')
        n_slices = pypic_algorithm.mesh.n_nodes
        n_sigma_z = None
        z_cuts = None

        self.pypic_algorithm = pypic_algorithm
        self.config = (mode, n_slices, n_sigma_z, z_cuts)

    def get_long_cuts(self, beam):
        '''Return boundaries of slicing region defined by the PyPIC
        mesh: (z_cut_tail, z_cut_head).
        '''
        mesh = self.pypic_algorithm.mesh
        z_cuts = (mesh.origin[-1],
                  mesh.origin[-1] + mesh.shape_r[-1] * mesh.distances[-1])
        return z_cuts

    def compute_sliceset_kwargs(self, beam):
        '''Return argument dictionary to create a new SliceSet
        according to the saved configuration for
        uniformly binned SliceSet objects.
        '''
        z_cut_tail, z_cut_head = self.get_long_cuts(beam)
        slice_width = (z_cut_head - z_cut_tail) / float(self.n_slices)

        z_bins = gpuarray.arange(z_cut_tail, z_cut_head + 1e-7*slice_width,
                                 slice_width, dtype=np.float64)

        slice_index_of_particle = self.pypic_algorithm.mesh.get_node_ids(beam.z)

        return dict(z_bins=z_bins,
                    slice_index_of_particle=slice_index_of_particle,
                    mode=self.mode,
                    pypic_algorithm=self.pypic_algorithm)
