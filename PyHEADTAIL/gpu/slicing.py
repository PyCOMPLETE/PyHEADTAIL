'''
@authors: Adrian Oeftiger
@date:    30/07/2015
'''
from __future__ import division

import os
where = os.path.dirname(os.path.abspath(__file__)) + '/'

import numpy as np

# default classes imports from modules as assigned in gpu/__init__.py
from .oldinit import def_slicing

from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import pycuda.driver as cuda
from pycuda import cumath

from skcuda.misc import diff

import thrust_interface as thrust

get_sort_perm_int = thrust.get_sort_perm_int
lower_bound_int = thrust.lower_bound_int
upper_bound_int = thrust.upper_bound_int


# load kernels
with open(where + 'stats.cu') as stream:
    source = stream.read()
stats_kernels = SourceModule(source)

sorted_mean_per_slice = stats_kernels.get_function('sorted_mean_per_slice')
sorted_std_per_slice = stats_kernels.get_function('sorted_std_per_slice')

with open(where + 'smoothing_kernels.cu') as stream:
    source = stream.read()
smoothing_kernels = SourceModule(source)

# uniform_smoothing does not work yet with several blocks launched!
# uniform_smoothing = smoothing_kernels.get_function('smoothing_stencil_1d')
gaussian_smoothing = smoothing_kernels.get_function('gaussian_smoothing_1d')

# prepare calls to kernels!!!
sorted_mean_per_slice.prepare('PPPIP')
sorted_std_per_slice.prepare('PPPIP')

# for both smoothing kernels, the launched threads need to cover the slices exactly!
# block size block=(32, 1, 1) is fixed!
# uniform_smoothing.prepare('PPi')
gaussian_smoothing.prepare('PPi')


slice_to_particles = ElementwiseKernel(
    "unsigned int* slice_index_of_particle, double* input_slice_quantity, "
    "double* output_particle_array",
    # i is the particle index within slice_index_of_particle
    "output_particle_array[i] = "
        "input_slice_quantity[slice_index_of_particle[i]]",
    "slice_to_particles_kernel"
)


from PyPIC.GPU.meshing import UniformMesh1D
from PyPIC.GPU.gradient.gradient import make_GPU_gradient


class MeshSliceSet(def_slicing.SliceSet):
    '''Defines a set of longitudinal slices with PyPIC as the algorithm,
    this allows for the use of the GPU.
    '''

    # '''Lower boundary indices of the respective slice (determined by
    # the index within lower_bounds) within the particle attributes.
    # '''
    # lower_bounds = []
    # '''Upper boundary indices of the respective slice (determined by
    # the index within upper_bounds) within the particle attributes.
    # '''
    # upper_bounds = []

    def __init__(self, z_bins, slice_index_of_particle, mode, mesh, context,
                 n_macroparticles_per_slice=None, beam_parameters={}):
        '''Is intended to be created by the SlicerGPU factory method.
        A SliceSet is given a set of intervals defining the slicing
        region and the histogram over the thereby defined slices.

        beam_parameters is a dictionary containing certain beam
        parameters to be recorded with this SliceSet.
        (e.g. beta being saved via beam_parameters['beta'] = beam.beta)
        beam_parameters contains lower_bounds and upper_bounds (see SlicerGPU).
        '''
        # # overwrite these afterwards via beam_parameters:
        # delattr(MeshSliceSet, 'lower_bounds')
        # delattr(MeshSliceSet, 'upper_bounds')

        super(MeshSliceSet, self).__init__(
            z_bins=z_bins,
            slice_index_of_particle=slice_index_of_particle,
            mode=mode,
            n_macroparticles_per_slice=n_macroparticles_per_slice,
            beam_parameters=beam_parameters
        )

        self.mesh = mesh
        self._context = context
        self._gradient = make_GPU_gradient(mesh, context)

    @property
    def z_cut_head(self):
        return float(self.z_bins[-1].get())

    @property
    def z_cut_tail(self):
        return float(self.z_bins[0].get())

    @property
    def slice_widths(self):
        '''Array of the widths of the slices.'''
        return diff(self.z_bins)

    @property
    def slice_positions(self):
        '''Position of the respective slice start within the array
        self.particle_indices_by_slice .
        '''
        if not hasattr(self, '_slice_positions'):
            # the last entry of slice_positions needs to be n_slices,
            # the other entries are the same as lower_bounds
            self._slice_positions = gpuarray.zeros(
                self.n_slices + 1, dtype=self.lower_bounds.dtype)
            self._slice_positions += self.n_slices
            cuda.memcpy_dtod(self._slice_positions.gpudata,
                             self.lower_bounds.gpudata,
                             self.lower_bounds.nbytes)
        return self._slice_positions

    @property
    def n_macroparticles_per_slice(self):
        '''Slice distribution, i.e. number of macroparticles in each
        slice.
        '''
        if self._n_macroparticles_per_slice is None:
            self._n_macroparticles_per_slice = (
                self.upper_bounds - self.lower_bounds)
        return self._n_macroparticles_per_slice

    @property
    def charge_per_slice(self):
        '''Array of slice charges, i.e. summing up all the
        particle charges for each slice.
        '''
        return self.charge_per_mp * self.n_macroparticles_per_slice

    @property
    def particles_within_cuts(self):
        '''All particle indices which are situated within the slicing
        region defined by [z_cut_tail, z_cut_head).'''
        particles_within_cuts_ = gpuarray.arange(self.lower_bounds[0],
                                                 self.upper_bounds[-1] + 1,
                                                 dtype=np.int32)
        return particles_within_cuts_

    # particles are automatically sorted by slice affiliation!
    particle_indices_by_slice = particles_within_cuts

    # for consequent and thorough implementation of all these
    # derivations and interpolations below use 1D textures as in
    # http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter20.html:

    def lambda_bins(self, sigma=1, smoothen=True):
        '''Line charge density with respect to bins along the slices.'''
        if sigma is not 1:
            raise NotImplementedError('Sigma != 1 for Gaussian smoothing not '
                                      'implemented yet! Check out the file '
                                      'smoothing_kernels.cu.')
        lambda_of_bins = self.n_macroparticles_per_slice * self.charge_per_mp
        if smoothen:
            new = gpuarray.empty_like(lambda_of_bins)
            block = (min(32, self.n_slices), 1, 1)
            grid = ((self.n_slices + block[0] - 1) // block[0], 1, 1)
            gaussian_smoothing.prepared_call(
                grid, block,
                lambda_of_bins.gpudata, new.gpudata,
                np.int32(len(lambda_of_bins))
            )
            self._context.synchronize()
            lambda_of_bins = new
        return lambda_of_bins

    def lambda_prime_bins(self, sigma=1, smoothen_before=True,
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
        if sigma is not 1:
            raise NotImplementedError('Sigma != 1 for Gaussian smoothing not '
                                      'implemented yet! Check out the file '
                                      'smoothing_kernels.cu.')
        line_density = self.n_macroparticles_per_slice.astype(np.float64)
        if smoothen_before:
            new = gpuarray.empty_like(line_density)
            block = (min(32, self.n_slices), 1, 1)
            grid = ((self.n_slices + block[0] - 1) // block[0], 1, 1)
            gaussian_smoothing.prepared_call(
                grid, block,
                line_density.gpudata, new.gpudata,
                np.int32(len(line_density))
            )
            self._context.synchronize()
            line_density = new
        mp_density_derivative = self._gradient(line_density)[0][0]
        if smoothen_after:
            new = gpuarray.empty_like(mp_density_derivative)
            block = (min(32, self.n_slices), 1, 1)
            grid = ((self.n_slices + block[0] - 1) // block[0], 1, 1)
            gaussian_smoothing.prepared_call(
                grid, block,
                mp_density_derivative.gpudata, new.gpudata,
                np.int32(len(mp_density_derivative))
            )
            self._context.synchronize()
            mp_density_derivative = new
        return mp_density_derivative * self.charge_per_mp

    def lambda_z(self, z, sigma=None, smoothen=True):
        '''Line charge density with respect to z along the slices.'''
        raise NotImplementedError('GPU splining needs to be implemented!')
        # lambda_along_bins = (self.lambda_bins(sigma, smoothen)
        #                      / self.slice_widths)
        # tck = interpolate.splrep(self.z_centers, lambda_along_bins, s=0)
        # l_of_z = interpolate.splev(z, tck, der=0, ext=1)
        # return l_of_z

    def lambda_prime_z(self, z, sigma=None, smoothen_before=True,
                       smoothen_after=True):
        '''Line charge density derivative with respect to z along
        the slices.
        '''
        raise NotImplementedError('GPU splining needs to be implemented!')
        # lp_along_bins = self.lambda_prime_bins(
        #     sigma, smoothen_before, smoothen_after) / self.slice_widths
        # tck = interpolate.splrep(self.z_centers, lp_along_bins, s=0)
        # lp_of_z = interpolate.splev(z, tck, der=0, ext=1)
        # return lp_of_z

    def particle_indices_of_slice(self, slice_index):
        '''Return an array of particle indices which are located in the
        slice defined by the given slice_index.
        '''
        return gpuarray.arange(self.lower_bounds[slice_index],
                               self.upper_bounds[slice_index] + 1,
                               dtype=np.int32)

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
            empty_particles = gpuarray.empty(self.slice_index_of_particle.shape,
                                             dtype=np.float64)
        slice_to_particles(self.slice_index_of_particle,
                           slice_array, empty_particles)
        return empty_particles


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

        sorting_permutation = gpuarray.zeros(beam.macroparticlenumber,
                                             dtype=np.int32)
        # also resorts slice_index_of_particle:
        get_sort_perm_int(slice_index_of_particle, sorting_permutation)
        beam.reorder(sorting_permutation)
        del sorting_permutation
        lower_bounds = gpuarray.empty(self.n_slices, dtype=np.int32)
        upper_bounds = gpuarray.empty(self.n_slices, dtype=np.int32)
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

        sliceset = MeshSliceSet(**sliceset_kwargs)

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
        grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
        mean_u = gpuarray.zeros(sliceset.n_slices, dtype=np.float64)
        sorted_mean_per_slice(lower_bounds.gpudata,
                              upper_bounds.gpudata,
                              u.gpudata,
                              self.n_slices,
                              mean_u.gpudata,
                              block=block, grid=grid)
        return mean_u

    def _sigma(self, sliceset, u, lower_bounds, upper_bounds):
        block = (256, 1, 1)
        grid = (max(sliceset.n_slices // block[0], 1), 1, 1)
        cov_u = gpuarray.zeros(sliceset.n_slices, dtype=np.float64)
        sorted_std_per_slice(lower_bounds.gpudata,
                             upper_bounds.gpudata,
                             u.gpudata,
                             self.n_slices,
                             cov_u.gpudata,
                             block=block, grid=grid)
        return cumath.sqrt(cov_u)

    # def _epsn(self, sliceset, u, up, dp):
    #     epsn_u = np.zeros(sliceset.n_slices)
    #     cp.emittance_per_slice(sliceset.slice_index_of_particle,
    #                            sliceset.particles_within_cuts,
    #                            sliceset.n_macroparticles_per_slice,
    #                            u, up, dp, epsn_u)
    #     return epsn_u


class MeshSlicer(SlicerGPU):
    '''Slices with respect to the mesh (from the PyPIC package).
    For GPU use.
    '''

    def __init__(self, mesh, context, *args, **kwargs):
        '''Set up a Slicer with the PyPIC mesh. It should be a 1D
        rectangular mesh (PyPIC.UniformMesh1D).
        z_cuts are set according to the left and right node of the mesh.
        '''
        if isinstance(mesh, UniformMesh1D):
            self.mode = 'uniform_bin'
        else:
            raise NotImplementedError(
                'beam.z is hard-coded in the MeshSlicer, other slicing '
                'schemes than UniformMesh1D have to be implemented first.')
        self.n_slices = mesh.n_nodes
        self.n_sigma_z = None
        self.mesh = mesh
        self._context = context

    @property
    def z_cuts(self):
        mesh = self.mesh
        z_cuts = (mesh.origin[-1],
                  mesh.origin[-1] + mesh.shape_r[-1] * mesh.distances[-1])
        return z_cuts

    def get_long_cuts(self, beam):
        '''Return boundaries of slicing region defined by the PyPIC
        mesh range, the tuple format being (z_cut_tail, z_cut_head).
        '''
        return self.z_cuts

    def compute_sliceset_kwargs(self, beam):
        '''Return argument dictionary to create a new SliceSet
        according to the saved configuration for
        uniformly binned SliceSet objects.
        '''
        z_cut_tail, z_cut_head = self.get_long_cuts(beam)
        slice_width = (z_cut_head - z_cut_tail) / float(self.n_slices)

        z_bins = gpuarray.arange(z_cut_tail, z_cut_head + 1e-7*slice_width,
                                 slice_width, dtype=np.float64)

        slice_index_of_particle = self.mesh.get_node_ids(beam.z)

        return dict(z_bins=z_bins,
                    slice_index_of_particle=slice_index_of_particle,
                    mode=self.mode,
                    mesh=self.mesh,
                    context=self._context)
