'''Uses particle-in-cell algorithms from PyPIC for
space charge modelling in transverse 2.5D and 3D.

PyPIC can be found under
https://github.com/PyCOMPLETE/PyPIC .

@authors: Stefan Hegglin, Adrian Oeftiger
@date: 08.10.2015
'''

from __future__ import division

import numpy as np
from scipy.constants import c

from . import Element

from PyPIC.meshing import RectMesh3D


# for GPU use:
# def argsort(array):
#     idx = gpuarray.zeros(n_particles, dtype=np.int32)
#     get_sort_perm_int(array, idx)
#     return
argsort = np.argsort

# for GPU use:
    # lower_bounds = gpuarray.empty(mesh.n_nodes, dtype=np.int32)
    # upper_bounds = gpuarray.empty(mesh.n_nodes, dtype=np.int32)
    # seq = gpuarray.arange(mesh.n_nodes, dtype=np.int32)
    # lower_bound_int(node_ids, seq, lower_bounds)
    # upper_bound_int(node_ids, seq, upper_bounds)
searchsorted = np.searchsorted # used with side='left' and side='right'

arange = np.arange

class TransverseSpaceCharge(Element):
    '''Transverse slice-by-slice (2.5D) space charge using a
    particle-in-cell algorithm via PyPIC. Uses the same fixed 2D mesh
    for all slices.
    '''

    def __init__(self, slicer, length, pypic_algorithm, sort_particles=False,
                 *args, **kwargs):
        '''Arguments:
            - slicer: particles.Slicer instance.
            - length: interaction length over which the space charge is
              integrated.
            - pypic_algorithm: PyPIC.pypic.PyPIC(_GPU) instance which
              has the particle-in-cell algorithm encoded. This has to
              be set up by the user beforehand (i.e. the mesh,
              poisson solver, particle-to-mesh deposition method etc.).
            - sort_particles: determines whether to sort the particles
              by their mesh ID. This may speed up the PyPIC
              particle-to-mesh and mesh-to-particles methods
              due to coalesced memory access, especially on the GPU
              (test the timing for your parameters though!).

              (NB: sort_particles=True is necessarily required for the
               PyPIC_GPU.sorted_particles_to_mesh method.)
        '''
        self.slicer = slicer
        self.length = length
        self.pypic = pypic_algorithm
        self.sort_particles = sort_particles
        if pypic.mesh.dimension != 2:
            raise RuntimeError('2.5D space charge requires a two-dimensional '
                               'mesh!')

    def _create_3d_mesh(self, mesh_2d, z_cut_tail, z_cut_head, n_slices):
        '''For sorting purposes, in order for each slice to have all
        particles sorted by their transverse 2D mesh node ID.
        '''
        dz = (z_cut_head - z_cut_tail) / float(n_slices)
        return RectMesh3D(
            mesh_2d.x0, mesh_2d.y0, z_cut_tail,
            mesh_2d.dx, mesh_2d.dy, dz,
            mesh_2d.nx, mesh_2d.ny, n_slices,
            mathlib=mesh_2d.mathlib
        )

    def _align_particles(self, beam, mesh_3d):
        '''Sort all particles by their transverse 2D mesh node IDs via
        the given 3D mesh.
        '''
        ids = mesh_3d.get_node_ids(beam.x, beam.y, beam.z)
        permutation = argsort(ids)
        beam.reorder(permutation)
        # node ids have changed by now!

    def get_bounds(self, beam, mesh_2d, idx_relevant_particles):
        '''Determine indices of sorted particles for each cell, i.e.
        lower and upper index bounds.
        '''
        seq = arange(len(idx_relevant_particles), dtype=np.int32)
        ids = mesh_2d.get_node_ids(beam.x[idx_relevant_particles],
                                   beam.y[idx_relevant_particles])
        lower_bounds = searchsorted(ids, seq, side='left')
        upper_bounds = searchsorted(ids, seq, side='right')
        return lower_bounds, upper_bounds

    def track(self, beam):
        slices = self.slicer.slice(beam)

        if self.sort_particles:
            mesh_3d = self._create_3d_mesh(self.pypic.mesh, slices.z_cut_tail,
                                           slices.z_cut_head, slices.n_slices)
            self._align_particles(beam, mesh_3d)

        # last slice is always empty!
        for (sid, n_mp_in_slice) in enumerate(
                slices.n_macroparticles_per_slice[:-1]):
            # dz = slices.z_bins[i + 1] - slices.z_bins[i]
            pids_of_slice = slices.particle_indices_of_slice(sid)
            solve_kwargs = {}
            if self.sort_particles:
                solve_kwargs['lower_bounds'], solve_kwargs['upper_bounds'] = \
                    self.get_bounds(beam, self.pypic.mesh, pids_of_slice)
            # something with the density of particles: 'cylinders' n_mp_in_slice / dz
            e_x, e_y = self.pypic.pic_solve(beam.x, beam.y, **solve_kwargs)
            e_x *= beam.gamma**-2
            e_y *= beam.gamma**-2
            # calculate kick and apply!


class SpaceCharge3D(Element):
    '''Space charge in all three planes using a particle-in-cell
    algorithm via PyPIC.
    The 3D mesh does not adapt and remains constant.
    '''

    def __init__(self, slicer, length, pypic_algorithm, sort_particles=False,
                 *args, **kwargs):
        '''Arguments:
            - slicer: particles.Slicer instance, slicer.n_slices
              determines the longitudinal mesh size (for the 3D mesh)
            - length: interaction length over which the space charge is
              integrated.
            - mesh_nx: horizontal mesh size (for the mesh to be created)
            - mesh_ny: vertical mesh size (for the mesh to be created)

        Optional arguments:
            - pypic_algorithm: pre-configured PyPIC.pypic.PyPIC(_GPU)
              instance with the particle-in-cell algorithm encoded, has
              to be consistently set it up beforehand (i.e. the mesh
              w.r.t. the slicer, poisson solver, particle-to-mesh
              deposition method w.r.t. sort_particles etc.).
            - sort_particles: determines whether to sort the particles
              by their mesh ID. This may speed up the PyPIC
              particle-to-mesh and mesh-to-particles methods
              due to coalesced memory access, especially on the GPU
              (test the timing for your parameters though!).

              (NB: sort_particles=True is necessarily required for the
               PyPIC_GPU.sorted_particles_to_mesh method.)
        '''
        self.slicer = slicer
        self.length = length
        self.sort_particles = sort_particles
        if pypic_algorithm.mesh.dimension != 3:
            raise RuntimeError('3D space charge requires a three-dimensional '
                               'mesh!')
        self.pypic = pypic_algorithm

    @staticmethod
    def align_particles(beam, mesh):
        '''Sort all particles by their mesh node IDs.'''
        ids = mesh.get_node_ids(beam.x, beam.y, beam.z)
        permutation = argsort(ids)
        beam.reorder(permutation)
        # node id array has changed by now!

    @staticmethod
    def get_bounds(beam, mesh):
        '''Determine indices of sorted particles for each cell, i.e.
        lower and upper index bounds.
        '''
        ids = mesh.get_node_ids(beam.x, beam.y)
        seq = arange(beam.macroparticlenumber, dtype=np.int32)
        lower_bounds = searchsorted(ids, seq, side='left')
        upper_bounds = searchsorted(ids, seq, side='right')
        return lower_bounds, upper_bounds

    def track(self, beam):
        slices = self.slicer.slice(beam)
        mesh = self.pypic.mesh
        dz = mesh.dz # slice length

        if self.sort_particles:
            self.align_particles(beam, mesh)
            lower_bounds, upper_bounds = self.get_bounds(beam, self.pypic.mesh)
            solve_kwargs = {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
            }
        else:
            solve_kwargs = {}

        # something with the density of particles: 'cylinders' n_mp_in_slice / dz
        e_x, e_y, e_z = self.pypic.pic_solve(beam.x, beam.y, **solve_kwargs)
        e_x *= beam.gamma**-2
        e_y *= beam.gamma**-2
        # calculate kick and apply!

