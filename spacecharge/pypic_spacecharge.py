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


i_gpu = True
# this whole paragraph simply provides GPU versions and needs the flag
# ---> to be replaced by new gpu interface from Stefan
if not i_gpu:
    arange = np.arange
    empty = np.empty
    empty_like = np.empty_like
    def argsort(*args, **kwargs):
        kwargs.pop('dest_array', None)
        return np.argsort(*args, **kwargs)
    def searchsortedleft(a, v, sorter=None, dest_array=None):
        return np.searchsorted(a, v, side='left', sorter=sorter)
    def searchsortedright(a, v, sorter=None, dest_array=None):
        return np.searchsorted(a, v, side='right', sorter=sorter)
else:
    from pycuda import gpuarray
    arange = gpuarray.arange
    empty = gpuarray.empty
    empty_like = gpuarray.empty_like

    import PyHEADTAIL.gpu
    mod = PyHEADTAIL.gpu.thrust_interface.compiled_module

    def argsort(array, dest_array=None):
        if dest_array is None:
            dest_array = gpuarray.zeros(len(array), dtype=np.int32)
        mod.get_sort_perm_int(array, dest_array)
        return dest_array

    def searchsortedleft(array, values, dest_array=None):
        if dest_array is None:
            dest_array = gpuarray.empty(len(values), dtype=np.int32)
        mod.lower_bound_int(array, values, dest_array)
        return dest_array

    def searchsortedright(array, values, dest_array=None):
        if dest_array is None:
            dest_array = gpuarray.empty(len(values), dtype=np.int32)
        mod.upper_bound_int(array, values, dest_array)
        return dest_array


class SpaceCharge25D(Element):
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

    @staticmethod
    def align_particles(beam, mesh_3d):
        '''Sort all particles by their transverse 2D mesh node IDs via
        the given 3D mesh.
        '''
        ids = mesh_3d.get_node_ids(beam.x, beam.y, beam.z)
        permutation = argsort(ids)
        beam.reorder(permutation)
        # node ids have changed by now!

    @staticmethod
    def get_bounds(beam, mesh_2d, idx_relevant_particles):
        '''Determine indices of sorted particles for each cell, i.e.
        lower and upper index bounds.
        '''
        seq = arange(mesh_2d.n_nodes, dtype=np.int32)
        # seq = arange(len(idx_relevant_particles), dtype=np.int32)
        ids = mesh_2d.get_node_ids(beam.x[idx_relevant_particles],
                                   beam.y[idx_relevant_particles])
        lower_bounds = searchsortedleft(ids, seq)
        upper_bounds = searchsortedright(ids, seq)
        return lower_bounds, upper_bounds

    def track(self, beam):
        slices = self.slicer.slice(beam)

        if self.sort_particles:
            mesh_3d = self._create_3d_mesh(self.pypic.mesh, slices.z_cut_tail,
                                           slices.z_cut_head, slices.n_slices)
            self.align_particles(beam, mesh_3d)


        # scale to macro-particle charges, integrate over length
        kick_factor = (self.length / (beam.beta*c) *
                       beam.charge_per_mp**2 / beam.p0)

        # last slice is always empty!
        for (sid, n_mp_in_slice) in enumerate(
                slices.n_macroparticles_per_slice[:-1]):

            pids_of_slice = slices.particle_indices_of_slice(sid)
            solve_kwargs = {
                'charge': beam.charge_per_mp,
            }
            if self.sort_particles:
                solve_kwargs['lower_bounds'], solve_kwargs['upper_bounds'] = \
                    self.get_bounds(beam, self.pypic.mesh, pids_of_slice)

            en_x, en_y = self.pypic.pic_solve(beam.x, beam.y, **solve_kwargs)

            en_x *= beam.gamma**-2
            en_y *= beam.gamma**-2

            beam.dx += en_x * kick_factor
            beam.dy += en_y * kick_factor



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
              due to coalesced memory access, especially on the GPU.

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

    def align_particles(self, beam, mesh):
        '''Sort all particles by their mesh node IDs.'''
        # if not hasattr(self, '_perm'):
        #     self._perm = empty(mesh.n_nodes, dtype=np.int32)
        ids = mesh.get_node_ids(beam.x, beam.y, beam.z_beamframe)
        # permutation = argsort(ids, dest_array=self._perm)
        permutation = argsort(ids)
        beam.reorder(permutation)
        # node id array has changed by now!

    def get_bounds(self, beam, mesh):
        '''Determine indices of sorted particles for each cell, i.e.
        lower and upper index bounds.
        '''
        if not hasattr(self, '_seq'):
            self._seq = arange(mesh.n_nodes, dtype=np.int32)
        # if not hasattr(self, '_bounds'):
        #     self._bounds = empty_like(self._seq)
        ids = mesh.get_node_ids(beam.x, beam.y, beam.z_beamframe)
        lower_bounds = searchsortedleft(ids, self._seq)#, dest_array=self._bounds)
        upper_bounds = searchsortedright(ids, self._seq)#, dest_array=self._bounds)
        return lower_bounds, upper_bounds

    def track(self, beam):
        slices = self.slicer.slice(beam)
        mesh = self.pypic.mesh

        solve_kwargs = {
            'charge': beam.charge_per_mp,
        }
        if self.sort_particles:
            self.align_particles(beam, mesh)

            solve_kwargs['lower_bounds'], solve_kwargs['upper_bounds'] = \
                self.get_bounds(beam, mesh)

        # charge normalised electric fields in beam frame [V/m / Coul]
        en_x, en_y, en_z = self.pypic.pic_solve(
            beam.x, beam.y, beam.z_beamframe, **solve_kwargs)

        # Lorentz boost to lab frame --> magnetic fields:
        en_x *= beam.gamma**-2
        en_y *= beam.gamma**-2

        # scale to macro-particle charges, integrate over length
        kick_factor = (self.length / (beam.beta*c) *
                       beam.charge_per_mp / beam.p0)

        beam.x += en_x * kick_factor
        beam.y += en_y * kick_factor
        beam.z += en_z * kick_factor
