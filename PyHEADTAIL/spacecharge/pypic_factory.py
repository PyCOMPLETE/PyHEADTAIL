'''Factory for PyPIC particle-in-cell algorithms with FFT Poisson
solving.
Used for space charge modelling under pypic_spacecharge.

PyPIC can be found under
https://github.com/PyCOMPLETE/PyPIC .

@authors: Adrian Oeftiger
@date: 18.01.2016
'''

from __future__ import division
import numpy as np

from PyPIC.GPU import pypic, meshing
from PyPIC.GPU.poisson_solver import FFT_solver

try:
    pypic_algorithm_class = pypic.PyPIC_GPU
    solver_class = FFT_solver.GPUFFTPoissonSolver
except ImportError:
    print ('PyHEADTAIL -- PyPIC factory: GPU version not available.')
    pypic_algorithm_class = pypic.PyPIC
    def solver_class(mesh, context=None):
        if not np.isclose(mesh.dx, mesh.dy):
            raise ValueError('CPU version of the FFT Poisson Solver '
                             '(Gianni\'s version) only accepts square '
                             'cells with mesh.dx == mesh.dy')
        Dh = mesh.dx
        x_aper = 0.5 * mesh.nx * Dh
        y_aper = 0.5 * mesh.ny * Dh
        return FFT_solver.FFT_OpenBoundary_SquareGrid(
            x_aper, y_aper, Dh,
            # we do not want exterior 5 and 4 cells from PyPIC <= v1.03:
            ext_boundary=False,
        )

from ..general import pmath as pm

def ensure_cpu(array):
    try:
        return array.get()
    except:
        return array

def create_pypic(slices, context=None, **mesh_args):
    '''Factory method for PyPIC.pypic.PyPIC(_GPU) instances.

    Arguments:
        - slices: SliceSet instance for the longitudinal mesh quantities
        - mesh_args: dictionary with arguments for create_mesh method,
          defines the origin, distances and mesh size for the
          transverse plane (only!).
        - context: the device context (only for GPU use), e.g. via
          >>> from pycuda.autoinit import context

    Requires uniformly binned slices.
    '''
    mesh = create_mesh(slices=slices, **mesh_args)
    poisson_solver = solver_class(mesh, context)
    return pypic_algorithm_class(mesh, poisson_solver, context)


def create_mesh(mesh_origin, mesh_distances, mesh_size,
                slices=None, symmetrize_mesh_to_slices=False):
    '''Create a (PyPIC) rectangular mesh. The dimension is
    determined by the length of the list arguments.

    Arguments:
        - mesh_origin: origin from which the mesh is spanned,
          e.g. [x0, y0, z0]
        - mesh_distances: distance unit for each dimension,
          e.g. [dx, dy, dz]
        - mesh_size: list with number of nodes per dimension,
          e.g. [nx, ny, nz]

    Optional arguments:
        - slices: if a SliceSet instance is given,
          the previous mesh arguments are assumed to be transversal
          only and the longitudinal information as defined by the slices
          are added to the lists. Uses slices.gamma for the Lorentz
          boost to the beam frame!

    Requires uniformly binned slices.
    '''
    if slices:
        if slices.mode != 'uniform_bin':
            raise RuntimeError('Requires the slices to have uniformly '
                               'sized bins in order to create a '
                               'PyPIC.meshing.RectMesh3D.')
        mesh_origin = mesh_origin + [slices.z_cut_tail * slices.gamma]
        mesh_distances = mesh_distances + [# Lorentz trafo!
            (slices.z_cut_head - slices.z_cut_tail) / slices.n_slices *
            slices.gamma
        ]
        if symmetrize_mesh_to_slices:
            mesh_origin[-1] += mesh_distances[-1]/2.
        mesh_size = mesh_size + [slices.n_slices]
    dim = len(mesh_origin)
    if not dim == len(mesh_distances) == len(mesh_size):
        raise ValueError('All arguments for the mesh need to have as '
                         'many entries as the mesh should have dimensions!')
    mesh_class = getattr(meshing, 'RectMesh{dim}D'.format(dim=dim))
    return mesh_class(map(ensure_cpu, mesh_origin),
                      map(ensure_cpu, mesh_distances), mesh_size,
                      mathlib=pm)


def create_3dmesh_from_beam(beam, mesh_size, n_beam_sigma, slices=None,
                            symmetrize_mesh_to_slices=False):
    '''Create a (PyPIC) 3D rectangular mesh. The mesh is centered
    at the beam centroid and spans as many beam size sigma as indicated.

    Arguments:
        - mesh_size: list of length 2 or 3 with number of nodes per
          dimension,
          e.g. [nx, ny, nz]
        - n_beam_sigma: list of length 2 or 3 with factor of beam sigma
          determining the mesh radius for each dimension,
          e.g. [n_sigma_x, n_sigma_y, n_sigma_z]

    Optional arguments:
        - slices: if a SliceSet instance is given, only the first two
          entries from the mesh_size and n_beam_sigma arguments will be
          regarded. The given SliceSet determines the longitudinal
          meshing.
          Requires uniformly binned slices.
    '''
    length_x = 2. * beam.sigma_x() * n_beam_sigma[0]
    length_y = 2. * beam.sigma_y() * n_beam_sigma[1]
    mesh_origin = [beam.mean_x() - length_x / 2.,
                   beam.mean_y() - length_y / 2.]
    mesh_distances = [length_x / mesh_size[0], length_y / mesh_size[1]]

    if slices is None:
        length_z = 2. * beam.gamma * beam.sigma_z() * n_beam_sigma[2]
        mesh_origin += [beam.gamma * beam.mean_z() - length_z / 2.]
        mesh_distances += [length_z / mesh_size[2]]
        if symmetrize_mesh_to_slices:
            mesh_origin[-1] += mesh_distances[-1]/2.

    return create_mesh(mesh_origin, mesh_distances, mesh_size, slices,
                       symmetrize_mesh_to_slices)
