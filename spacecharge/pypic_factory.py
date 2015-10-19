'''Factory for PyPIC particle-in-cell algorithms with FFT Poisson
solving.
Used for space charge modelling under pypic_spacecharge.

PyPIC can be found under
https://github.com/PyCOMPLETE/PyPIC .

@authors: Stefan Hegglin, Adrian Oeftiger
@date: 19.10.2015
'''

from __future__ import division
import numpy as np

from PyPIC import pypic, poisson_solver, meshing

i_gpu = True

if i_gpu:
    pypic_algorithm_class = pypic.PyPIC_GPU
    solver_class = poisson_solver.FFT_solver.GPUFFTPoissonSolver
    from pycuda import cumath as mathlib

else:
    pypic_algorithm_class = pypic.PyPIC
    def solver_class(mesh, context=None):
        if not np.isclose(mesh.dx, mesh.dy):
            raise ValueError('CPU version of the FFT Poisson Solver '
                             '(Gianni\'s version) only accepts square '
                             'cells with mesh.dx == mesh.dy')
        Dh = mesh.dx
        x_aper = 0.5 * mesh.nx * Dh
        y_aper = 0.5 * mesh.ny * Dh
        return poisson_solver.FFT_OpenBoundary_SquareGrid(
            x_aper, y_aper, Dh,
            # we do not want exterior 5 and 4 cells from PyPIC <= v1.03:
            ext_boundary=False,
        )
    mathlib = np

def create_pypic(slicer, mesh_args, context=None):
    '''Factory method for PyPIC.pypic.PyPIC(_GPU) instances.

    Arguments:
        - slicer: particles.slicer.Slicer instance
        - mesh_args: dictionary with arguments for create_mesh method,
          defines the origin, distances and mesh size for the
          transverse plane (only!).
        - context: the device context (only for GPU use), e.g. via
          >>> from pycuda.autoinit import context

    Requires the slicer to have uniformly sized bins.
    '''
    mesh = create_mesh(slicer=slicer, **mesh_args)
    poisson_solver = solver_class(mesh, context)
    return pypic_algorithm_class(mesh, poisson_solver, context)


def create_mesh(mesh_origin, mesh_distances, mesh_size, slicer=None):
    '''Create a (PyPIC) rectangular mesh. The dimension is
    determined by the length of the list arguments.

    Arguments:
        - mesh_origin: origin from which the mesh is spanned,
          e.g. [x0, y0, z0]
        - mesh_distances: distance unit for each dimension,
          e.g. [dx, dy, dz]
        - mesh_size: list with number of nodes per dimension,
          e.g. [nx, ny, nz]
        - slicer: if a particles.slicer.Slicer instance is given,
          the previous mesh arguments are assumed to be transversal
          only and the longitudinal information as defined by the slicer
          are added to the lists.

    Requires the slicer to have uniformly sized bins.
    '''
    if slicer:
        if slicer.mode != 'uniform_bin':
            raise RuntimeError('Requires the slicer to have uniformly '
                               'sized bins in order to create a '
                               'PyPIC.meshing.RectMesh3D.')
        mesh_origin.append(slicer.z_cut_tail)
        mesh_distances.append(slicer.slice_widths[0])
        mesh_size.append(slicer.n_slices)
    dim = len(mesh_origin)
    if not dim == len(mesh_distances) == len(mesh_size):
        raise ValueError('All arguments for the mesh need to have as '
                         'many entries as the mesh should have dimensions!')
    mesh_class = getattr(meshing, 'RectMesh{dim}D'.format(dim=dim))
    return mesh_class(*zip(mesh_origin, mesh_distances, mesh_size),
                      mathlib=mathlib)
