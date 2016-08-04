'''Uses particle-in-cell algorithms from PyPIC for
space charge modelling in transverse 2.5D and 3D.

PyPIC can be found under
https://github.com/PyCOMPLETE/PyPIC .
NB: the (feature/redesign) branch is required for this!

@authors: Adrian Oeftiger
@date: 18.01.2016
'''

from __future__ import division

import numpy as np
from scipy.constants import c

from . import Element

from ..general import pmath as pm


def align_particles(beam, mesh_3d):
    '''Sort all particles by their mesh node IDs.'''
    ids = mesh_3d.get_node_ids(beam.x, beam.y, beam.z_beamframe)
    permutation = pm.argsort(ids)
    beam.reorder(permutation)
    # node id array has changed by now!

def get_bounds(beam, mesh_3d):
    '''Determine indices of sorted particles for each cell, i.e.
    lower and upper index bounds.
    '''
    seq = pm.seq(mesh_3d.n_nodes)
    ids = mesh_3d.get_node_ids(beam.x, beam.y, beam.z_beamframe)
    lower_bounds = pm.searchsortedleft(ids, seq)
    upper_bounds = pm.searchsortedright(ids, seq)
    return lower_bounds, upper_bounds

class SpaceChargePIC(Element):
    '''Transverse slice-by-slice (2.5D) or full (3D) space charge using
    a particle-in-cell algorithm via PyPIC. Uses a fixed 3D mesh
    with respect to beam.z_beamframe , i.e. the mesh does not adapt
    and remains constant.
    '''

    def __init__(self, length, pypic_algorithm, sort_particles=False,
                 *args, **kwargs):
        '''Arguments:
            - length: interaction length over which the space charge
              force is integrated.
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
        self.length = length
        self.pypic = pypic_algorithm
        self.sort_particles = sort_particles
        self.is_25D = getattr(self.pypic.poissonsolver, 'is_25D', False)
        if self.pypic.mesh.dimension != 3:
            raise RuntimeError(
                '2.5D / 3D space charge requires a three-dimensional mesh!')

    def track(self, beam, pypic_state=None):
        mesh = self.pypic.mesh

        solve_kwargs = {
            'charge': beam.charge_per_mp,
            'state': pypic_state,
        }
        if self.is_25D:
            # 2.5D: macro-particle charge becomes line density in beam frame
            # (in 3D this is implicit via rho=mesh_charges/mesh_3d.volume_elem)
            solve_kwargs['charge'] /= mesh.dz

        if self.sort_particles:
            align_particles(beam, mesh)

            solve_kwargs['lower_bounds'], solve_kwargs['upper_bounds'] = \
                get_bounds(beam, mesh)

        # electric fields for each particle in beam frame [V/m]
        force_fields = self.pypic.pic_solve(
            beam.x, beam.y, beam.z_beamframe, **solve_kwargs)

        # we want force F = q * (1 - beta^2) E_r where E_r is in lab frame
        # --> Lorentz boost E_r from beam frame to lab frame (*gamma)
        # --> include magnetic fields (1 - beta^2) = 1/gamma^2
        # ==> overall factor 1/gamma
        force_fields[0] /= beam.gamma
        force_fields[1] /= beam.gamma

        # integrate over dt and apply the force to each charged particle,
        # p0 comes from kicking xp=p_x/p0 instead of p_x
        kick_factor = (self.length / (beam.beta*c) * beam.charge / beam.p0)

        beam.xp += force_fields[0] * kick_factor
        beam.yp += force_fields[1] * kick_factor
        if not self.is_25D:
            # need 1/gamma^2: one gamma factor is contained in d/dz_beamframe
            # gradient in PyPIC, another gamma factor included here:
            beam.dp += force_fields[2] * kick_factor/beam.gamma

