'''Uses particle-in-cell algorithms from PyPIC for
space charge modelling in transverse 2.5D and 3D.

PyPIC can be found under
https://github.com/PyCOMPLETE/PyPIC .
NB: the (feature/redesign) branch is required for this!

@authors: Adrian Oeftiger
@date: 18.01.2016
'''

from __future__ import division, print_function

from scipy.constants import c

from . import Element

from ..general import pmath as pm
from ..field_maps.field_map import FieldMapSliceWise
from ..gpu.pypic import make_PyPIC
from pypic_factory import create_mesh
from spacecharge import TransverseGaussianSpaceCharge

import numpy as np

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
                 pic_dtype=np.float64, *args, **kwargs):
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
            - pic_dtype: can be np.float32 or np.float64 and determines
              (for sort_particles == False) which atomicAdd should be
              used on the GPU. On GPUs with computing capability <v6.0
              the double precision atomicAdd is not hardware accelerated
              and thus much slower.

              (NB: sort_particles=True is necessarily required for the
               PyPIC_GPU.sorted_particles_to_mesh method.)
        '''
        self.length = length
        self.pypic = pypic_algorithm
        self.sort_particles = sort_particles
        self.pic_dtype = pic_dtype
        self.is_25D = getattr(self.pypic.poissonsolver, 'is_25D', False)
        if self.pypic.mesh.dimension != 3:
            raise RuntimeError(
                '2.5D / 3D space charge requires a three-dimensional mesh!')

    def track(self, beam, pypic_state=None):
        mesh = self.pypic.mesh

        solve_kwargs = {
            'charge': beam.charge_per_mp,
            'state': pypic_state,
            'dtype': self.pic_dtype,
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


class SpaceChargePIC_Adaptive25D(SpaceChargePIC):
    '''Transverse slice-by-slice (2.5D) space charge using a
    particle-in-cell algorithm via PyPIC. Uses a 3D mesh with respect
    to beam.z_beamframe . The slices along the beam are continuously
    centred to the grid. Optionally, the grid size can be adapted to
    the most extreme RMS beam size along the slices if a relative change
    of the beam size is fixed via the argument sigma_rtol. If sigma_rtol
    is None, the grid size is not updated and remains constant.
    '''
    def __init__(self, length, slicer, beam,#=None, sigma_x=None, sigma_y=None,
                 n_mesh_sigma=[6, 6], mesh_size=[1024, 1024],
                 sigma_rtol=None, sort_particles=False, pic_dtype=np.float64,
                 save_memory=False, *args, **kwargs):
        '''Arguments:
            - length: interaction length over which the space charge
              force is integrated.
            - beam: the Particles instance of which sigma_x, sigma_y and
              gamma are evaluated to compute the field map.
              Cannot provide sigma_x and sigma_y arguments when
              giving beam argument.
            - sigma_x, sigma_y: the horizontal and vertical RMS width of
              the transverse Gaussian distribution modelling the beam
              distribution. Either provide both sigma_x and sigma_y
              as an argument or the beam instead.

        Optional arguments:
            - n_mesh_sigma: 2-list of number of beam RMS values in
              [x, y] to span across half the mesh width for the
              field interpolation.
            - mesh_size: 2-list of number of mesh nodes per transverse
              plane [x, y].
            - sigma_rtol: relative tolerance (float between 0 and 1)
              which is compared between the stored fields RMS size and
              the tracked beam RMS size, it determines when the fields
              are recomputed. For sigma_rtol == None (default value)
              the fields remain constant. E.g. sigma_rtol = 0.05 means
              if beam.sigma_x changes by more than 5%, the fields will
              be recomputed taking into account the new beam.sigma_x.
              A negative sigma_rtol will lead to continuous field
              updates at every track call.
            - sort_particles: determines whether to sort the particles
              by their mesh ID. This may speed up the PyPIC
              particle-to-mesh and mesh-to-particles methods
              due to coalesced memory access, especially on the GPU
              (test the timing for your parameters though!).
            - pic_dtype: can be np.float32 or np.float64 and determines
              (for sort_particles == False) which atomicAdd should be
              used on the GPU. On GPUs with computing capability <v6.0
              the double precision atomicAdd is not hardware accelerated
              and thus much slower.
            - save_memory: True means treat one slice at a time which
              saves memory but is slower, False means treat all slices
              simultaneously (faster).

              (NB: sort_particles=True is necessarily required for the
               PyPIC_GPU.sorted_particles_to_mesh method.)

        NB: SpaceChargePIC_Adaptive25D instances should be initialised
        in the proper context. If the SpaceChargePIC_Adaptive25D will
        track on the GPU, it should be initiated within a GPU context:
        >>> with PyHEADTAIL.general.contextmanager.GPU(beam):
        >>>     pic_sc_node = SpaceChargePIC_Adaptive25D(...)
        '''

        # TODO:
        # - do not check beam emittance but instead slice emittances
        # - do this check on the CPU/GPU staying on the same device
        #   instead of always CPU
        # - update mesh and green's function of poisson solver only
        # - allow non-GPU poisson solvers...

        self._wrt_beam_centroid = True
        self._n_mesh_sigma = n_mesh_sigma
        self.sigma_rtol = sigma_rtol
        self.save_memory = save_memory
        self.slicer = slicer

        sigma_x = pm.ensure_CPU(beam.sigma_x())
        sigma_y = pm.ensure_CPU(beam.sigma_y())
        # if beam is not None:
        #     if sigma_x is not None or sigma_y is not None:
        #         raise ValueError('SpaceChargePIC_Adaptive25D accepts either '
        #                          'beam as an argument or both sigma_x '
        #                          'and sigma_y. Do not mix these args!')
        #     sigma_x = pm.ensure_CPU(beam.sigma_x())
        #     sigma_y = pm.ensure_CPU(beam.sigma_y())
        # else:
        #     if sigma_x is None or sigma_y is None:
        #         raise ValueError('SpaceChargePIC_Adaptive25D requires both '
        #                          'sigma_x and sigma_y as arguments. '
        #                          'Alternatively provide a beam whose sigma are '
        #                          'evaluated.')

        mesh = create_mesh(
            mesh_origin=[-n_mesh_sigma[0] * sigma_x,
                         -n_mesh_sigma[1] * sigma_y],
            mesh_distances=[sigma_x * 2 * n_mesh_sigma[0] / mesh_size[0],
                            sigma_y * 2 * n_mesh_sigma[1] / mesh_size[1]],
            mesh_size=mesh_size,
            slices=beam.get_slices(self.slicer)
        )

        if pm.device is not 'GPU':
            raise NotImplementedError(
                'Currently only implemented for GPU, sorry!')

        # to be removed:
        from PyPIC.GPU.poisson_solver.FFT_solver import GPUFFTPoissonSolver_2_5D
        from pycuda.autoinit import context

        poissonsolver = GPUFFTPoissonSolver_2_5D(
            mesh=mesh, context=context, save_memory=self.save_memory)

        pypic = make_PyPIC(
            poissonsolver=poissonsolver,
            mesh=mesh)

        super(SpaceChargePIC_Adaptive25D, self).__init__(
            length=length, pypic_algorithm=pypic,
            sort_particles=sort_particles, pic_dtype=np.float64,
            *args, **kwargs)

    @property
    def sigma_x(self):
        return -self.pypic.mesh.x0 / self._n_mesh_sigma[0]

    @property
    def sigma_y(self):
        return -self.pypic.mesh.y0 / self._n_mesh_sigma[1]

    @property
    def mesh_size(self):
        return self.pypic.mesh.shape_r

    def update_grid(self, beam=None, sigma_x=None, sigma_y=None):
        '''Update the grid to new RMS sizes.
        Take either the RMS values from the provided Particles instance
        beam or use sigma_x and sigma_y.

        NB: make sure to call this method in the proper context
        (CPU/GPU), cf. comment in __init__ method.
        '''
        sigma_x = pm.ensure_CPU(sigma_x)
        sigma_y = pm.ensure_CPU(sigma_y)
        self.__init__(length=self.length, slicer=self.slicer, beam=beam,
                      sigma_x=sigma_x, sigma_y=sigma_y,
                      n_mesh_sigma=self._n_mesh_sigma,
                      mesh_size=self.mesh_size,
                      sigma_rtol=self.sigma_rtol,
                      sort_particles=self.sort_particles,
                      pic_dtype=self.pic_dtype,
                      save_memory=self.save_memory)

    def track(self, beam, pypic_state=None):
        # update field?
        if self.sigma_rtol is not None:
            x_too_large = (
                np.abs(pm.ensure_CPU(beam.sigma_x()) - self.sigma_x)
                    > self.sigma_rtol * self.sigma_x)
            y_too_large = (
                np.abs(pm.ensure_CPU(beam.sigma_y()) - self.sigma_y)
                    > self.sigma_rtol * self.sigma_y)
            if x_too_large or y_too_large:
                self.prints('SpaceChargePIC_Adaptive25D: '
                            'updating field maps...')
                self.update_grid(beam)

        mx, my = 0, 0
        if self._wrt_beam_centroid:
            slices = beam.get_slices(
                self.slicer, statistics=["mean_x", "mean_y"])
            mx = slices.convert_to_particles(slices.mean_x)
            my = slices.convert_to_particles(slices.mean_y)

        mp_coords = [beam.x - mx,
                     beam.y - my,
                     beam.z_beamframe] # zip will cut to #fields

        mesh = self.pypic.mesh

        solve_kwargs = {
            'charge': beam.charge_per_mp,
            'state': pypic_state,
            'dtype': self.pic_dtype,
        }

        # 2.5D: macro-particle charge becomes line density in beam frame
        # (in 3D this is implicit via rho=mesh_charges/mesh_3d.volume_elem)
        solve_kwargs['charge'] /= mesh.dz

        if self.sort_particles:
            align_particles(beam, mesh)

            solve_kwargs['lower_bounds'], solve_kwargs['upper_bounds'] = \
                get_bounds(beam, mesh)

        # electric fields for each particle in beam frame [V/m]
        force_fields = self.pypic.pic_solve(*mp_coords, **solve_kwargs)

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


class FrozenGaussianSpaceCharge25D(FieldMapSliceWise):
    '''Transverse slice-by-slice (2.5D) frozen space charge assuming
    a static transverse Gaussian distribution of a fixed RMS size.
    The present class is essentially a field_map.FieldMapSliceWise with
    a pre-filled Bassetti-Erskine formula computed field map
    (cf. spacecharge.TransverseGaussianSpaceCharge).
    The same electric transverse field is applied to all slices while
    being multiplied by the local line charge density [Coul/m].
    In particular, this means that the strength of the local field is
    self-consistent in the longitudinal plane but frozen in the
    transverse plane (with the fixed Gaussian electric field).

    This frozen space charge model essentially acts equivalently to an
    external magnet and fails to provide self-consistent treatment of
    space charge related effects like quadrupolar envelope breathing
    etc. Also the sum of all kicks does not vanish resulting in a net
    kick for the centroid as opposed to the SpaceChargePIC treatment.

    FrozenGaussianSpaceCharge25D can continuously recompute and update
    the fields during tracking of the beam if a relative change of the
    beam RMS sizes is fixed via the argument sigma_rtol. If sigma_rtol
    is None, the stored fields are not updated and remain constant.
    '''

    '''A tuple of field maps in the beam rest frame.'''
    fields = None

    def __init__(self, slicer, length, beam=None, sigma_x=None, sigma_y=None,
                 n_mesh_sigma=[6, 6], mesh_size=[1024, 1024],
                 sigma_rtol=None, *args, **kwargs):
        '''Arguments:
            - slicer: determines the longitudinal discretisation for the
              local line charge density, with which the field is
              multiplied at each track call.
            - length: interaction length around the accelerator over
              which the force of the field is integrated.
            - beam: the Particles instance of which sigma_x, sigma_y and
              gamma are evaluated to compute the field map.
              Cannot provide sigma_x and sigma_y arguments when
              giving beam argument.
            - sigma_x, sigma_y: the horizontal and vertical RMS width of
              the transverse Gaussian distribution modelling the beam
              distribution. Either provide both sigma_x and sigma_y
              as an argument or the beam instead.

        Optional arguments:
            - n_mesh_sigma: 2-list of number of beam RMS values in
              [x, y] to span across half the mesh width for the
              field interpolation.
            - mesh_size: 2-list of number of mesh nodes per transverse
              plane [x, y].
            - sigma_rtol: relative tolerance (float between 0 and 1)
              which is compared between the stored fields RMS size and
              the tracked beam RMS size, it determines when the fields
              are recomputed. For sigma_rtol == None (default value)
              the fields remain constant. E.g. sigma_rtol = 0.05 means
              if beam.sigma_x changes by more than 5%, the fields will
              be recomputed taking into account the new beam.sigma_x.
              A negative sigma_rtol will lead to continuous field
              updates at every track call.

        NB: FrozenGaussianSpaceCharge25D instances should be initialised
        in the proper context. If the FrozenGaussianSpaceCharge25D will
        track on the GPU, it should be initiated within a GPU context:
        >>> with PyHEADTAIL.general.contextmanager.GPU(beam):
        >>>     frozen_sc_node = FrozenGaussianSpaceCharge25D(...)
        '''
        wrt_beam_centroid = True
        self._n_mesh_sigma = n_mesh_sigma
        self.sigma_rtol = sigma_rtol

        if beam is not None:
            if sigma_x is not None or sigma_y is not None:
                raise ValueError('FrozenGaussianSpaceCharge25D accepts either '
                                 'beam as an argument or both sigma_x '
                                 'and sigma_y. Do not mix these args!')
            sigma_x = pm.ensure_CPU(beam.sigma_x())
            sigma_y = pm.ensure_CPU(beam.sigma_y())
        else:
            if sigma_x is None or sigma_y is None:
                raise ValueError('FrozenGaussianSpaceCharge25D requires both '
                                 'sigma_x and sigma_y as arguments. '
                                 'Alternatively provide a beam whose sigma are '
                                 'evaluated.')

        mesh = create_mesh(
            mesh_origin=[-n_mesh_sigma[0] * sigma_x,
                         -n_mesh_sigma[1] * sigma_y],
            mesh_distances=[sigma_x * 2 * n_mesh_sigma[0] / mesh_size[0],
                            sigma_y * 2 * n_mesh_sigma[1] / mesh_size[1]],
            mesh_size=mesh_size,
        )

        # calculate Bassetti-Erskine formula on either CPU or GPU:
        ## prepare arguments for the proper device:
        xg, yg = map(pm.ensure_same_device, np.meshgrid(
            np.linspace(mesh.x0, mesh.x0 + mesh.dx*mesh.nx, mesh.nx),
            np.linspace(mesh.y0, mesh.y0 + mesh.dy*mesh.ny, mesh.ny)
        ))
        sigma_x, sigma_y = map(pm.ensure_same_device, [sigma_x, sigma_y])
        ## compute fields
        be_sc = TransverseGaussianSpaceCharge(None, None)
        fields_beamframe = be_sc.get_efieldn(xg, yg, 0, 0, sigma_x, sigma_y)

        super(FrozenGaussianSpaceCharge25D, self).__init__(
            slicer=slicer, length=length, mesh=mesh,
            fields=fields_beamframe,
            wrt_beam_centroid=wrt_beam_centroid, *args, **kwargs)

    @property
    def sigma_x(self):
        return -self.pypic.mesh.x0 / self._n_mesh_sigma[0]

    @property
    def sigma_y(self):
        return -self.pypic.mesh.y0 / self._n_mesh_sigma[1]

    @property
    def mesh_size(self):
        return self.pypic.mesh.shape_r

    def update_field(self, beam=None, sigma_x=None, sigma_y=None):
        '''Update the stored Gaussian field to new RMS sizes.
        Take either the RMS values from the provided Particles instance
        beam or use sigma_x and sigma_y.

        NB: make sure to call this method in the proper context
        (CPU/GPU), cf. comment in __init__ method.
        '''
        sigma_x = pm.ensure_CPU(sigma_x)
        sigma_y = pm.ensure_CPU(sigma_y)
        self.__init__(slicer=self.slicer, length=self.length, beam=beam,
                      sigma_x=sigma_x, sigma_y=sigma_y,
                      n_mesh_sigma=self._n_mesh_sigma,
                      mesh_size=self.mesh_size,
                      sigma_rtol=self.sigma_rtol)

    def track(self, beam):
        # in contrast to FieldMap's length, here length becomes a float
        # storing the interaction length including the Lorentz transform,
        # so the integrated part of the accelerator x gamma**-2.
        # Lorentz trafo to lab frame + magnetic fields:
        # F = q E_beam / gamma = q (E_n_BassErsk * lambda_beam) / gamma
        #   = q E_n_BassErsk * lambda_lab / gamma^2
        # this trick avoids multiplying the field arrays twice:
        interaction_length = self.length
        self.length *= beam.gamma**-2

        # update field?
        if self.sigma_rtol is not None:
            x_too_large = (
                np.abs(pm.ensure_CPU(beam.sigma_x()) - self.sigma_x)
                    > self.sigma_rtol * self.sigma_x)
            y_too_large = (
                np.abs(pm.ensure_CPU(beam.sigma_y()) - self.sigma_y)
                    > self.sigma_rtol * self.sigma_y)
            if x_too_large or y_too_large:
                self.prints('FrozenGaussianSpaceCharge25D: '
                            'updating field maps...')
                self.update_field(beam)

        super(FrozenGaussianSpaceCharge25D, self).track(beam)

        self.length = interaction_length
