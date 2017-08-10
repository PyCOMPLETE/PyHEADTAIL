'''A FieldMap represents a static (electric) field defined in the lab
frame which interacts with a particle beam distribution in a weak-strong
approach. Potential applications include e.g. frozen space charge or
(weak-strong) beam-beam interaction with arbitrary beam distributions.

The FieldMap class seeks to generalise the Transverse_Efield_map (being
fixed to explicit 2D slice-by-slice treatment) to include 3D treatment
and extend usage to both CPU and GPU hardware architectures.
For these purposes FieldMap uses the PyPIC/GPU module (which is
based on PyPIClib).

PyPIC can be found under https://github.com/PyCOMPLETE/PyPIC .

@authors: Adrian Oeftiger
@date:    13.06.2017
'''

from __future__ import division, print_function

from scipy.constants import c

from . import Element

from ..general import pmath as pm
from ..gpu.pypic import make_PyPIC

class FieldMap(Element):
    '''This static field in the lab frame applies kicks to the beam
    distribution in a weak-strong interaction model.
    '''
    def __init__(self, length, mesh, fields, wrt_beam_centroid=False,
                 *args, **kwargs):
        '''Arguments:
            - length: interaction length around the accelerator over
              which the force of the field is integrated.
            - mesh: Mesh instance from PyPIC meshing with dimension 2
              or 3.
            - fields: list of arrays with field values defined over the
              mesh nodes. The field arrays need to have the same
              dimension as the mesh. Also the shape of the field arrays
              needs to coincide with the mesh.shape . The list of arrays
              can have 1 to 3 entries in the order of [E_x, E_y, E_z].
              Use as many entries as beam planes that you want to apply
              the field kicks to.
            - wrt_beam_centroid: if true, the beam centroid will be set
              to zero during the calculation of the field kicks.

        NB 1: FieldMap instances should be initialised in the proper
        context. If the FieldMap will track on the GPU, it should be
        initiated within a GPU context:
        >>> with PyHEADTAIL.general.contextmanager.GPU(beam) as cmg:
        >>>     fieldmap = FieldMap(...)

        NB 2: fields defined in the beam frame need to be Lorentz
        transformed to the lab frame. This is the case e.g. for fields
        determined by the particle-in-cell algorithm of PyPIC (where the
        longitudinal meshing includes the stretched beam distribution):

        1. charge density in beam frame:
        >>> mesh_charges = pypic.particles_to_mesh(
                beam.x, beam.y, beam.z_beamframe, charge=beam.charge)
        >>> rho = mesh_charges / mesh.volume_elem

        2. electrostatic potential in beam frame:
        >>> phi = pypic.poisson_solve(rho)

        3. electric fields in beam frame via gradient (here for 3D):
        >>> E_x, E_y, E_z = pypic.get_electric_fields(phi)

        4. Lorentz transform to lab frame:
        >>> E_x /= beam.gamma
        >>> E_y /= beam.gamma
        >>> E_z /= beam.gamma # longitudinal: need gamma^-2, however,
            # the gradient in pypic.get_electric_fields already
            # includes one gamma factor in d/dz_beamframe

        Use these lab frame fields [E_x, E_y, E_z] in the FieldMap
        argument. Attention in the 2.5D slice-by-slice case with the
        volume element (which should be a 2D area and not a 3D volume).
        '''
        self.length = length
        self.pypic = make_PyPIC(
            poissonsolver=None,
            gradient=lambda *args, **kwargs: None,
            mesh=mesh)
        self.fields = map(pm.ensure_same_device, fields)
        self.wrt_beam_centroid = wrt_beam_centroid

    def track(self, beam):
        # prepare argument for PyPIC mesh to particle interpolation
        mx, my, mz = 0, 0, 0
        if self.wrt_beam_centroid:
            mx, my, mz = beam.mean_x(), beam.mean_y(), beam.mean_z()
        mp_coords = [beam.x - mx,
                     beam.y - my,
                     beam.z - mz] # zip will cut to #fields

        mesh_fields_and_mp_coords = zip(self.fields, mp_coords)

        # electric fields at each particle position in lab frame [V/m]
        part_fields = self.pypic.field_to_particles(*mesh_fields_and_mp_coords)

        # integrate over dt, p0 comes from kicking xp=p_x/p0 instead of p_x
        kick_factor = self.length / (beam.beta*c) * beam.charge / beam.p0

        # apply kicks for 1-3 planes depending on #entries in fields
        for beam_momentum, force_field in zip(['xp', 'yp', 'zp'], part_fields):
            val = getattr(beam, beam_momentum)
            setattr(beam, beam_momentum, val + force_field * kick_factor)
        # for 3D, the for loop explicitly does:
        # beam.xp += part_fields[0] * kick_factor
        # beam.yp += part_fields[1] * kick_factor
        # beam.dp += part_fields[2] * kick_factor


class FieldMapSliceWise(FieldMap):
    '''As for the FieldMap, this represents a static two-dimensional
    transverse field in the lab frame. Kicks are applied in a
    slice-by-slice manner to the beam distribution in a weak-strong
    interaction model. The same field is applied to all slices while
    being multiplied by the local line charge density [Coul/m].

    A possible application is a slice-by-slice frozen space charge
    model.
    '''
    def __init__(self, slicer, *args, **kwargs):
        '''Arguments in addition to FieldMap arguments:
            - slicer: determines the longitudinal discretisation for the
              local line charge density, with which the field is
              multiplied at each track call.

        NB: mesh needs to be a two-dimensional mesh describing the
        discrete domain of the transverse fields.

        NB2: wrt_beam_centroid=True is implemented as a slice-by-slice
        transverse centring of the beam as opposed to the superclass
        FieldMap's 3D implementation, which sets the entire beam
        centroid including the longitudinal centre-of-gravity to the
        zero origin.

        NB3: the field values should be charge-density-normalised as
        they are multiplied by the line charge density for each slice,
        c.f. e.g. the Bassetti-Erskine formula without Q (as in the
        spacecharge module's
        TransverseGaussianSpaceCharge.get_efieldn()).
        '''
        self.slicer = slicer
        super(FieldMapSliceWise, self).__init__(*args, **kwargs)

        # require 2D!
        assert self.pypic.mesh.dimension == 2, \
            'mesh needs to be two-dimensional!'
        assert all(map(lambda f: f.ndim == 2, self.fields)), \
            'transverse field components need to be two-dimensional arrays!'
        #

    def track(self, beam):
        # prepare argument for PyPIC mesh to particle interpolation
        mx, my = 0, 0
        if self.wrt_beam_centroid:
            slices = beam.get_slices(
                self.slicer, statistics=["mean_x", "mean_y"])
            mx = slices.convert_to_particles(slices.mean_x)
            my = slices.convert_to_particles(slices.mean_y)
        else:
            slices = beam.get_slices(self.slicer)

        mp_coords = [beam.x - mx,
                     beam.y - my,
                     beam.z] # zip will cut to #fields
        mesh_fields_and_mp_coords = zip(self.fields, mp_coords)

        # electric fields at each particle position in lab frame [V/m]
        part_fields = self.pypic.field_to_particles(*mesh_fields_and_mp_coords)

        # weigh electric field with slice line charge density;
        # integrate over dt, p0 comes from kicking xp=p_x/p0 instead of p_x
        lambda_z = slices.convert_to_particles(slices.lambda_z(smoothen=False))
        kick_factor = (self.length / (beam.beta*c) * beam.charge / beam.p0
                       * lambda_z)

        # apply kicks for 1-3 planes depending on #entries in fields
        for beam_momentum, force_field in zip(['xp', 'yp', 'dp'], part_fields):
            val = getattr(beam, beam_momentum)
            setattr(beam, beam_momentum, val + force_field * kick_factor)
        # for 3D, the for loop explicitly does:
        # beam.xp += part_fields[0] * kick_factor
        # beam.yp += part_fields[1] * kick_factor
        # beam.dp += part_fields[2] * kick_factor
