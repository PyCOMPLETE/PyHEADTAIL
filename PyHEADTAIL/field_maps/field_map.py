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

from . import Element

from ..gpu.pypic import make_PyPIC

class FieldMap(Element):
    '''This static (electric) field in the lab frame applies kicks
    to the beam distribution in a weak-strong interaction model.
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
              (NB: the field arrays need to be either numpy ndarray or
              pycuda GPUArray instances, depending on whether you want
              to apply the kicks on the CPU or on the GPU!)
            - wrt_beam_centroid: if true, the beam centroid will be set
              to zero during the calculation of the field kicks.

        NB: fields defined in the beam frame need to be Lorentz
        transformed to the lab frame. This particularly true for fields
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
            poissonsolver=None, gradient=lambda mesh: None, mesh=mesh)
        self.fields = fields
        self.wrt_beam_centroid = wrt_beam_centroid

    def track(self, beam):
        # prepare argument for PyPIC mesh to particle interpolation
        mp_coords = np.array([beam.x, beam.y, beam.z]) # zip will cut to #fields
        if self.wrt_beam_centroid:
            mp_coords -= np.array(
                [beam.mean_x(), beam.mean_y(), beam.mean_z()])
        mesh_fields_and_mp_coords = zip(self.fields, mp_coords)

        # electric fields at each particle position in lab frame [V/m]
        part_fields = self.pypic.field_to_particles(*mesh_fields_and_mp_coords)

        # integrate over dt, p0 comes from kicking xp=p_x/p0 instead of p_x
        kick_factor = (self.length / (beam.beta*c) * beam.charge / beam.p0)

        # apply kicks for 1-3 planes depending on #entries in fields
        for beam_momentum, force_field in zip(['xp', 'yp', 'zp'], part_fields):
            getattr(beam, beam_momentum) += force_field * kick_factor
        # beam.xp += part_fields[0] * kick_factor
        # beam.yp += part_fields[1] * kick_factor
        # beam.dp += part_fields[2] * kick_factor

