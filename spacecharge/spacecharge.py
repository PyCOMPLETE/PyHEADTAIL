'''
@authors: Adrian Oeftiger
@date:    12/09/2014
'''
from __future__ import division

from . import Element, clean_slices

import numpy as np
from scipy.constants import m_p, c, e, epsilon_0

from scipy.interpolate import splrep, splev

class LongSpaceCharge(Element):
    '''
    Contains longitudinal space charge via Chao's expression:

    dp' = - e^2 * g * lambda'(z) / (2 * pi * eps_0 * gamma^2 * p_0)

    cf. the original HEADTAIL version.
    '''

    def __init__(self, slicer, pipe_radius, time_step, *args, **kwargs):
        self.slicer = slicer
        self.pipe_radius = pipe_radius
        self.time_step = time_step
        # include slice_sigma

    @clean_slices
    def track(self, beam):
        '''
        Add the longitudinal space charge contribution to the beam's
        dp kick.
        '''
        slices = beam.get_slices(self.slicer,
                                 statistics=['sigma_x', 'sigma_y'])
        lambda_prime = (slices.line_density_derivative_gauss(sigma=3) *
                        beam.particlenumber_per_mp)
        slice_kicks = (self._prefactor(beam) * self._gfactor(beam) *
                       lambda_prime) * self.time_step

        p_id = slices.particles_within_cuts
        s_id = slices.slice_index_of_particle.take(p_id)

        beam.dp[p_id] -= slice_kicks.take(s_id)

    @staticmethod
    def _prefactor(sliceset):
        return e**2 / (4 * np.pi * epsilon_0 * sliceset.gamma**2 * sliceset.p0)

    def _gfactor(self, sliceset):
        beam_radius = (sliceset.sigma_x + sliceset.sigma_y) / 2
        return 0.5 + 2 * np.log(self.pipe_radius / beam_radius)

    def make_force(self, sliceset):
        '''Return the electric force field due to space charge
        of the given SliceSet instance as a function of z
        in units of Coul*Volt/metre.
        '''
        gfac_spline = splrep(sliceset.z_centers, self._gfactor(sliceset), s=0)
        def force(z):
            gfac = splev(z, gfac_spline, der=0)
            return (self._prefactor(sliceset) * gfac *
                    -sliceset.lambda_prime_z(z))
        return force

    def make_potential(self, sliceset):
        '''Return the electric potential energy due to space charge
        of the given SliceSet instance as a function of z
        in units of Coul*Volt.
        '''
        gfac_spline = splrep(sliceset.z_centers, self._gfactor(sliceset), s=0)
        def potential(z):
            gfac = splev(z, gfac_spline, der=0)
            return self._prefactor(sliceset) * gfac * sliceset.lambda_z(z)
        return potential
