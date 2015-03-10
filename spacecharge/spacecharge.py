'''
@authors: Adrian Oeftiger
@date:    12/09/2014
'''
from __future__ import division

from . import Element, clean_slices

import numpy as np
from scipy.constants import m_p, c, e, epsilon_0, pi

class LongSpaceCharge(Element):
    '''
    Contains longitudinal space charge via Chao's expression:

    dp' = - e^2 * g * lambda'(z) / (2 * pi * eps_0 * gamma^2 * p_0)

    cf. the original HEADTAIL version.
    '''

    def __init__(self, slicer, pipe_radius, time_step, *args, **kwargs):
        """Attention: Do not forget to adapt time_step during
        acceleration, as the revolution period changes.
        """
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
        charge = beam.particlenumber_per_mp * beam.charge
        slices = beam.get_slices(self.slicer)
        lambda_prime = (slices.line_density_derivative_gauss(sigma=3) *
                        charge)
        slice_kicks = (self._prefactor(beam) * self._gfactor0(beam) *
                       lambda_prime) * self.time_step

        p_id = slices.particles_within_cuts
        s_id = slices.slice_index_of_particle.take(p_id)

        beam.dp[p_id] -= slice_kicks.take(s_id)

    @staticmethod
    def _prefactor(beam):
        return e**2 / (4. * np.pi * epsilon_0 * beam.gamma**2 * beam.p0)

    def _gfactor0(self, beam):
        """Giovanni Rumolo has put 0.67 into HEADTAIL instead of 0.5."""
        beam_radius = (beam.sigma_x() + beam.sigma_y()) / 2
        return 0.5 + 2. * np.log(self.pipe_radius / beam_radius)
