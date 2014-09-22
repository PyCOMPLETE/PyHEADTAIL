'''
@authors: Adrian Oeftiger
@date:    12/09/2014
'''
from __future__ import division

from . import Element

import numpy as np
from scipy.constants import m_p, c, e, epsilon_0

class LongSpaceCharge(Element):
    '''
    Contains longitudinal space charge via Chao's expression:

    dp' = - e^2 * g * lambda'(z) / (2 * pi * eps_0 * gamma^2 * p_0)

    cf. the original HEADTAIL version.
    '''

    def __init__(self, slices, pipe_radius, time_step,
                 line_density_derivative_function=None):
        self.slices = slices
        if line_density_derivative_function:
            self.line_density_derivative = line_density_derivative_function
        else:
            self.line_density_derivative = slices.line_density_derivative_smooth
        self.pipe_radius = pipe_radius
        self.time_step = time_step

    def track(self, beam):
        '''
        Add the longitudinal space charge contribution to the beam's
        dp kick.
        '''
        lambda_prime = (self.line_density_derivative()[1] *
                        beam.n_particles_per_mp)
        slice_kicks = (self._prefactor(beam) * self._gfactor(beam) *
                       lambda_prime) * self.time_step

        p_id = self.slices.particles_within_cuts
        s_id = self.slices.slice_index_of_particle.take(p_id)

        beam.dp[p_id] -= slice_kicks.take(s_id)

    @staticmethod
    def _prefactor(beam):
        return e**2 / (2 * np.pi * epsilon_0 * beam.gamma**2 * beam.p0)

    def _gfactor(self, beam):
        beam_radius = (beam.sigma_x() + beam.sigma_y()) / 2
        return 1. / 3 + np.log(self.pipe_radius / beam_radius)
