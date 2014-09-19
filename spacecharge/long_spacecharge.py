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

        dp_kicks = np.empty(beam.n_macroparticles)
        prev_index = 0
        for slice_index, next_index in enumerate(
                self.slices.first_particle_index_in_slice[1:]):
            dp_kicks[prev_index:next_index] = slice_kicks[slice_index]
            prev_index = next_index

        beam.dp -= dp_kicks

    @staticmethod
    def _prefactor(beam):
        return e**2 / (2 * np.pi * epsilon_0 * beam.gamma**2 * beam.p0)

    def _gfactor(self, beam):
        beam_radius = (beam.sigma_x() + beam.sigma_y()) / 2
        return 1. / 3 + np.log(self.pipe_radius / beam_radius)
