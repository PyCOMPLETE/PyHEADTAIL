'''
@authors: Adrian Oeftiger
@date:    12/09/2014
'''
from __future__ import division

from . import Element, clean_slices

import numpy as np
from scipy.constants import m_p, c, e, epsilon_0, pi

from scipy.interpolate import splrep, splev

class LongSpaceCharge(Element):
    '''
    Contains longitudinal space charge via Chao's expression:

    dp' = - e^2 * g * lambda'(z) / (2 * pi * eps_0 * gamma^2 * p_0)

    cf. the original HEADTAIL version.
    '''

    def __init__(self, slicer, pipe_radius, time_step, n_slice_sigma=3,
                 *args, **kwargs):
        '''Arguments:
        - pipe_radius is the the radius of the vacuum pipe in metres.
        - time_step is the time duration over which the space charge
        should be applied. Usually you want to set this to the
        revolution period (if longitudinal space charge is applied once
        per turn).
        Attention: Do not forget to adapt the time_step during
        acceleration, as the revolution period changes.
        - n_slice_sigma indicates the number of slices taken as a
        sigma for the Gaussian kernel that smoothens the line charge
        density derivative (see SliceSet.lambda_prime_bins for more
        info).
        '''
        self.slicer = slicer
        self.pipe_radius = pipe_radius
        self.time_step = time_step
        self.n_slice_sigma = n_slice_sigma

    @clean_slices
    def track(self, beam):
        '''
        Add the longitudinal space charge contribution to the beam's
        dp kick.
        '''
        charge = beam.particlenumber_per_mp * beam.charge
        slices = beam.get_slices(self.slicer,
                                 statistics=['sigma_x', 'sigma_y'])
        lambda_prime = (slices.line_density_derivative_gauss(self.n_slice_sigma) *
                        charge)
        slice_kicks = (self._prefactor(slices) * self._gfactor0(slices) *
                       lambda_prime) * self.time_step

        p_id = slices.particles_within_cuts
        s_id = slices.slice_index_of_particle.take(p_id)

        beam.dp[p_id] -= slice_kicks.take(s_id)

    @staticmethod
    def _prefactor(sliceset):
        return e / (4. * np.pi * epsilon_0 * sliceset.gamma**2 * sliceset.p0)

    def _gfactor0(self, sliceset):
        """Giovanni Rumolo has put 0.67 into HEADTAIL instead of 0.5."""
        beam_radius = 0.5 * (sliceset.sigma_x + beam.sigma_y)
        return 0.5 + 2. * np.log(self.pipe_radius / beam_radius)

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
