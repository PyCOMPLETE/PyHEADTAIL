'''
@authors: Adrian Oeftiger
@date:    12/09/2014
'''
from __future__ import division

from . import Element, clean_slices

import numpy as np
from scipy.constants import m_p, c, e, epsilon_0, pi

from scipy.interpolate import splrep, splev

log = np.log
exp = np.exp

class LongSpaceCharge(Element):
    '''
    Contains longitudinal space charge via Chao's expression:

    dp' = - e^2 * g * lambda'(z) / (2 * pi * eps_0 * gamma^2 * p_0)

    cf. the original HEADTAIL version.
    '''

    def __init__(self, slicer, pipe_radius, length, n_slice_sigma=3,
                 *args, **kwargs):
        '''Arguments:
        - pipe_radius is the the radius of the vacuum pipe in metres.
        - length is an s interval along which the space charge force
        is integrated. Usually you want to set this to the circumference
        in conjunction with the LongitudinalOneTurnMap RFSystems.
        - n_slice_sigma indicates the number of slices taken as a
        sigma for the Gaussian kernel that smoothens the line charge
        density derivative (see SliceSet.lambda_prime_bins for more
        info).
        '''
        self.slicer = slicer
        self.pipe_radius = pipe_radius
        self.length = length
        self.n_slice_sigma = n_slice_sigma
        self._gfactor = self._gfactor0

    @clean_slices
    def track(self, beam):
        '''
        Add the longitudinal space charge contribution to the beam's
        dp kick.
        '''
        slices = beam.get_slices(self.slicer,
                                 statistics=['sigma_x', 'sigma_y'])
        lambda_prime = slices.lambda_prime_bins(sigma=self.n_slice_sigma)
        slice_kicks = (self._prefactor(slices) * self._gfactor(slices) *
                       lambda_prime) * (self.length / beam.beta * c)

        p_id = slices.particles_within_cuts
        s_id = slices.slice_index_of_particle.take(p_id)

        beam.dp[p_id] -= slice_kicks.take(s_id)

    @staticmethod
    def _prefactor(sliceset):
        return (sliceset.charge /
                (4.*np.pi*epsilon_0 * sliceset.gamma**2 * sliceset.p0))

    def _gfactor0(self, sliceset):
        """Giovanni Rumolo has put 0.67 into HEADTAIL instead of 0.5."""
        slice_radius = 0.5 * (sliceset.sigma_x + sliceset.sigma_y)
        slice_radius[slice_radius == 0] = exp(-0.25) * self.pipe_radius
        return 0.5 + 2. * log(self.pipe_radius / slice_radius)

    def make_force(self, sliceset):
        '''Return the electric force field due to space charge
        of the given SliceSet instance as a function of z
        in units of Coul*Volt/metre.
        '''
        gfac_spline = splrep(sliceset.z_centers, self._gfactor(sliceset), s=0)
        def force(z):
            gfac = splev(z, gfac_spline, ext=1)
            return (self._prefactor(sliceset) * gfac *
                    -sliceset.lambda_prime_z(z) * sliceset.p0)
        return force

    def make_potential(self, sliceset):
        '''Return the electric potential energy due to space charge
        of the given SliceSet instance as a function of z
        in units of Coul*Volt.
        '''
        gfac_spline = splrep(sliceset.z_centers, self._gfactor(sliceset), s=0)
        def potential(z):
            gfac = splev(z, gfac_spline, ext=1)
            return (self._prefactor(sliceset) * gfac *
                    sliceset.lambda_z(z) * sliceset.p0)
        return potential
