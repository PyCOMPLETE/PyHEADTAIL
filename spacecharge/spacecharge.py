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

from errfff import errf as errf_f
errf = np.vectorize(errf_f)
from scipy.special import erfc
errfsp = lambda z: np.exp(-z**2) * erfc(z * -1j)


class LongSpaceCharge(Element):
    '''Contains longitudinal space charge (SC) via Chao's expression:

    dp' = - e^2 * g * lambda'(z) / (2 * pi * eps_0 * gamma^2 * p_0)

    cf. the original HEADTAIL version.
    '''

    def __init__(self, slicer, pipe_radius, length, n_slice_sigma=3,
                 *args, **kwargs):
        '''Arguments:
        - pipe_radius is the the radius of the vacuum pipe in metres.
        - length is an s interval (in metres) along which the SC force
        is integrated. Usually you want to set this to circumference
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
        '''Add the longitudinal space charge contribution to the beam's
        dp kick.
        '''
        slices = beam.get_slices(self.slicer,
                                 statistics=['sigma_x', 'sigma_y'])
        lambda_prime = slices.lambda_prime_bins(sigma=self.n_slice_sigma)
        slice_kicks = (self._prefactor(slices) * self._gfactor(slices) *
                       lambda_prime) * (self.length / beam.beta * c)

        kicks = slices.convert_to_particles(slice_kicks)
        beam.dp -= kicks
#        p_id = slices.particles_within_cuts
#        s_id = slices.slice_index_of_particle.take(p_id)

#        beam.dp[p_id] -= slice_kicks.take(s_id)

    @staticmethod
    def _prefactor(sliceset):
        return (sliceset.charge /
                (4.*np.pi*epsilon_0 * sliceset.gamma**2 * sliceset.p0))

    def _gfactor0(self, sliceset, directSC=0.67):
        """Geometry factor for long bunched bunches.
        Involved approximations:
        - transversely round beam
        - finite wall resistivity (perfectly conducting boundary)
        - geometry factor averaged along z 
        (considering equivalent linear longitudinal electric field)
        
        use directSC = 0.67 for further assumptions:
        - ellipsoidally bunched beam of uniform density
        - bunch length > 3/2 pipe radius
        
        use directSC = 0.5 for further assumptions:
        - continuous beam
        - low frequency disturbance (displacement currents neglected)
        - emittance dominated beam
        
        cf. Martin Reiser's discussion in 
        'Theory and Design of Charged Particle Beams'.
        """
        slice_radius = 0.5 * (sliceset.sigma_x + sliceset.sigma_y)
        # the following line prevents ZeroDivisionError for pencil slices
        slice_radius[slice_radius == 0] = exp(-0.25) * self.pipe_radius
        return directSC + 2. * log(self.pipe_radius / slice_radius)

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

class TransverseGaussianSpaceCharge(Element):
    '''
    Contains transverse space charge for a Gaussian configuration
    applying the Bassetti-Erskine electric field expression slice-wise.
    '''

    def __init__(self, slicer):
        self.slicer = slicer

    def track(self, beam):
        '''
        Add the transverse space charge contribution to the beam's
        transverse kicks.
        '''
        slices = beam.get_slices(self.slicer)
        Q_mp = beam.particlenumber_per_mp * beam.charge

        for p_id, Q_sl, sig_x, sig_y in zip(slices.particle_indices_by_slice,
                                            slices.charge_per_slice,
                                            slices.sigma_x(beam),
                                            slices.sigma_y(beam)):
            sig_sqrt = np.sqrt(2 * (sig_x**2 - sig_y**2))
            efields_x, efields_y = Q_sl * self.efieldn(
                beam.x[p_id], beam.y[p_id], sig_x, sig_y, sig_sqrt)

            beam.xp[p_id] += Q_mp * efields_x
            beam.yp[p_id] += Q_mp * efields_y



    @staticmethod
    def efieldn(x, y, sig_x, sig_y, sig_sqrt):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

        Return (E_x / Q, E_y / Q).

        Assumes sig_x > sig_y.
        '''
        # timing was ~208 us for:
        # x = np.arange(-1e-5, 1e-5, 1e-7)
        # y = np.empty(len(x))
        # sig_x = 1.2e-6
        # sig_y = 1e-6
        # sig_sqrt = np.sqrt(2 * (sig_x**2 - sig_y**2))
        w1 = errfsp((x + 1j * y) / sig_sqrt)
        ex = np.exp(-x**2 / (2 * sig_x**2) +
                    -y**2 / (2 * sig_y**2))
        w2 = errfsp(x * sig_y/(sig_x*sig_sqrt) +
                    y * sig_x/(sig_y*sig_sqrt) * 1j)
        val = (w1 - ex * w2) / (2 * epsilon_0 * np.sqrt(pi) * sig_sqrt)
        return val.imag, val.real

    @staticmethod
    def efieldn2(x, y, sig_x, sig_y, sig_sqrt):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

        Return (E_x / Q, E_y / Q).

        Assumes sig_x > sig_y.

        25%% slower than efieldn (for numpy vectors), uses CERN library.
        '''
        # timing was ~268 us for same situation as efieldn
        w1re, w1im = errf(x/sig_sqrt, y/sig_sqrt)
        ex = np.exp(-x**2 / (2 * sig_x**2) +
                    -y**2 / (2 * sig_y**2))
        w2re, w2im = errf(x * sig_y/(sig_x*sig_sqrt),
                          y * sig_x/(sig_y*sig_sqrt))
        pref = 1. / (2 * epsilon_0 * np.sqrt(pi) * sig_sqrt)
        return pref * (w1im - ex * w2im), pref * (w1re - ex * w2re)
