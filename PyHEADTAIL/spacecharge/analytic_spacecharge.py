'''Transverse frozen space charge models based on analytic
beam field expressions with static and adaptive approaches.

@authors: Adrian Oeftiger
@date: 24.03.2020
'''

import numpy as np

from scipy.constants import c, epsilon_0

import PyHEADTAIL.general.pmath as pm
from PyHEADTAIL.spacecharge.spacecharge import (
    TransverseGaussianSpaceCharge)

sqrt2pi = np.sqrt(2 * np.pi)

class AnalyticTransverseGaussianSC(TransverseGaussianSpaceCharge):
    '''Analytic transverse space charge fields based on
    3D Gaussian distribution (Bassetti-Erskine formula).

    Can track with respect to the bunch centroid and can cumulatively
    update transverse bunch size every n steps.

    As opposed to self-consistent longitudinal line charge density
    through slicing in TransverseGaussianSpaceCharge class, this
    AnalyticTransverseGaussianSC class assumes a longitudinal Gaussian
    line charge density based on a given RMS sigma_z.
    '''

    'Horizontal closed orbit offset in case of wrt_centroid=False.'
    x_co = 0
    'Vertical closed orbit offset in case of wrt_centroid=False.'
    y_co = 0
    'Longitudinal closed orbit offset in case of wrt_centroid=False.'
    z_co = 0

    def __init__(self, length, sigma_x, sigma_y, sigma_z,
                 wrt_centroid=True, update_every=None,
                 *args, **kwargs):
        '''Initialise analytic Gaussian (Bassetti-Erskine)
        space charge element based on analytic 3D Gaussian
        formula.

        Arguments:
            - length: path length interval along which the
              space charge force is integrated.
            - sigma_x: horizontal RMS bunch size
            - sigma_y: vertical RMS bunch size
            - sigma_z: longitudinal RMS bunch size
            - wrt_centroid: if True, the space charge kick
              is centred around the instantaneous 3D
              bunch centroid computed from the macro-particle
              distribution (beam.mean_x() etc.).
              If False, the space charge field is centred at
              fixed transverse closed orbit values,
              cf. x_co, y_co, z_co.
            - update_every: if None, the initially set RMS
              beam sizes sigma_x,y,z remain constant.
              For a given finite integer n > 0, the bunch
              size is taken from the tracked beam and averaged
              over n subsequent track calls. At the n-th call,
              the stored sigma_x,y,z values are updated with
              the new averaged values.
              For update_every=1, the RMS beam sizes are updated
              at every track call.

        Attention: only works for a single bunch due to the
        assumed longitudinal Gaussian profile (extending to infinity).
        '''
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.wrt_centroid = wrt_centroid
        self.update_every = update_every

        self._update_counter = 0
        self._cum_sigma_x = 0
        self._cum_sigma_y = 0
        self._cum_sigma_z = 0

        super().__init__(
            slicer=None, length=length, sig_check=True,
            other_efieldn=self._efieldn_koelbig)

    def compute_lambda(self, z, total_charge):
        '''Compute the local line charge density [Coul/m] for
        the given longitudinal position z and total charge in
        the bunch total_charge (i.e.
        intensity * charge_per_particle) .

        Assumes a Gaussian bunch profile.

        Replace this function by another expression to change to
        an arbitrary longitudinal bunch profile.
        '''
        return total_charge * pm.exp(
                -z * z / (2 * self.sigma_z * self.sigma_z)
            ) / (sqrt2pi * self.sigma_z)

    def track(self, beam):
        n = self.update_every
        if n and n > 0:
            self._cum_sigma_x += beam.sigma_x() / n
            self._cum_sigma_y += beam.sigma_y() / n
            self._cum_sigma_z += beam.sigma_z() / n
            self._update_counter += 1
            if self._update_counter == n:
                self.sigma_x = self._cum_sigma_x
                self.sigma_y = self._cum_sigma_y
                self.sigma_z = self._cum_sigma_z
                self._update_counter = 0
                self._cum_sigma_x = 0
                self._cum_sigma_y = 0
                self._cum_sigma_z = 0

        prefactor = (beam.charge * self.length /
                     (beam.p0 * beam.betagamma * beam.gamma * c))

        if self.wrt_centroid:
            mean_x = beam.mean_x()
            mean_y = beam.mean_y()
            mean_z = beam.mean_z()
        else:
            mean_x = self.x_co
            mean_y = self.y_co
            mean_z = self.z_co

        en_x, en_y = self.get_efieldn(
            beam.x, beam.y, mean_x, mean_y,
            self.sigma_x, self.sigma_y)

        lmbda = self.compute_lambda(
            beam.z - mean_z, beam.intensity * beam.charge)

        beam.xp += (lmbda * en_x) * prefactor
        beam.yp += (lmbda * en_y) * prefactor

    @staticmethod
    def _efieldn_koelbig(x, y, sig_x, sig_y):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.
        Return (E_x / Q, E_y / Q).
        Assumes sig_x > sig_y and mean_x == 0 as well as mean_y == 0.
        For convergence reasons of the erfc, use only x > 0 and y > 0.
        Uses CERN library from K. Koelbig.
        '''
        sig_sqrt = TransverseGaussianSpaceCharge._sig_sqrt(sig_x, sig_y)
        w1re, w1im = pm.wofz(x/sig_sqrt, y/sig_sqrt)
        ex = pm.exp(-x*x / (2 * sig_x*sig_x) +
                    -y*y / (2 * sig_y*sig_y))
        w2re, w2im = pm.wofz(x * sig_y/(sig_x*sig_sqrt),
                             y * sig_x/(sig_y*sig_sqrt))
        pref = 1. / (2 * epsilon_0 * np.sqrt(np.pi) * sig_sqrt)
        return (w1im - ex * w2im) * pref, (w1re - ex * w2re) * pref
