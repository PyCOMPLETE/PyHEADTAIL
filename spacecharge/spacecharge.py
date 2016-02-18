'''
@authors: Adrian Oeftiger
@date:    17/04/2015
'''
from __future__ import division

from . import Element, clean_slices

import numpy as np
from scipy.constants import c, epsilon_0, pi

from scipy.interpolate import splrep, splev
from functools import wraps

log = np.log
exp = np.exp
take = np.take

try:
    from errfff import errf as errf_f
    errf = np.vectorize(errf_f)
except ImportError:
    errf = None
from scipy.special import erfc, wofz
def errfadd(z):
    return np.exp(-z**2) * erfc(z * -1j)


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
                       lambda_prime) * (self.length / (beam.beta * c))

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
        - transversely averaged contribution

        use directSC = 0.5 for further assumptions:
        - continuous beam
        - low frequency disturbance (displacement currents neglected)
        - emittance dominated beam
        - transversely averaged contribution

        use directSC = 1.0 for further assumptions:
        - same as directSC = 0.5 only transversely maximum contribution
        directly on z-axis

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
    '''Contains transverse space charge for a Gaussian configuration.
    Applies the Bassetti-Erskine electric field expression slice-wise
    for each particle centred around the slice centre.
    '''

    def __init__(self, slicer, length, sig_check=True, other_efieldn=None):
        '''Arguments:
        - slicer determines the slicing parameters for the slices over
        which the Bassetti-Erskine electric field expression is applied,
        given a slicer with n_slices == 1, you can apply a
        longitudinally averaged kick over the whole beam.
        - length is an s interval along which the space charge force
        is integrated.
        - sig_check exchanges x and y quantities for sigma_x < sigma_y
        and applies the round beam formula for sigma_x == sigma_y .
        sig_check defaults to True and should not usually be False.
        - other_efieldn can be used to use a different implementation of
        the charge-normalised electric field expression (there are four
        different implementations to choose from in this class:
        _efieldn_mit, _efield_mitmod, _efieldn_koelbig,
        _efieldn_pyecloud; in order of computational time consumption)
        '''
        self.slicer = slicer
        self.length = length
        if other_efieldn is None:
            self._efieldn = self._efieldn_mit
        else:
            self._efieldn = other_efieldn
        if sig_check:
            self._efieldn = self.add_sigma_check(self._efieldn)

    def track(self, beam):
        '''Add the transverse space charge contribution to the beam's
        transverse kicks.
        '''
        slices = beam.get_slices(
            self.slicer, statistics=["mean_x", "mean_y", "sigma_x", "sigma_y"])
        prefactor = (beam.particlenumber_per_mp * beam.charge
                     / beam.p0 * self.length / (beam.beta * c)**2)

        for s_i, Q_sl, mean_x, mean_y, sig_x, sig_y \
                in zip(xrange(slices.n_slices),
                       slices.lambda_bins(smoothen=False),
                       slices.mean_x, slices.mean_y,
                       slices.sigma_x, slices.sigma_y):
            p_id = slices.particle_indices_of_slice(s_i)
            if len(p_id) == 0:
                continue

            en_x, en_y = self.get_efieldn(
                take(beam.x, p_id), take(beam.y, p_id),
                mean_x, mean_y, sig_x, sig_y)

            beam.xp[p_id] += prefactor * (Q_sl * en_x)
            beam.yp[p_id] += prefactor * (Q_sl * en_y)


    def get_efieldn(self, xr, yr, mean_x, mean_y, sig_x, sig_y):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

        Return (E_x / Q, E_y / Q).
        '''
        x = xr - mean_x
        y = yr - mean_y

        # absolute values for convergence reasons of erfc
        en_x, en_y = self._efieldn(np.abs(x), np.abs(y), sig_x, sig_y)
        en_x = np.abs(en_x) * np.sign(x)
        en_y = np.abs(en_y) * np.sign(y)

        return en_x, en_y

    @staticmethod
    def _sig_sqrt(sig_x, sig_y):
        return np.sqrt(2 * (sig_x**2 - sig_y**2))

    @staticmethod
    def _efieldn_mit(x, y, sig_x, sig_y):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

        Return (E_x / Q, E_y / Q).

        Assumes sig_x > sig_y and mean_x == 0 as well as mean_y == 0.
        For convergence reasons of the erfc, use only x > 0 and y > 0.

        Uses FADDEEVA C++ implementation from MIT (via SciPy >= 0.13.0).
        '''
        # timing was ~0.522 ms for:
        # x = np.arange(-1e-5, 1e-5, 1e-8)
        # y = np.empty(len(x))
        # sig_x = 1.2e-6
        # sig_y = 1e-6
        sig_sqrt = TransverseGaussianSpaceCharge._sig_sqrt(sig_x, sig_y)
        w1 = wofz((x + 1j * y) / sig_sqrt)
        ex = exp(-x**2 / (2 * sig_x**2) +
                 -y**2 / (2 * sig_y**2))
        w2 = wofz(x * sig_y/(sig_x*sig_sqrt) +
                    y * sig_x/(sig_y*sig_sqrt) * 1j)
        val = (w1 - ex * w2) / (2 * epsilon_0 * np.sqrt(pi) * sig_sqrt)
        return val.imag, val.real

    @staticmethod
    def _efieldn_mitmod(x, y, sig_x, sig_y):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

        Return (E_x / Q, E_y / Q).

        Assumes sig_x > sig_y and mean_x == 0 as well as mean_y == 0.
        For convergence reasons of the erfc, use only x > 0 and y > 0.

        Uses erfc C++ implementation from MIT (via SciPy >= 0.13.0)
        and calculates wofz (FADDEEVA function) explicitely.
        '''
        # timing was ~1.01ms for same situation as _efieldn_mit
        sig_sqrt = TransverseGaussianSpaceCharge._sig_sqrt(sig_x, sig_y)
        w1 = errfadd((x + 1j * y) / sig_sqrt)
        ex = exp(-x**2 / (2 * sig_x**2) +
                 -y**2 / (2 * sig_y**2))
        w2 = errfadd(x * sig_y/(sig_x*sig_sqrt) +
                    y * sig_x/(sig_y*sig_sqrt) * 1j)
        val = (w1 - ex * w2) / (2 * epsilon_0 * np.sqrt(pi) * sig_sqrt)
        return val.imag, val.real

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
        # timing was ~3.35ms for same situation as _efieldn_mit
        if not errf:
            raise ImportError('errfff cannot be imported for using ' +
                              'TransverseSpaceCharge._efield_koelbig .' +
                              'Did you f2py errfff.f?')
        sig_sqrt = TransverseGaussianSpaceCharge._sig_sqrt(sig_x, sig_y)
        w1re, w1im = errf(x/sig_sqrt, y/sig_sqrt)
        ex = exp(-x**2 / (2 * sig_x**2) +
                 -y**2 / (2 * sig_y**2))
        w2re, w2im = errf(x * sig_y/(sig_x*sig_sqrt),
                          y * sig_x/(sig_y*sig_sqrt))
        pref = 1. / (2 * epsilon_0 * np.sqrt(pi) * sig_sqrt)
        return pref * (w1im - ex * w2im), pref * (w1re - ex * w2re)

    @staticmethod
    def wfun(z):
        '''FADDEEVA function as implemented in PyECLOUD, vectorised.'''
        x=z.real
        y=z.imag
        if not errf:
            raise ImportError('errfff cannot be imported for using ' +
                              'TransverseSpaceCharge._efield_koelbig .' +
                              'Did you f2py errfff.f?')
        wx,wy=errf(x,y) # was only errf_f
        return wx+1j*wy

    @staticmethod
    def _efieldn_pyecloud(xin, yin, sigmax, sigmay):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

        Return (E_x / Q, E_y / Q).

        Effective copy of PyECLOUD.BassErsk.BassErsk implementation.
        '''
        # timing was ~3.52ms for same situation as _efieldn_mit
        wfun = TransverseGaussianSpaceCharge.wfun
        x=abs(xin);
        y=abs(yin);
        eps0=8.854187817620e-12;
        if sigmax>sigmay:
            S=np.sqrt(2*(sigmax*sigmax-sigmay*sigmay));
            factBE=1/(2*eps0*np.sqrt(pi)*S);
            etaBE=sigmay/sigmax*x+1j*sigmax/sigmay*y;
            zetaBE=x+1j*y;
            val=factBE*(wfun(zetaBE/S)-
                        np.exp( -x*x/(2*sigmax*sigmax)-y*y/(2*sigmay*sigmay))*
                        wfun(etaBE/S) );
            Ex=abs(val.imag)*np.sign(xin);
            Ey=abs(val.real)*np.sign(yin);
        else:
            S=np.sqrt(2*(sigmay*sigmay-sigmax*sigmax));
            factBE=1/(2*eps0*np.sqrt(pi)*S);
            etaBE=sigmax/sigmay*y+1j*sigmay/sigmax*x;
            yetaBE=y+1j*x;
            val=factBE*(wfun(yetaBE/S)-
                        np.exp( -y*y/(2*sigmay*sigmay)-x*x/(2*sigmax*sigmax))*
                        wfun(etaBE/S) );
            Ey=abs(val.imag)*np.sign(yin);
            Ex=abs(val.real)*np.sign(xin);
        return Ex, Ey

    @staticmethod
    def _efieldn_round(x, y, sig_r):
        '''Return (E_x / Q, E_y / Q) for a round distribution
        with sigma_x == sigma_y == sig_r .
        '''
        r2 = x**2 + y**2
        amplitude = (1 - exp(-r2/(2*sig_r**2))) / (2*pi*epsilon_0 * r2)
        return x * amplitude, y * amplitude

    @staticmethod
    def add_sigma_check(efieldn):
        '''Exchanges x and y quantities if sigma_x < sigma_y.
        Applies round beam field instead when sigma_x == sigma_y.
        '''
        efieldn_round = TransverseGaussianSpaceCharge._efieldn_round
        @wraps(efieldn)
        def efieldn_checked(x, y, sig_x, sig_y, *args, **kwargs):
            if sig_x == sig_y:
                if sig_x == 0:
                    en_x, en_y = 0, 0
                else:
                    en_x, en_y = efieldn_round(x, y, sig_x)
            elif sig_x < sig_y:
                en_y, en_x = efieldn(y, x, sig_y, sig_x)
            else:
                en_x, en_y = efieldn(x, y, sig_x, sig_y)
            return en_x, en_y
        return efieldn_checked
