'''
@file matching
@author Kevin Li, Michael Schenk, Adrian Oeftiger
@date 17.10.2014
@brief Module for matching transverse and longitudinal distributions
@copyright CERN
'''
from __future__ import division

from abc import ABCMeta, abstractmethod
import sys

import numpy as np
from numpy.random import normal, uniform, RandomState

from scipy.constants import c, e
from scipy.optimize import brentq, brenth, bisect, newton
from scipy.interpolate import interp2d
from scipy.integrate import quad, fixed_quad, dblquad, cumtrapz, romb

from particles import Particles
from ..trackers.rf_bucket import RFBucket
from . import Printing


class ParticleGenerator(Printing):
    '''Factory to provide Particle instances according to certain
    distributions (which are implemented in inheriting classes via
    the self.distribute() function).
    The Particle instance is obtained via ParticleGenerator.generate() .
    '''
    __metaclass__ = ABCMeta

    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, coords_n_momenta,
                 *args, **kwargs):
        '''coords_n_momenta is a list with the name strings of the
        coordinates and conjugate momenta.
        '''
        self.macroparticlenumber = macroparticlenumber
        self.particlenumber_per_mp = intensity / macroparticlenumber
        self.charge = charge
        self.mass = mass
        self.circumference = circumference
        self.gamma_reference = gamma_reference
        self.coords_n_momenta = coords_n_momenta

    @abstractmethod
    def distribute(self):
        '''Implement the specific distribution of this generator.
        The self.coords_n_momenta may be used for the attribute names.
        Return a coords_n_momenta_dict to instantiate Particles.
        '''
        pass

    def generate(self):
        '''Generate the Particles instance (factory method).'''
        coords_n_momenta_dict = self.distribute()
        return Particles(macroparticlenumber=self.macroparticlenumber,
                         particlenumber_per_mp=self.particlenumber_per_mp,
                         charge=self.charge, mass=self.mass,
                         circumference=self.circumference,
                         gamma_reference=self.gamma_reference,
                         coords_n_momenta_dict=coords_n_momenta_dict)

    def update(self, beam):
        '''Update the given Particles instance with the coordinate and
        momentum distribution configured in this ParticleGenerator.
        Attention: overwrites existing coordinate / momentum attributes
        in the Particles instance.
        '''
        coords_n_momenta_dict = self.distribute()
        beam.update(coords_n_momenta_dict)


class ImportDistribution(ParticleGenerator):
    '''Create a Particles instance from given distributed arrays.'''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, coords_n_momenta_dict,
                 *args, **kwargs):
        '''Directly uses the given dictionary coords_n_momenta_dict
        with the coordinate and conjugate momentum names as keys
        and their corresponding arrays as values.
        '''
        self.coords_n_momenta_dict = coords_n_momenta_dict
        super(ImportDistribution, self).__init__(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, coords_n_momenta_dict.keys(),
            *args, **kwargs)

    def distribute(self):
        return self.coords_n_momenta_dict


class HEADTAILcoords(object):
    '''The classic HEADTAIL phase space.'''
    coordinates = ('x', 'xp', 'y', 'yp', 'z', 'dp')
    transverse = coordinates[:4]
    longitudinal = coordinates[-2:]


class Uniform3D(ParticleGenerator):
    '''Uniform grid along 3D cuboid in classic HEADTAIL phase space
    with coordinates (x, xp, y, yp, z, dp).
    All conjugate momenta (xp, yp, dp) are zero.
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, xextent, yextent, zextent,
                 *args, **kwargs):
        '''Take the extents in all three spatial dimensions.'''
        super(Uniform3D, self).__init__(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, HEADTAILcoords.coordinates,
            *args, **kwargs)
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

    def distribute(self):
        x  = uniform(-self.xextent, self.xextent, self.macroparticlenumber)
        y  = uniform(-self.yextent, self.yextent, self.macroparticlenumber)
        z  = uniform(-self.zextent, self.zextent, self.macroparticlenumber)
        xp = np.zeros(self.macroparticlenumber)
        yp = np.zeros(self.macroparticlenumber)
        dp = np.zeros(self.macroparticlenumber)
        return {'x': x, 'xp': xp, 'y': y, 'yp': yp, 'z': z, 'dp': dp}


class Gaussian(ParticleGenerator):
    '''Provide a Particle instance with Gaussian distributions for
    all coordinates and conjugate momenta.
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, coords_n_momenta_with_sigmas,
                 *args, **kwargs):
        '''coords_n_momenta_with_sigmas is a dict: the keys are the
        name strings of the coordinates and conjugate momenta,
        the values are the respective sigmas.
        e.g.: coords_n_momenta_with_sigmas = {'x': 1e-6, 'xp': 1.2e-6}
        '''
        # FUTURE IDEA?
        # COORDINATES AND CONJUGATE MOMENTA ARE GROUPED TOGETHER IN TUPLES!
        # '''coords_n_momenta_with_sigmas is a dict with tuple keys
        # (with the name strings of the coordinate and conjugate momentum)
        # and tuple values (with the sigmas of the coordinate and the
        # conjugate momentum).
        # e.g.: coords_n_momenta_with_sigmas = {('x', 'xp'): (1e-6, 1e-6)}
        # '''
        # if any(len(key) is not 2 for key in
        #        coords_n_momenta_with_sigmas.key()):
        #     raise ValueError("the dictionary coords_n_momenta_with_sigmas" +
        #                      " takes tuple keys with two entries for the" +
        #                      " coordinate and the respective momentum.")
        # if any(len(value) is not 2 for value in
        #        coords_n_momenta_with_sigmas.values()):
        #     raise ValueError("the dictionary coords_n_momenta_with_sigmas" +
        #                      " takes tuple values with two entries for the" +
        #                      " two sigmas of the coordinate and the" +
        #                      " respective momentum.")

        coords_n_momenta = coords_n_momenta_with_sigmas.keys()
        # # flatten:
        # coords_n_momenta = list(itertools.chain.from_iterable(coords_n_momenta))
        super(Gaussian, self).__init__(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, coords_n_momenta, *args, **kwargs)
        self.coords_n_momenta_with_sigmas = coords_n_momenta_with_sigmas

    def distribute(self):
        '''Create the Gaussian distribution for all coordinates and
        conjugate momenta saved in self.coords_n_momenta_with_sigmas.
        '''
        coords = self.coords_n_momenta
        sigs = self.coords_n_momenta_with_sigmas
        n = self.macroparticlenumber
        return {coord: normal(0, sigs[coord], n) for coord in coords}


class Gaussian6D(Gaussian):
    '''The classic HEADTAIL phase space generator with coordinates
    (x, xp, y, yp, z, dp).
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, sigma_x, sigma_xp,
                 sigma_y, sigma_yp, sigma_z, sigma_dp, *args, **kwargs):
        '''Take the sigmas of all directions to generate the Gaussian
        phase space distribution.
        '''
        coords_n_momenta_with_sigmas = {
            'x': sigma_x, 'xp': sigma_xp,
            'y': sigma_y, 'yp': sigma_yp,
            'z': sigma_z, 'dp': sigma_dp}
        super(Gaussian6D, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, coords_n_momenta_with_sigmas, *args, **kwargs)


class Gaussian2DTwiss(Gaussian):
    '''Generates a coordinate and conjugate momentum pair according
    to the TWISS '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, coords,
                 alpha, beta, epsn_geo, *args, **kwargs):
        '''coords: indicate the name tuple with the strings of the
        coordinate and conjugate momentum name in this order,
        e.g. coords=('x', 'xp').
        Uses the geometric emittance (in the transverse plane it would
        be epsn_geo_{x,y} = epsn_{x,y} / betagamma while in the
        longitudinal plane epsn_geo_z = epsn_z * e / p0) to calculate
        the correspondingly matched coordinate and its conjugate
        momentum.
        '''
        if alpha != 0:
            raise NotImplementedError("alpha != 0 is not yet taken into" +
                                      " account")
        coordssig = self.coords_n_momenta_with_sigmas(coords, epsn_geo, beta)
        super(Gaussian2Dtwiss, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, coordssig, *args, **kwargs)

    @staticmethod
    def coords_n_momenta_with_sigmas(coords, epsn_geo, beta):
        return {
            coords[0]: np.sqrt(epsn_geo * beta), # sigma_u generalised coord
            coords[1]: np.sqrt(epsn_geo / beta)  # sigma_u' conjugate momentum
            }


class Gaussian6DTwiss(Gaussian):
    '''The classic HEADTAIL phase space generator with coordinates
    (x, xp, y, yp, z, dp) using the optics resp. TWISS parameters.
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, alpha_x, beta_x, epsn_x,
                 alpha_y, beta_y, epsn_y, beta_z, epsn_z, *args, **kwargs):
        '''Take the TWISS parameters at the point of injection along
        with the emittances to generate the Gaussian phase space
        distribution.
        '''
        betagamma = np.sqrt(gamma_reference**2 - 1)
        p0 = betagamma * mass * c
        epsn_geo_x = epsn_x / betagamma
        epsn_geo_y = epsn_y / betagamma
        epsn_geo_z = epsn_z * e / (4 * np.pi * p0)

        # super(Gaussian6Dtwiss, self).__init__(
        #     macroparticlenumber, intensity, charge, mass,
        #     circumference, gamma_reference,
        #     epsn_geo_x * beta_x, epsn_geo_x / beta_x,
        #     epsn_geo_y * beta_y, epsn_geo_y / beta_y,
        #     epsn_geo_z * beta_z, epsn_geo_z / beta_z)

        # the following is equivalent but consistently uses Gaussian2DTwiss:
        get_sigmas = Gaussian2DTwiss.coords_n_momenta_with_sigmas
        coordssig = {}
        coordssig.update(get_sigmas(('x', 'xp'), epsn_geo_x, beta_x))
        coordssig.update(get_sigmas(('y', 'yp'), epsn_geo_y, beta_y))
        coordssig.update(get_sigmas(('z', 'dp'), epsn_geo_z, beta_z))

        super(Gaussian6DTwiss, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, coordssig, *args, **kwargs)


class MatchTransverseMap(Gaussian):
    '''Transverse phase space (x, xp, y, yp) is generated with the
    optics resp. TWISS parameters taken from a TransverseMap instance.
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, transverse_map,
                 epsn_x, epsn_y, *args, **kwargs):
        '''Uses the transverse_map to extract the optics parameters.'''
        betagamma = np.sqrt(gamma_reference**2 - 1)
        epsn_geo_x = epsn_x / betagamma
        epsn_geo_y = epsn_y / betagamma
        alpha_x, beta_x, alpha_y, beta_y = transverse_map.get_injection_optics()

        if alpha_x != 0 or alpha_y != 0:
            raise NotImplementedError("alpha != 0 is not yet taken into" +
                                      " account")

        get_sigmas = Gaussian2DTwiss.coords_n_momenta_with_sigmas
        coordssig = {}
        coordssig.update(get_sigmas(('x', 'xp'), epsn_geo_x, beta_x))
        coordssig.update(get_sigmas(('y', 'yp'), epsn_geo_y, beta_y))
        super(MatchTransverseMap, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, coordssig, *args, **kwargs)


class MatchLinearLongMap(Gaussian):
    '''Longitudinal phase space (z, dp) is generated with the epsn_z
    or sigma_z, Qs (synchroton tune) and eta (slippage factor) taken
    from a LongitudinalMap instance.
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, longitudinal_map,
                 epsn_z=None, sigma_z=None, *args, **kwargs):
        '''Uses the longitudinal_map to extract the Qs and eta for the
        beta_z.
        '''
        self.check_long_input(epsn_z, sigma_z)
        p0 = np.sqrt(gamma_reference**2 - 1) * mass * c
        eta = longitudinal_map.eta(0, gamma_reference)
        try:
            Qs = longitudinal_map.Qs
        except AttributeError as exc:
            raise ValueError('"' + self.__name__ + '" expects a ' +
                             'longitudinal_map with a Qs attribute ' +
                             'yielding the linear synchroton frequency. ' +
                             'However, the given "' + exc.message
                             )
        beta_z = np.abs(eta) * circumference / (2 * np.pi * Qs)
        if sigma_z is None:
            sigma_z = np.sqrt(epsn_z * beta_z / (4*np.pi) * e/p0)
        sigma_dp = sigma_z / beta_z
        super(MatchLinearLongMap, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, {'z': sigma_z, 'dp': sigma_dp}, *args, **kwargs
            )

    @staticmethod
    def check_long_input(epsn_z, sigma_z):
        '''Check that exactly one of epsn_z and sigma_z is given,
        the other one should be None.
        '''
        if (epsn_z is None and sigma_z is None) or (epsn_z and sigma_z):
            raise ValueError('***ERROR: exactly one of sigma_z and ' +
                             ' epsn_z is required!')


class MatchGaussian6D(ParticleGenerator):
    '''The classic HEADTAIL phase space generator with coordinates
    (x, xp, y, yp, z, dp) using the given epsn_x, epsn_y and either
    epsn_z or sigma_z, as well as the optics resp. TWISS parameters
    taken from a TransverseMap instance and a LongitudinalMap instance.
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference,
                 transverse_map, epsn_x, epsn_y,
                 longitudinal_map, epsn_z=None, sigma_z=None,
                 *args, **kwargs):
        '''Uses the transverse_map to extract the optics parameters
        and the longitudinal_map to extract the Qs and eta for the
        beta_z.
        '''
        self._transverse_matcher = MatchTransverseMap(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, transverse_map,
            epsn_x, epsn_y, *args, **kwargs)
        self._longitudinal_matcher = MatchLinearLongMap(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, longitudinal_map,
            epsn_z, sigma_z, *args, **kwargs)
        super(MatchGaussian6D, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, HEADTAILcoords.coordinates, *args, **kwargs)

    def distribute(self):
        '''Create the Gaussian distribution for all 6D HEADTAIL
        coordinates and conjugate momenta in both the transverse and
        longitudinal plane.
        '''
        coords_n_momenta_dict = self._transverse_matcher.distribute()
        coords_n_momenta_dict.update(self._longitudinal_matcher.distribute())
        return coords_n_momenta_dict


class MatchRFBucket6D(ParticleGenerator):
    '''The classic HEADTAIL phase space generator with coordinates
    (x, xp, y, yp, z, dp) using the given epsn_x, epsn_y and either
    epsn_z or sigma_z, as well as the optics resp. TWISS parameters
    taken from a TransverseMap instance and the Hamiltonian from an
    RFBucket instance.
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference,
                 transverse_map, epsn_x, epsn_y,
                 rf_bucket, epsn_z=None, sigma_z=None,
                 *args, **kwargs):
        '''Uses the transverse_map to extract the optics parameters
        and the rf_bucket to match the longitudinal distribution.
        '''
        self._transverse_matcher = MatchTransverseMap(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, transverse_map,
            epsn_x, epsn_y, *args, **kwargs)
        self._rf_bucket_matcher = MatchRFBucket2D(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, rf_bucket,
            epsn_z, sigma_z, *args, **kwargs)
        super(MatchRFBucket6D, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, HEADTAILcoords.coordinates, *args, **kwargs)

    def distribute(self):
        '''Create the Gaussian distribution for all 6D HEADTAIL
        coordinates and conjugate momenta in both the transverse and
        longitudinal plane.
        '''
        coords_n_momenta_dict = self._transverse_matcher.distribute()
        coords_n_momenta_dict.update(self._rf_bucket_matcher.distribute())
        return coords_n_momenta_dict


class CutRFBucket6D(ParticleGenerator):
    '''The classic HEADTAIL phase space generator with coordinates
    (x, xp, y, yp, z, dp) using the given epsn_x, epsn_y as well as
    the optics resp. TWISS parameters taken from a TransverseMap
    instance. The longitudinal phase space is initialised as a
    bi-gaussian with given sigma_z and sigma_dp which is then cut along
    the function given by is_accepted. The usual choices for
    is_accepted are either RFBucket.is_in_separatrix or
    RFBucket.is_accepted . The former strictly follows the separatrix
    of the bucket (the equihamiltonian with a value of 0), while with
    the latter, tighter boundaries can be used (equihamiltonian lying
    inside the RFBucket) as a result of which particles are not
    initialised too close to the separatrix. This option is usually
    preferred as it avoids bucket leakage and particle losses which may
    occur as a consequence of the unmatched initialisation. The
    RFBucket.is_accepted method must be created first however, by
    calling the RFBucket.make_is_accepted(margin) method with a certain
    value for the margin (in % of RFBucket.Hmax, 5% by default).
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference,
                 transverse_map, epsn_x, epsn_y,
                 sigma_z, sigma_dp, is_accepted,
                 *args, **kwargs):
        '''Uses the transverse_map to extract the optics parameters
        and the rf_bucket to match the longitudinal distribution.
        '''
        self._transverse_matcher = MatchTransverseMap(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference, transverse_map,
            epsn_x, epsn_y, *args, **kwargs)
        self._rf_bucket_matcher = CutRFBucket2D(
            macroparticlenumber, intensity, charge, mass,
            circumference, gamma_reference,
            sigma_z, sigma_dp, is_accepted, *args, **kwargs)
        super(CutRFBucket6D, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, HEADTAILcoords.coordinates, *args, **kwargs)

    def distribute(self):
        '''Create the Gaussian distribution for all 6D HEADTAIL
        coordinates and conjugate momenta in both the transverse and
        longitudinal plane.
        '''
        coords_n_momenta_dict = self._transverse_matcher.distribute()
        coords_n_momenta_dict.update(self._rf_bucket_matcher.distribute())
        return coords_n_momenta_dict


# possible TODO: incorporate RFBucketMatcher class
class MatchRFBucket2D(ParticleGenerator):
    '''Longitudinal phase space (z, dp) is generated with the epsn_z
    or sigma_z and the Hamiltonian from the given RFBucket instance..
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, rf_bucket,
                 epsn_z=None, sigma_z=None, *args, **kwargs):
        '''Uses the RFBucket rf_bucket to match the longitudinal
        distribution.
        '''
        self.rf_bucket = rf_bucket
        self.epsn_z = epsn_z
        self.sigma_z = sigma_z
        MatchLinearLongMap.check_long_input(epsn_z, sigma_z)
        super(MatchRFBucket2D, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, HEADTAILcoords.longitudinal, *args, **kwargs)

    def distribute(self):
        '''Create the longitudinal distribution from an RFBucketMatcher
        and return it.
        '''
        rf_bucket_matcher = RFBucketMatcher(
            StationaryExponential, self.rf_bucket, self.sigma_z, self.epsn_z)
        z, dp, _, _ = rf_bucket_matcher.generate(self.macroparticlenumber)
        return {'z': z, 'dp': dp}


class CutRFBucket2D(ParticleGenerator):
    '''For HEADTAIL style matching into RF bucket.
    The argument is_accepted takes a function (i.e. reference to a
    function). The usual choices are RFBucket.is_in_separatrix or
    RFBucket.is_accepted . The former strictly follows the separatrix
    of the bucket (the equihamiltonian with a value of 0), while with
    the latter, tighter boundaries can be used (equihamiltonian lying
    inside the RFBucket) as a result of which particles are not
    initialised too close to the separatrix. This option is usually
    preferred as it avoids bucket leakage and particle losses which may
    occur as a consequence of the unmatched initialisation.
    The RFBucket.is_accepted method must be created first however, by
    calling the RFBucket.make_is_accepted(margin) method with a certain
    value for the margin (in % of RFBucket.Hmax, 5% by default).

    BY KEVIN: NEEDS TO BE CLEANED UP BY ADRIAN!
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma_reference, sigma_z, sigma_dp,
                 is_accepted, *args, **kwargs):

        self.sigma_z = sigma_z
        self.sigma_dp = sigma_dp
        self.is_accepted = is_accepted

        super(CutRFBucket2D, self).__init__(
            macroparticlenumber, intensity, charge, mass, circumference,
            gamma_reference, HEADTAILcoords.longitudinal, *args, **kwargs)

    def distribute(self):

        z = normal(0, self.sigma_z, self.macroparticlenumber)
        dp = normal(0, self.sigma_dp, self.macroparticlenumber)
        self._redistribute(z, dp)

        return {'z': z, 'dp': dp}

    def _redistribute(self, z, dp):

        mask_out = ~self.is_accepted(z, dp)
        while mask_out.any():
            n_gen = np.sum(mask_out)
            z[mask_out] = normal(0, self.sigma_z, n_gen)
            dp[mask_out] = normal(0, self.sigma_dp, n_gen)
            mask_out = ~self.is_accepted(z, dp)
            self.prints('Reiterate on non-accepted particles')


class RFBucketMatcher(object):
    def __init__(self, psi, rfbucket, sigma_z=None, epsn_z=None):

        self.psi = psi
        self.H = rfbucket
        self.sigma_z = sigma_z

        self.psi_object = psi(rfbucket.hamiltonian, rfbucket.Hmax)
        self.psi = self.psi_object.function
        self.p_limits = rfbucket.separatrix

        self._compute_std = self._compute_std_cumtrapz

        if sigma_z and not epsn_z:
            self.variable = sigma_z
            self.psi_for_variable = self.psi_for_bunchlength_newton_method
        elif not sigma_z and epsn_z:
            self.variable = epsn_z
            self.psi_for_variable = self.psi_for_emittance_newton_method
        else:
            raise ValueError("Can not generate mismatched matched distribution!")

        self.seed = np.random.randint(sys.maxint)

    def psi_for_emittance_newton_method(self, epsn_z):
        H = self.H

        # Maximum emittance
        self._set_psi_sigma(H.circumference)
        # zc_left, zc_right = self._get_edges_for_cut(np.exp(-2**2/2.))
        # epsn_max = self._compute_zero_quad(lambda y, x: 1, H.equihamiltonian(zc_left), H.zleft, H.zright) * 2*H.p0_reference/e
        z, dp = self._regenerate(seed=self.seed)
        epsn_max = self._compute_emittance(z, dp) * 4*np.pi*H.p0_reference/e
        if epsn_z > epsn_max:
            print '\n*** RMS emittance larger than bucket; using full bucket emittance', epsn_max, ' [eV s].'
            epsn_z = epsn_max*0.99
        print '\n*** Maximum RMS emittance', epsn_max, 'eV s.'

        # @profile
        def get_zc_for_epsn_z(ec):
            self._set_psi_epsn(ec)
            # zc_left, zc_right = self._get_edges_for_cut(np.exp(-2**2/2.))
            # emittance = self._compute_zero_quad(lambda y, x: 1, H.equihamiltonian(zc_left), H.zleft, H.zright) * 2*H.p0_reference/e
            z, dp = self._regenerate(seed=self.seed)
            emittance = self._compute_emittance(z, dp) * 4*np.pi*H.p0_reference/e
            print '... distance to target emittance: {:.2e}'.format(emittance-epsn_z)

            return emittance-epsn_z

        try:
            ec_bar = newton(get_zc_for_epsn_z, epsn_z, tol=5e-4, maxiter=30)
        except RuntimeError:
            print '*** WARNING: failed to converge using Newton-Raphson method. Trying classic Brent method...'
            ec_bar = brentq(get_zc_for_epsn_z, epsn_z/2, 2*epsn_max)

        self._set_psi_epsn(ec_bar)
        # zc_left, zc_right = self._get_edges_for_cut(np.exp(-2**2/2.))
        # emittance = self._compute_zero_quad(lambda y, x: 1, H.equihamiltonian(zc_left), H.zleft, H.zright) * 2*H.p0_reference/e
        z, dp = self._regenerate(seed=self.seed)
        emittance = self._compute_emittance(z, dp) * 4*np.pi*H.p0_reference/e
        sigma = self._compute_std(self.psi, H.separatrix, H.zleft, H.zright)

        print '\n--> Emittance:', emittance
        print '--> Bunch length:', sigma
        # H.zleft_for_eps, H.zright_for_eps = zc_left, zc_right
        H.emittance, H.sigma = emittance, sigma

    # @profile
    def psi_for_bunchlength_newton_method(self, sigma):
        H = self.H

        # Maximum bunch length
        self._set_psi_sigma(H.circumference)
        sigma_max = self._compute_std(self.psi, H.separatrix, H.zleft, H.zright)
        if sigma > sigma_max:
            print ('\n*** RMS bunch larger than bucket; using full bucket' +
                   ' rms length', sigma_max, ' m.')
            sigma = sigma_max*0.99
        print '\n*** Maximum RMS bunch length', sigma_max, 'm.'

        def get_zc_for_sigma(zc):
            '''Width for bunch length'''
            self._set_psi_sigma(zc)
            length = self._compute_std(self.psi, H.separatrix,
                                       H.zleft, H.zright)
            if np.isnan(length):
                raise ValueError
            print '... distance to target bunchlength:', length-sigma

            return length-sigma

        zc_bar = newton(get_zc_for_sigma, sigma)

        self._set_psi_sigma(zc_bar)
        # zc_left, zc_right = self._get_edges_for_cut(np.exp(-2**2/2.))
        # emittance = self._compute_zero_quad(lambda y, x: 1, H.equihamiltonian(zc_left), H.zleft, H.zright) * 2*H.p0_reference/e
        z, dp = self._regenerate(seed=self.seed)
        emittance = self._compute_emittance(z, dp) * 4*np.pi*H.p0_reference/e
        sigma = self._compute_std(self.psi, H.separatrix, H.zleft, H.zright)

        print '--> Emittance:', emittance
        print '\n--> Bunch length:', sigma
        # H.zleft_for_eps, H.zright_for_eps = zc_left, zc_right
        H.emittance, H.sigma = emittance, sigma

    def generate(self, macroparticlenumber, particles=None):
        '''
        Generate a 2d phase space of n_particles particles randomly distributed
        according to the particle distribution function psi within the region
        [xmin, xmax, ymin, ymax].
        '''
        self.psi_for_variable(self.variable)
        u, v = self._regenerate(macroparticlenumber, self.seed)

        if particles:
            particles.z = u
            particles.dp = v
            particles.psi = self.psi
            particles.linedensity = self.linedensity

        return u, v, self.psi, self.linedensity

    def linedensity(self, xx):
        quad_type = fixed_quad

        L = []
        try:
            L = np.array([quad_type(lambda y: self.psi(x, y), 0, self.p_limits(x))[0] for x in xx])
        except TypeError:
            L = quad_type(lambda y: self.psi(xx, y), 0, self.p_limits(xx))[0]
        L = np.array(L)

        return 2*L

    def _regenerate(self, macroparticlenumber=50000, seed=None):

        # # Bin
        # i, j = 0, 0
        # nx, ny = 128, 128
        # xx = np.linspace(xmin, xmax, nx + 1)
        # yy = np.linspace(ymin, ymax, ny + 1)
        # XX, YY = np.meshgrid(xx, yy)
        # HH = self.psi(XX, YY)
        # psi_interp = interp2d(xx, yy, HH)

        # while j < particles.n_macroparticles:
        #     u = xmin + lx * np.random.random()
        #     v = ymin + ly * np.random.random()

        #     s = np.random.random()

        #     i += 1
        #     if s < psi_interp(u, v):
        #         x[j] = u
        #         y[j] = v
        #         # TODO: check if this does not cause problems! Setter for item does not work - not implemented!
        #         # particles.dp[j] = v
        #         j += 1

        # ================================================================
        # mask_out = ~self.is_accepted(z, dp)
        # while mask_out.any():
        #     n_gen = np.sum(mask_out)
        #     z[mask_out] = self.sigma_z * self.random_state.randn(n_gen)
        #     dp[mask_out] = self.sigma_dp * self.random_state.randn(n_gen)
        #     mask_out = ~self.is_accepted(z, dp)
        #     print 'Reiterate on non-accepted particles'

        # for i in xrange(n):
        #     while not self.is_accepted(z[i], dp[i]):
        #         z[i]  = self.sigma_z * self.random_state.randn()
        #         dp[i] = self.sigma_dp * self.random_state.randn()
        # ================================================================

        xmin, xmax = self.H.zleft, self.H.zright
        ymin, ymax = -self.H.p_max(self.H.zright), self.H.p_max(self.H.zright)
        lx = (xmax - xmin)
        ly = (ymax - ymin)

        if seed:
            random_state = RandomState()
            random_state.seed(seed)

        n_gen = macroparticlenumber
        u = xmin + lx * random_state.uniform(size=n_gen)
        v = ymin + ly * random_state.uniform(size=n_gen)
        s = random_state.uniform(size=n_gen)
        mask_out = ~(s<self.psi(u, v))
        while mask_out.any():
            n_gen = np.sum(mask_out)
            u[mask_out] = xmin + lx * random_state.uniform(size=n_gen)
            v[mask_out] = ymin + ly * random_state.uniform(size=n_gen)
            s[mask_out] = random_state.uniform(size=n_gen)
            mask_out = ~(s<self.psi(u, v))
            # print 'regenerating '+str(n_gen)+' macroparticles...'

        return u, v

    def _set_psi_sigma(self, sigma):
        self.psi_object.H0 = self.H.H0_from_sigma(sigma)

    def _set_psi_epsn(self, epsn):
        self.psi_object.H0 = self.H.H0_from_epsn(epsn)

    # @profile
    def _get_edges_for_cut(self, h_cut):
        zz = np.linspace(self.H.zmin, self.H.zmax, 128)
        ll = self.linedensity(zz)
        lmax = np.amax(ll)
        # plt.plot(zz, linedensity(zz)/lmax, 'r', lw=2)
        # plt.plot(zz, psi(zz, 0))
        # plt.axhline(h_cut)
        # plt.axvline(zcut_bar)
        # plt.show()
        return self.H._get_zero_crossings(
            lambda x: self.linedensity(x) - h_cut*lmax)

    def _compute_emittance(self, z, dp):
        var_z    = np.var(z)
        var_dp   = np.var(dp)
        mean_zdp = np.mean( (z-np.mean(z)) * (dp-np.mean(dp)) )

        return np.sqrt(var_z*var_dp - mean_zdp**2)

    def _compute_zero_quad(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin
        to xmax along the contours p_sep using numerical integration
        methods.
        '''

        Q, error = dblquad(lambda y, x: psi(x, y), xmin, xmax,
                    lambda x: 0, lambda x: p_sep(x))

        return Q

    def _compute_mean_quad(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin
        to xmax along the contours p_sep using numerical integration
        methods.
        '''

        Q = self._compute_zero_quad(psi, p_sep, xmin, xmax)
        M, error = dblquad(lambda y, x: x * psi(x, y), xmin, xmax,
                    lambda x: 0, lambda x: p_sep(x))

        return M/Q

    def _compute_std_quad(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin
        to xmax along the contours p_sep using numerical integration
        methods.
        '''

        Q = self._compute_zero_quad(psi, p_sep, xmin, xmax)
        M = self._compute_mean_quad(psi, p_sep, xmin, xmax)
        V, error = dblquad(lambda y, x: (x-M) ** 2 * psi(x, y), xmin, xmax,
                           lambda x: 0, lambda x: p_sep(x))

        return np.sqrt(V/Q)

    def _compute_zero_cumtrapz(self, psi, p_sep, xmin, xmax):

        x_arr = np.linspace(xmin, xmax, 257)
        dx = x_arr[1] - x_arr[0]

        Q = 0
        for x in x_arr:
            y = np.linspace(0, p_sep(x), 257)
            z = psi(x, y)
            Q += cumtrapz(z, y)[-1]
        Q *= dx

        return Q

    def _compute_mean_cumtrapz(self, psi, p_sep, xmin, xmax):

        Q = self._compute_zero_cumtrapz(psi, p_sep, xmin, xmax)

        x_arr = np.linspace(xmin, xmax, 257)
        dx = x_arr[1] - x_arr[0]

        M = 0
        for x in x_arr:
            y = np.linspace(0, p_sep(x), 257)
            z = x * psi(x, y)
            M += cumtrapz(z, y)[-1]
        M *= dx

        return M/Q

    def _compute_std_cumtrapz(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin
        to xmax along the contours p_sep using numerical integration
        methods.
        '''

        Q = self._compute_zero_cumtrapz(psi, p_sep, xmin, xmax)
        M = self._compute_mean_cumtrapz(psi, p_sep, xmin, xmax)

        x_arr = np.linspace(xmin, xmax, 257)
        dx = x_arr[1] - x_arr[0]

        V = 0
        for x in x_arr:
            y = np.linspace(0, p_sep(x), 257)
            z = (x-M)**2 * psi(x, y)
            V += cumtrapz(z, y)[-1]
        V *= dx

        return np.sqrt(V/Q)

    def _compute_std_romberg(self, psi, p_sep, xmin, xmax):
        '''
        Compute the variance of the distribution function psi from xmin
        to xmax along the contours p_sep using numerical integration
        methods.
        '''

        x_arr = np.linspace(xmin, xmax, 257)
        dx = x_arr[1] - x_arr[0]

        Q, V = 0, 0
        for x in x_arr:
            y = np.linspace(0, p_sep(x), 257)
            dy = y[1] - y[0]
            z = psi(x, y)
            Q += romb(z, dy)
            z = x**2 * psi(x, y)
            V += romb(z, dy)
        Q *= dx
        V *= dx

        return np.sqrt(V/Q)


class StationaryExponential(object):

    def __init__(self, H, Hmax=None, width=1000, Hcut=0):
        self.H = H
        self.H0 = 1
        if not Hmax:
            self.Hmax = H(0, 0)
        else:
            self.Hmax = Hmax
        self.Hcut = Hcut
        self.width = width

    def function(self, z, dp):
        psi = np.exp(self.H(z, dp).clip(min=0)/self.H0) - 1
        psi_norm = np.exp(self.Hmax/self.H0) - 1
        return psi/psi_norm
