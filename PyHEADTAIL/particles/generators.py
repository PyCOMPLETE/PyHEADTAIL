'''
@author Kevin Li, Michael Schenk, Adrian Oeftiger, Stefan Hegglin
@date 30.03.2015
@brief module for generating & matching particle distributions
'''

from __future__ import division

import numpy as np

from particles import Particles
from rfbucket_matching import RFBucketMatcher, ThermalDistribution

# backwards compatibility:
StationaryExponential = ThermalDistribution

from . import Printing

from scipy.constants import e, c


def generate_Gaussian6DTwiss(macroparticlenumber, intensity, charge, mass,
                             circumference, gamma,
                             alpha_x, alpha_y, beta_x, beta_y, beta_z,
                             epsn_x, epsn_y, epsn_z,
                             dispersion_x=None, dispersion_y=None):
    """ Convenience wrapper which generates a 6D gaussian phase space
    with the specified parameters
    Args:
        the usual suspects
    Returns: A particle instance with the phase space matched to the arguments
    """
    beta = np.sqrt(1.-gamma**(-2))
    p0 = np.sqrt(gamma**2 - 1) * mass * c
    eps_geo_x = epsn_x/(beta*gamma)
    eps_geo_y = epsn_y/(beta*gamma)
    eps_geo_z = epsn_z * e / (4. * np.pi * p0)
    # a bit of a hack: epsn_z is a parameter even though the ParticleGenerator
    # does not have such a parameter. This is kept for backwards compatiblity.
    # Therefore, some fake eta, Qs parameters are invented s.t.
    # beta_z = |eta| * circumference / (2 * pi * Qs)
    # holds (circumference is fixed). Does not have any side effects.
    Qs = 1./(2 * np.pi)
    eta = beta_z / circumference
    return ParticleGenerator(
        macroparticlenumber, intensity, charge, mass, circumference, gamma,
        gaussian2D(eps_geo_x), alpha_x, beta_x, dispersion_x,
        gaussian2D(eps_geo_y), alpha_y, beta_y, dispersion_y,
        gaussian2D(eps_geo_z), Qs, eta
        ).generate()


def transverse_linear_matcher(alpha, beta, dispersion=None):
    ''' Return a transverse matcher with the desired parameters.
    Args:
        alpha: Twiss parameter
        beta: Twiss parameter
        dispersion: (optional) only use in combination with a longitudinal
                    phase space
    Returns: Matcher(closure) taking two parameters: coords and direction
    '''
#    if dispersion and alpha:
#        raise NotImplementedError('Transverse phase space matching: for '
#                                  'alpha != 0 we need to match including the '
#                                  'D\' (dispersion derivative). This is '
#                                  'currently not implemented.')
    sqrt = np.sqrt
    # build the M matrix: only depends on twiss parameters for the
    # special case of alpha0=0, beta0=1 and phi = 0 (=2pi)
    # Wiedemann, Particle Accelerator Physics 3rd edition, p. 170
    M = np.zeros(shape=(2, 2))
    M[0, 0] = sqrt(beta)
    M[0, 1] = 0
    M[1, 0] = -alpha/sqrt(beta)
    M[1, 1] = sqrt(1./beta)

    def _transverse_linear_matcher(beam, direction):
        '''Match the coords specified by direction list
        Args:
            coords: a Particle instance
            direction: list (len>=2) specifying which coordinates to match
                       the first element corresponds to space, the second to
                       the momentum coordinate. e.g. ['x', 'xp']
        Returns:
            Nothing, transforms coords dictionary in place
        '''
        space_coords = getattr(beam, direction[0])
        space_coords_copy = space_coords.copy()
        momentum_coords = getattr(beam, direction[1])
        space_coords =    (M[0, 0]*space_coords + # copy if not using +=, *=..
                           M[0, 1]*momentum_coords)
        momentum_coords = (M[1, 0]*space_coords_copy +
                           M[1, 1]*momentum_coords)
        # add dispersion effects, raise exception if coords['dp'] inexistent
        if dispersion:
            try:
                space_coords += dispersion * getattr(beam, 'dp')
            except KeyError:
                print ('Dispersion in the transverse phase space depends on' +
                       'dp, however no longitudinal phase space was specified. '+
                       'No matching performed')
        setattr(beam, direction[0], space_coords)
        setattr(beam, direction[1], momentum_coords)

    return _transverse_linear_matcher


def longitudinal_linear_matcher(Qs, eta, C):
    '''Return simple longitudinal matcher
    Internally calls the transverse linear matcher with beta=beta_z
    and alpha = 0.
    beta_z = |eta| * C / (2*pi*Qs)t p
    Args:
        Qs: synchroton tune
        eta: slippage factor (zeroth order),
             is \alpha_c - gamma^2 (\alpha_c = momentum compaction factor)
        C: circumference
    Returns:
        A matcher with the specified Qs, eta (closure)
    '''
    beta_z = np.abs(eta) * C / (2. * np.pi * Qs)
    internal_transverse_matcher = transverse_linear_matcher(alpha=0.,
                                                            beta=beta_z)

    def _longitudinal_linear_matcher(beam, *args, **kwargs):
        '''Match the beam to the specified parameters:
        Qs, eta, beam.circumference
        Args:
            beam: provides beam.z, beam.dp, beam.beta, beam.circumference
        Returns:
            nothing, modifies beam in place
        '''
        internal_transverse_matcher(beam, direction=['z', 'dp'])
    return _longitudinal_linear_matcher


def RF_bucket_distribution(rfbucket, sigma_z=None, epsn_z=None,
                           margin=0, distribution_type=ThermalDistribution,
                           *args, **kwargs):
    '''Return a distribution function which generates particles
    which are matched to the specified bucket and target emittance or std
    Specify only one of sigma_z, epsn_z
    Args:
        rfbucket: An object of type RFBucket
        sigma_z: target std
        epsn_z: target normalized emittance in z-direction
        margin: relative margin from the separatrix towards the
            inner stable fix point in which particles are avoided
        distribution_type: longitudinal distribution type from
            rfbucket_matching (default is ThermalDistribution which
            produces a Gaussian-like matched Boltzmann distribution)
    Returns:
        A matcher with the specified bucket properties (closure)
    Raises:
        ValueError: If neither or both of sigma_z, epsn_z are specified
    '''
    rf_bucket_matcher_impl = RFBucketMatcher(rfbucket, distribution_type,
                                             sigma_z=sigma_z, epsn_z=epsn_z,
                                             *args, **kwargs)

    def _RF_bucket_dist(n_particles):
        z, dp, _, _ = rf_bucket_matcher_impl.generate(n_particles, margin)
        return [z, dp]
    return _RF_bucket_dist


def cut_distribution(distribution, is_accepted):
    """Generate coordinates according to some distribution inside the
    region specified by where the function is_accepted returns 1.
    (Wrapper for distributions, based on RF_cut..)
    Args:
        distribution: a function which takes the n_particles as a
                      parameter and returns a list-like object
                      containing a 2D phase space. result[0] should
                      stand for the spatial, result[1] for the momentum
                      coordinate
        is_accepted: function taking two parameters (z, dp)
                     [vectorised as arrays] and returning a boolean
                     specifying whether the coordinate lies
                     inside the desired phase space volume. A possible
                     source to provide such an is_accepted function
                     is the RFBucket.make_is_accepted or
                     generators.make_is_accepted_within_n_sigma .
    Returns:
        A matcher with the specified bucket properties (closure)
    """
    def _cut_distribution(n_particles):
        '''Regenerates all particles which fall outside a previously
        specified phase space region (via the function is_accepted
        in generators.cut_distribution) until all generated particles
        have valid coordinates and momenta.
        '''
        z = np.zeros(n_particles)
        dp = np.zeros(n_particles)
        new_coords = distribution(n_particles)
        z = new_coords[0]
        dp = new_coords[1]
        mask_out = ~is_accepted(z, dp)
        while mask_out.any():
            n_gen = np.sum(mask_out)
            new_coords = distribution(n_gen)
            z[mask_out] = new_coords[0]
            dp[mask_out] = new_coords[1]
            mask_out = ~is_accepted(z, dp)
        return [z, dp]
    return _cut_distribution

def make_is_accepted_within_n_sigma(epsn_rms, limit_n_rms, twiss_beta=1):
    '''Closure creating an is_accepted function (e.g. for
    cut_distribution). The is_accepted function will return whether
    the canonical coordinate and momentum pair lies within the phase
    space region limited by the action value limit_n_rms * epsn_rms.

    Coordinate u and momentum up are assumed to be connected to the
    amplitude J via the twiss_beta value,
    J = sqrt(u^2 + twiss_beta^2 up^2) .
    The amplitude is required to be below the limit to be accepted,
    J < limit_n_rms * epsn_rms.
    The usual use case will be generating u and up in normalised Floquet
    space (i.e. before the normalised phase space coordinates
    get matched to the optics or longitudinal eta and Qs).
    In this case, twiss_beta takes the default value 1 in normalised
    Floquet space. Consequently, the 1 sigma RMS reference value
    epsn_rms corresponds to the normalised 1 sigma RMS emittance
    (i.e. amounting to beam.epsn_x() and beam.epsn_y() in the transverse
    plane, and beam.epsn_z()/4 in the longitudinal plane).
    '''
    threshold_amplitude_squared = (limit_n_rms * epsn_rms)**2
    def is_accepted(u, up):
        Jsq = u**2 + (twiss_beta * up)**2
        return Jsq < threshold_amplitude_squared
    return is_accepted


class ParticleGenerator(Printing):
    '''Factory to generate Particle instances according to distributions
    specified by the parameters in the initializer.
    The Particle instance can be generated via the .generate() method
    '''
    def __init__(self, macroparticlenumber, intensity, charge, mass,
                 circumference, gamma,
                 distribution_x=None, alpha_x=0., beta_x=1., D_x=None,
                 distribution_y=None, alpha_y=0., beta_y=1., D_y=None,
                 distribution_z=None, Qs=None, eta=None,
                 *args, **kwargs):
        '''
        Specify the distribution for each phase space seperately. Only
        the phase spaces for which a distribution has been specified
        will be generated.
        The transverse phase space can be matched by specifying the Twiss
        parameters alpha and/or beta. The dispersion will be take into
        account after the beam has been matched longitudinally (if matched).
        The longitudinal phase space will only get matched
        if both Qs and eta are specified.
        Args:
            distribution_[x,y,z]: a function which takes the n_particles
                as a parameter and returns a list-like object containing
                a 2D phase space. result[0] should stand for the spatial,
                result[1] for the momentum coordinate
            alpha_[x,y]: Twiss parameter. The corresponding transverse phase
                space gets matched to (alpha_[], beta_[])
            beta_[x,y]: Twiss parameter. The corresponding transverse phase
                space gets matched to (alpha_[], beta_[])
            D_[x,y]: Dispersion. Only valid in combination with a longitudinal
                phase space.
            Qs: Synchrotron tune. If Qs and eta are specified the
                longitudinal phase space gets matched to these parameters.
            eta: Slippage factor (zeroth order).If Qs and eta are specified
                the longitudinal phase space gets matched to these parameters.
        '''
        self.macroparticlenumber = macroparticlenumber
        self.intensity = intensity
        self.charge = charge
        self.mass = mass
        self.circumference = circumference
        self.gamma = gamma
        # bind the generator methods and parameters for the matching
        self.distribution_x = distribution_x
        self.distribution_y = distribution_y
        self.distribution_z = distribution_z

        # bind the matching methods with the correct parameters
        if Qs is not None and eta is not None:  # match longitudinally iff
            self.linear_matcher_z = longitudinal_linear_matcher(
                Qs, eta, circumference)
        else:
            self.linear_matcher_z = None
        self.linear_matcher_x = transverse_linear_matcher(alpha_x, beta_x, D_x)
        self.linear_matcher_y = transverse_linear_matcher(alpha_y, beta_y, D_y)

    def generate(self):
        ''' Returns a particle  object with the parameters specified
        in the constructor of the Generator object
        '''
        coords = self._create_phase_space()
        particles = Particles(self.macroparticlenumber,
                              self.intensity/self.macroparticlenumber,
                              self.charge, self.mass, self.circumference,
                              self.gamma,
                              coords_n_momenta_dict=coords)
        self._linear_match_phase_space(particles)
        return particles

    def update(self, beam):
        '''Updates the beam coordinates specified in the constructor of the
        Generator object. Existing coordinates will be overriden, new ones
        will be added. Calls beam.update()
        '''
        coords = self._create_phase_space()
        beam.update(coords)
        self._linear_match_phase_space(beam)

    def _create_phase_space(self):
        coords = {}
        if self.distribution_x is not None:
            x_phase_space = self.distribution_x(self.macroparticlenumber)
            coords.update({'x': np.ascontiguousarray(x_phase_space[0]),
                           'xp': np.ascontiguousarray(x_phase_space[1])})
            assert len(coords['x']) == len(coords['xp'])
        if self.distribution_y is not None:
            y_phase_space = self.distribution_y(self.macroparticlenumber)
            coords.update({'y': np.ascontiguousarray(y_phase_space[0]),
                           'yp': np.ascontiguousarray(y_phase_space[1])})
            assert len(coords['y']) == len(coords['yp'])
        if self.distribution_z is not None:
            z_phase_space = self.distribution_z(self.macroparticlenumber)
            coords.update({'z': np.ascontiguousarray(z_phase_space[0]),
                           'dp': np.ascontiguousarray(z_phase_space[1])})
            assert len(coords['z']) == len(coords['dp'])

        return coords

    def _linear_match_phase_space(self, beam):
        # NOTE: keep this ordering (z as first, as x,y dispersion effects
        # depend on the dp coordinate!
        if self.linear_matcher_z is not None:
            self.linear_matcher_z(beam, ['z', 'dp'])
        if self.distribution_x is not None:
            self.linear_matcher_x(beam, ['x', 'xp'])
        if self.distribution_y is not None:
            self.linear_matcher_y(beam, ['y', 'yp'])


def import_distribution2D(coords):
    '''Return a closure which generates the phase space specified
    by the coords list
    Args:
        coords: list containing the coordinates to use
            coords[0] is the space, coords[1] the momentum coordinate
    '''
    assert len(coords[0]) == len(coords[1])

    def _import_distribution2D(n_particles):
        '''Return the specified coordinates
        Args:
            n_particles: number of particles, must be equal len(coords[0/1])
        '''
        assert len(coords[0]) == n_particles
        return coords

    return _import_distribution2D


def gaussian2D(emittance_geo):
    '''Closure which generates a gaussian distribution with the given
    geometrical emittance. Uncorrelated and symmetrical.
    Args:
        -emittance_geo: geometrical emittance (normalized emittance/betagamma
                        for transverse, emittance*e/(4*pi*p0) for longitudinal)
    Returns:
        A function generating a 2d gaussian with the desired parameters
    '''

    def _gaussian2D(n_particles):
        std = np.sqrt(emittance_geo)  # bc. std = sqrt(emittance_geo)
        coords = [np.random.normal(loc=0., scale=std, size=n_particles),
                  np.random.normal(loc=0., scale=std, size=n_particles)]
        return coords
    return _gaussian2D


def gaussian2D_asymmetrical(sigma_u, sigma_up):
    ''' Closure which generates a gaussian distribution with the given
    standard deviations. No correlation between u and up
    Args:
        - sigma_u: standard deviation of the marginal spatial distribution
        - sigma_up: standard deviation of the marginal momentum distribution
    Returns:
        A function generating a 2d gaussian with the desired parameters
    '''
    def _gaussian2D(n_particles):
        coords = [np.random.normal(loc=0., scale=sigma_u, size=n_particles),
                  np.random.normal(loc=0., scale=sigma_up, size=n_particles)]
        return coords
    return _gaussian2D


def uniform2D(low, high):
    '''Closure which generates a uniform distribution for the space coords.
    All momenta are 0.
    '''
    def _uniform2D(n_particles):
        '''Create a 2xn_particles ndarray. The first row will be uniformly
        distributed random numbers between high,low. The second row will
        be zero (equals zero momentum)
        '''
        coords = np.zeros(shape=(2, n_particles))
        coords[0] = np.random.uniform(low=low, high=high, size=n_particles)
        return coords
    return _uniform2D


'''
Why we have this ll algo in here? We had a better one (Knuth):
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.8671&rep=rep1&type=pdf


import numpy as np


def transform_into_kv(bunch, a_x, a_xp, a_y, a_yp):

    # KV distribution
    # ==================================================
    d = 4
    for i in range(bunch.macroparticlenumber):
       u = np.random.normal(size=d)
       r = np.sqrt(np.sum(u**2))
       u *= 1./r

       bunch.x[i]  = u[0]
       bunch.xp[i] = u[1]
       bunch.y[i]  = u[2]
       bunch.yp[i] = u[3]
    # ==================================================

    bunch.x  *= a_x
    bunch.xp *= a_xp
    bunch.y  *= a_y
    bunch.yp *= a_yp


def transform_into_waterbag(bunch, a_x, a_xp, a_y, a_yp):

    # waterbag distribution
    # ==================================================
    d = 4
    for i in range(bunch.macroparticlenumber):
        u = np.random.normal(size=d)
        r = np.sqrt(np.sum(u**2))
        u *= (np.random.rand(1))**(1./d)/r

        bunch.x[i]  = u[0]
        bunch.xp[i] = u[1]
        bunch.y[i]  = u[2]
        bunch.yp[i] = u[3]
    # ==================================================

    bunch.x  *= a_x
    bunch.xp *= a_xp
    bunch.y  *= a_y
    bunch.yp *= a_yp


Want to get this back in the near future...
'''


def kv2D(r_u, r_up):
    '''Closure which generates a Kapchinski-Vladimirski-type uniform
    distribution in 2D. The extent is determined by the arguments.

    Args:
        - r_u: envelope edge radius for the spatial axis
        - r_up: envelope edge angle for the momentum axis
    '''
    def _kv2d(n_particles):
        '''Create a two-dimensional phase space (u, up)
        Kapchinski-Vladimirski-type uniform distribution.
        '''
        rand = np.random.uniform(low=-0.5, high=0.5, size=n_particles)
        u = np.sin(2 * np.pi * rand)
        r = np.where(u > 1, 2 - u, u)
        sign = (-1)**np.random.randint(2, size=n_particles)
        up = sign * np.sqrt(1. - r**2)
        return [u, up]
    return _kv2d


def kv4D(r_x, r_xp, r_y, r_yp):
    '''Closure which generates a Kapchinski-Vladimirski-type uniform
    distribution in 4D. The extent of the phase space ellipses is
    determined by the arguments.

    Args:
        - r_x: envelope edge radius for the horizontal spatial axis
        - r_xp: envelope edge angle for the horizontal momentum axis
        - r_y: envelope edge radius for the vertical spatial axis
        - r_yp: envelope edge angle for the vertical momentum axis
    '''
    def _kv4d(n_particles):
        '''Create a four-dimensional phase space (x, xp, y, yp)
        Kapchinski-Vladimirski-type uniform distribution.
        '''
        t = 2 * np.pi * np.random.uniform(low=-0.5, high=0.5, size=n_particles)
        u = (np.random.uniform(low=0, high=1, size=n_particles) +
             np.random.uniform(low=0, high=1, size=n_particles))
        r = np.where(u > 1, 2 - u, u)
        x, y = r_x * r * np.cos(t), r_y * r * np.sin(t)
        t = 2 * np.pi * np.random.uniform(low=-0.5, high=0.5, size=n_particles)
        rp = np.sqrt(1. - r**2)
        xp, yp = r_xp * rp * np.cos(t), r_yp * rp * np.sin(t)
        return [x, xp, y, yp]
    return _kv4d


class HEADTAILcoords(object):
    '''The classic HEADTAIL phase space.'''
    coordinates = ('x', 'xp', 'y', 'yp', 'z', 'dp')
    transverse = coordinates[:4]
    longitudinal = coordinates[-2:]
