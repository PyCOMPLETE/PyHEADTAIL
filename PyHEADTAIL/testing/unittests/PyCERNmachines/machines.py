from __future__ import division

import numpy as np
from scipy.constants import c, e, m_p

from PyHEADTAIL.general.element import Element
import PyHEADTAIL.particles.generators as gen
# try:
#     from PyHEADTAIL.trackers.transverse_tracking_cython import TransverseMap
#     from PyHEADTAIL.trackers.detuners_cython import (Chromaticity,
#                                                      AmplitudeDetuning)
# except ImportError as e:
#     print ("*** Warning: could not import cython variants of trackers, "
#            "did you cythonize (use the following command)?\n"
#            "$ ./install \n"
#            "Falling back to (slower) python version.")
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap, RFSystems


class Synchrotron(Element):

    def __init__(self, *args, **kwargs):
        '''
        Currently (because the RFSystems tracking uses a Verlet
        velocity integrator) the RFSystems element will be installed at
        s == circumference/2, which is correct for the smoothip
        approximation.
        '''
        super(Synchrotron, self).__init__(*args, **kwargs)

        self.chromaticity_on = kwargs.pop('chromaticity_on', True)
        self.amplitude_detuning_on = kwargs.pop('amplitude_detuning_on', True)

        if not hasattr(self, 'longitudinal_focusing'):
            self.longitudinal_focusing = kwargs.pop('longitudinal_focusing')
        if self.longitudinal_focusing not in ['linear', 'non-linear']:
            raise ValueError('longitudinal_focusing not recognized!!!')

        for attr in kwargs.keys():
            if kwargs[attr] is not None:
                self.prints('Synchrotron init. From kwargs: %s = %s'
                            % (attr, repr(kwargs[attr])))
                setattr(self, attr, kwargs[attr])

        self.create_transverse_map(self.chromaticity_on,
                                   self.amplitude_detuning_on)

        # create the one_turn map: install the longitudinal map at
        # s = circumference/2
        self.one_turn_map = []
        for m in self.transverse_map:
            self.one_turn_map.append(m)

        # compute the index of the element before which to insert
        # the longitudinal map
        for insert_before, si in enumerate(self.s):
            if si > 0.5 * self.circumference:
                break
        insert_before -= 1
        n_segments = len(self.transverse_map)
        self.create_longitudinal_map(insert_before)
        self.one_turn_map.insert(insert_before, self.longitudinal_map)

    def install_after_each_transverse_segment(self, element_to_add):
        '''Attention: Do not add any elements which update the dispersion!'''
        one_turn_map_new = []
        for element in self.one_turn_map:
            one_turn_map_new.append(element)
            if element in self.transverse_map:
                one_turn_map_new.append(element_to_add)
        self.one_turn_map = one_turn_map_new

    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma**-2)
        self._betagamma = np.sqrt(self.gamma**2 - 1)
        self._p0 = self.betagamma * m_p * c

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1-value**2)

    @property
    def betagamma(self):
        return self._betagamma
    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value**2 + 1)

    @property
    def p0(self):
        return self._p0
    @p0.setter
    def p0(self, value):
        self.gamma = value / (m_p * self.beta * c)

    @property
    def eta(self):
        return self.alpha - self.gamma**-2

    @property
    def Q_s(self):
        if hasattr(self, '_Q_s'):
            return self._Q_s
        else:
            if self.p_increment!=0 or self.dphi1!=0:
                raise ValueError('Formula not valid in this case!!!!')
            return np.sqrt( e*np.abs(self.eta)*(self.h1*self.V1 + self.h2*self.V2)
                            / (2*np.pi*self.p0*self.beta*c) )
    @Q_s.setter
    def Q_s(self, value):
        self._Q_s = value

    @property
    def beta_z(self):
        return np.abs(self.eta)*self.circumference/2./np.pi/self.Q_s

    @property
    def R(self):
        return self.circumference/(2*np.pi)

    def track(self, bunch, verbose=False):
        for m in self.one_turn_map:
            if verbose:
                self.prints('Tracking through:\n' + str(m))
            m.track(bunch)

    def create_transverse_map(self, chromaticity_on=True,
                              amplitude_detuning_on=True):
        detuners = []
        if chromaticity_on:
            detuners.append(Chromaticity(self.Qp_x, self.Qp_y))
        if amplitude_detuning_on:
            detuners.append(
                AmplitudeDetuning(self.app_x, self.app_y, self.app_xy)
            )

        self.transverse_map = TransverseMap(
            self.s,
            self.alpha_x, self.beta_x, self.D_x,
            self.alpha_y, self.beta_y, self.D_y,
            self.Q_x, self.Q_y, detuners, printer=self._printer)

    def create_longitudinal_map(self, one_turn_map_insert_idx=0):
        if self.longitudinal_focusing == 'linear':
            self.longitudinal_map = LinearMap(
                [self.alpha],
                self.circumference, self.Q_s,
                D_x=self.D_x[one_turn_map_insert_idx],
                D_y=self.D_y[one_turn_map_insert_idx]
            )
        elif self.longitudinal_focusing == 'non-linear':
            self.longitudinal_map = RFSystems(
                self.circumference, [self.h1, self.h2],
                [self.V1, self.V2], [self.dphi1, self.dphi2],
                [self.alpha], self.gamma, self.p_increment,
                D_x=self.D_x[one_turn_map_insert_idx],
                D_y=self.D_y[one_turn_map_insert_idx],
                charge=self.charge,
                mass=self.mass,
            )
        else:
            raise NotImplementedError(
                'Something wrong with self.longitudinal_focusing')

    def generate_6D_Gaussian_bunch(self, n_macroparticles, intensity,
                                   epsn_x, epsn_y, sigma_z):
        '''Generate a 6D Gaussian distribution of particles which is
        transversely matched to the Synchrotron. Longitudinally, the
        distribution is matched only in terms of linear focusing.
        For a non-linear bucket, the Gaussian distribution is cut along
        the separatrix (with some margin). It will gradually filament
        into the bucket. This will change the specified bunch length.
        '''
        if self.longitudinal_focusing == 'linear':
            check_inside_bucket = lambda z,dp : np.array(len(z)*[True])
        elif self.longitudinal_focusing == 'non-linear':
            check_inside_bucket = self.longitudinal_map.get_bucket(
                gamma=self.gamma).make_is_accepted(margin=0.05)
        else:
            raise NotImplementedError(
                'Something wrong with self.longitudinal_focusing')

        beta_z    = np.abs(self.eta)*self.circumference/2./np.pi/self.Q_s
        sigma_dp  = sigma_z/beta_z
        epsx_geo = epsn_x/self.betagamma
        epsy_geo = epsn_y/self.betagamma

        bunch = gen.ParticleGenerator(macroparticlenumber=n_macroparticles,
                intensity=intensity, charge=self.charge, mass=self.mass,
                circumference=self.circumference, gamma=self.gamma,
                distribution_x=gen.gaussian2D(epsx_geo),
                alpha_x=self.alpha_x[0], beta_x=self.beta_x[0], D_x=self.D_x[0],
                distribution_y=gen.gaussian2D(epsy_geo),
                alpha_y=self.alpha_y[0], beta_y=self.beta_y[0], D_y=self.D_y[0],
                distribution_z=gen.cut_distribution(
                    gen.gaussian2D_asymmetrical(
                        sigma_u=sigma_z, sigma_up=sigma_dp),
                    is_accepted=check_inside_bucket)
                ).generate()

        return bunch

    def generate_6D_Gaussian_bunch_matched(
            self, n_macroparticles, intensity, epsn_x, epsn_y,
            sigma_z=None, epsn_z=None, margin=0):
        '''Generate a 6D Gaussian distribution of particles which is
        transversely as well as longitudinally matched.
        The distribution is found iteratively to exactly yield the
        given bunch length while at the same time being stationary in
        the non-linear bucket. Thus, the bunch length should amount
        to the one specificed and should not change significantly
        during the synchrotron motion.

        Requires self.longitudinal_focusing == 'non-linear'
        for the bucket.
        '''
        assert self.longitudinal_focusing == 'non-linear'
        epsx_geo = epsn_x/self.betagamma
        epsy_geo = epsn_y/self.betagamma

        bunch = gen.ParticleGenerator(macroparticlenumber=n_macroparticles,
                intensity=intensity, charge=self.charge, mass=self.mass,
                circumference=self.circumference, gamma=self.gamma,
                distribution_x=gen.gaussian2D(epsx_geo),
                alpha_x=self.alpha_x[0], beta_x=self.beta_x[0], D_x=self.D_x[0],
                distribution_y=gen.gaussian2D(epsy_geo),
                alpha_y=self.alpha_y[0], beta_y=self.beta_y[0], D_y=self.D_y[0],
                distribution_z=gen.RF_bucket_distribution(
                    rfbucket=self.longitudinal_map.get_bucket(gamma=self.gamma),
                    sigma_z=sigma_z, epsn_z=epsn_z, margin=margin,
                    printer=self._printer)
                ).generate()

        return bunch
