'''
@class Wakefields
@author Hannes Bartosik & Kevin Li & Giovanni Rumolo & Michael Schenk
@date March 2014
@Class for creation and management of wakefields from impedance sources
@copyright CERN
'''
from __future__ import division


from functools import partial
import numpy as np
from scipy.constants import c, e
from scipy.constants import physical_constants


sin = np.sin
cos = np.cos


class Wakefields(object):
    '''
    classdocs
    '''
    def __init__(self, wake_function_dictionary, slices):
        '''
        Constructor
        '''
        wake_keys = ['constant_xx',
                     'constant_xy',
                     'constant_yx',
                     'constant_yy',
                     'dipole_xx',
                     'dipole_xy',
                     'dipole_yx',
                     'dipole_yy',
                     'quadrupole_xx',
                     'quadrupole_xy',
                     'quadrupole_yx',
                     'quadrupole_yy',
                     'longitudinal'
                     ]
        self.wake_functions = {}
        [self.wake_functions[k] = None for k in wake_keys]
        [self.wake_functions[k] = wake_function_dictionary[k] for k in wake_function_dictionary.keys()]

        # self.constant_xx = None
        # self.constant_xy = None
        # self.constant_yx = None
        # self.constant_yy = None
        # self.dipole_xx = None
        # self.dipole_xy = None
        # self.dipole_yx = None
        # self.dipole_yy = None
        # self.quadrupole_xx = None
        # self.quadrupole_xy = None
        # self.quadrupole_yx = None
        # self.quadrupole_yy = None
        # self.longitudinal = None

        if slices.mode == 'const_charge':
            self.convolution = self._convolution_dot_product
        else:
            self.convolution = self._convolution_numpy
        self.slices = slices

    def wake_factor(self, bunch):
        particles_per_macroparticle = bunch.intensity / bunch.n_macroparticles
        return -(bunch.charge) ** 2 / (bunch.mass * bunch.gamma * (bunch.beta * c) ** 2) * particles_per_macroparticle

    def wakefield_kicks_x(self):

        ix = self.slices.slice_index_of_particle

        zeroth_moment = self.slices.n_macroparticles
        first_moment = self.slices.n_macroparticles * self.slices.mean_x

        try:
            constant_kick = self.convolution(bunch, self.wake_functions['constant_x'], zeroth_moment)
        try:
            dipole_kick = self.convolution(bunch, self.wake_functions['dipole_x'], first_moment)
        try:
            quadrupole_kick = self.convolution(bunch, self.wake_functions['quadrupole_x'], zeroth_moment)

        bunch.xp += constant_kick[ix]
                  + dipole_kick[ix]
                  + quadrupole_kick[ix] * bunch.x

    def wakefield_kicks_y(self): pass

    def wakefield_kicks_z(self): pass

    def transverse_wakefield_kicks(self, plane):
        assert(plane in ('x', 'y'))
        @profile
        def compute_apply_kicks(bunch):
            if plane == 'x':
                slice_position = self.slices.mean_x
                dipole_wake = self.dipole_wake_x
                quadrupole_wake = self.quadrupole_wake_x
                particle_position = bunch.x
                position_prime = bunch.xp
            if plane == 'y':
                slice_position = self.slices.mean_y
                dipole_wake = self.dipole_wake_y
                quadrupole_wake = self.quadrupole_wake_y
                particle_position = bunch.y
                position_prime = bunch.yp

            if self.slices.mode == 'const_charge':
                beam_profile = self.slices.n_macroparticles * slice_position
                self.dipole_kick = self._convolution_dot_product(bunch, dipole_wake, beam_profile)
            else:
                beam_profile = self.slices.n_macroparticles * slice_position
                self.dipole_kick = self._convolution_numpy(bunch, dipole_wake, beam_profile)

            #####################
            # quadrupole kicks
            dz_to_target_slice = [self.slices.z_centers] - np.transpose([self.slices.z_centers])
            self.quadrupolar_wake_sum = np.dot(self.slices.n_macroparticles, quadrupole_wake(bunch, dz_to_target_slice)) * self.wake_factor(bunch)

            # apply kicks
            position_prime += self.dipole_kick[self.slices.slice_index_of_particle] + self.quadrupolar_wake_sum[self.slices.slice_index_of_particle] * particle_position

        return compute_apply_kicks

    def _convolution_dot_product(self, bunch, f, g):

        dz_to_target_slice = [self.slices.z_centers] - np.transpose([self.slices.z_centers])
        wake = f(bunch, dz_to_target_slice)
        beam_profile = g
        kick = self.wake_factor(bunch) * np.dot(beam_profile, wake)

        return kick

    def _convolution_numpy(self, bunch, f, g):

        dz_to_target_slice = np.concatenate((self.slices.z_centers - self.slices.z_centers[-1],
                                            (self.slices.z_centers - self.slices.z_centers[0])[1:]))
        wake = f(bunch, dz_to_target_slice)
        beam_profile = g
        kick = self.wake_factor(bunch) * np.convolve(beam_profile, wake, 'valid')

        return kick

    #~ @profile
    def longitudinal_wakefield_kicks(self, bunch):
        wake = self.wake_longitudinal

        # matrix with distances to target slice
        dz_to_target_slice = [self.slices.z_centers] - np.transpose([self.slices.z_centers])

        # compute kicks
        self.longitudinal_kick = np.zeros(self.slices.n_slices)
        self.longitudinal_kick = np.dot(self.slices.n_macroparticles, wake(bunch, dz_to_target_slice)) * self.wake_factor(bunch)

        # apply kicks
        bunch.dp += self.longitudinal_kick[self.slices.slice_index_of_particle]

    def track(bunch):
        # if not self.slices:
        # self.slices = bunch.slices
        self.slices.compute_statistics(bunch)

        wakefield_kicks_x = self.transverse_wakefield_kicks('x')
        wakefield_kicks_x(bunch)
        wakefield_kicks_y = self.transverse_wakefield_kicks('y')
        wakefield_kicks_y(bunch)

        if ('dipolar_x' or 'quadrupolar_x') in self.wake_field_keys:
            wakefield_kicks_x = self.transverse_wakefield_kicks('x')
            wakefield_kicks_x(bunch)
        if ('dipolar_y' or 'quadrupolar_y') in self.wake_field_keys:
            wakefield_kicks_y = self.transverse_wakefield_kicks('y')
            wakefield_kicks_y(bunch)
        if 'longitudinal' in self.wake_field_keys:
            self.longitudinal_wakefield_kicks(bunch)

        self.longitudinal_wakefield_kicks(bunch)
