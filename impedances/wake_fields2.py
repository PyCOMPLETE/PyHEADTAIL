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
    def __init__(self, wake_sources, slices):
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
        [self.wake_functions[k] = wake_sources.wake_functions[k] for k in wake_sources.wake_functions.keys()]

        if slices.mode == 'const_charge':
            self.convolution = self._convolution_dot_product
        else:
            self.convolution = self._convolution_numpy
        self.slices = slices

    def _wake_factor(self, bunch):
        particles_per_macroparticle = bunch.intensity / bunch.n_macroparticles
        return -(bunch.charge) ** 2 / (bunch.mass * bunch.gamma * (bunch.beta * c) ** 2) * particles_per_macroparticle

    def _wakefield_kicks(self, beam, plane):

        assert((plane=='x' or plane=='y', or plane=='z'))

        if plane=='x':
            beam_position = beam.x
            beam_angle = beam.xp
            zeroth_moment = self.slices.n_macroparticles
            first_moment = self.slices.n_macroparticles * self.slices.mean_x
        if plane=='y':
            beam_position = beam.y
            beam_angle = beam.yp
            zeroth_moment = self.slices.n_macroparticles
            first_moment = self.slices.n_macroparticles * self.slices.mean_y
        if plane=='z':
            beam_position = beam.z
            beam_angle = beam.dp
            zeroth_moment = self.slices.n_macroparticles

        ix = self.slices.slice_index_of_particle

        if plane in ['x', 'y']:
            try:
                constant_kick = self.convolution(bunch, self.wake_functions['constant_' + plane], zeroth_moment)
            try:
                dipole_kick = self.convolution(bunch, self.wake_functions['dipole_' + plane], first_moment)
            try:
                quadrupole_kick = self.convolution(bunch, self.wake_functions['quadrupole_' + plane], zeroth_moment)

            beam_angle += constant_kick[ix] + dipole_kick[ix] + quadrupole_kick[ix] * beam_position

        else:
            longitudinal_kick = self.convolution(bunch, self.wake_functions['longitudinal'], zeroth_moment)

            beam_angle += longitudinal_kick[ix]

    def _convolution_dot_product(self, bunch, f, g):

        dz_to_target_slice = [self.slices.z_centers] - np.transpose([self.slices.z_centers])
        wake = f(bunch, dz_to_target_slice)
        beam_profile = g
        kick = self._wake_factor(bunch) * np.dot(beam_profile, wake)

        return kick

    def _convolution_numpy(self, bunch, f, g):

        dz_to_target_slice = np.concatenate((self.slices.z_centers - self.slices.z_centers[-1],
                                            (self.slices.z_centers - self.slices.z_centers[0])[1:]))
        wake = f(bunch, dz_to_target_slice)
        beam_profile = g
        kick = self._wake_factor(bunch) * np.convolve(beam_profile, wake, 'valid')

        return kick

    def track(beam):
        # if not self.slices:
        # self.slices = bunch.slices
        # self.slices.compute_statistics(bunch)

        self._wakefield_kicks(beam, 'x')
        self._wakefield_kicks(beam, 'y')
        self._wakefield_kicks(beam, 'z')
