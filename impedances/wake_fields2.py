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

    def __init__(self, wake_sources, slices):

        wake_keys = ['constant_x',
                     'constant_y',
                     'dipole_xx',
                     'dipole_xy',
                     'dipole_yx',
                     'dipole_yy',
                     'quadrupole_x',
                     'quadrupole_y',
                     'longitudinal'
                     ]
        self.wake_functions = {}
        [self.wake_functions[k] = None for k in wake_keys]

        for ws in wake_sources:
            for key in ws.wake_functions.keys():
                self.wake_functions[key] = ws.wake_functions[key]

        if slices.mode == 'const_charge':
            self.convolution = self._convolution_dot_product
        else:
            self.convolution = self._convolution_numpy
        self.slices = slices

    def _wake_factor(self, bunch):
        particles_per_macroparticle = bunch.intensity / bunch.n_macroparticles
        return -(bunch.charge) ** 2 / (bunch.mass * bunch.gamma * (bunch.beta * c) ** 2) * particles_per_macroparticle

    def _wakefield_kick_x(self, beam, plane):

        ix = self.slices.slice_index_of_particle
        zeroth_moment = self.slices.n_macroparticles
        first_moment_x = self.slices.n_macroparticles * self.slices.mean_x
        first_moment_y = self.slices.n_macroparticles * self.slices.mean_y

        try:
            constant_kick = self.convolution(bunch, self.wake_functions['constant_x'], zeroth_moment)
        try:
            dipole_kick = self.convolution(bunch, self.wake_functions['dipole_xx'], first_moment_x)
        try:
            quadrupole_kick = self.convolution(bunch, self.wake_functions['quadrupole_x'], zeroth_moment)

        beam.xp += constant_kick[ix] + dipole_kick_x[ix] + quadrupole_kick[ix] * beam.x

    def _wakefield_kick_y(self, beam):

        ix = self.slices.slice_index_of_particle
        zeroth_moment = self.slices.n_macroparticles
        first_moment_x = self.slices.n_macroparticles * self.slices.mean_x
        first_moment_y = self.slices.n_macroparticles * self.slices.mean_y

        try:
            constant_kick = self.convolution(bunch, self.wake_functions['constant_x'], zeroth_moment)
        try:
            dipole_kick_x = self.convolution(bunch, self.wake_functions['dipole_yy'], first_moment_y)
        try:
            quadrupole_kick = self.convolution(bunch, self.wake_functions['quadrupole_x'], zeroth_moment)

        beam.yp += constant_kick[ix] + dipole_kick_x[ix] + quadrupole_kick[ix] * beam.y

    def _wakefield_kick_z(self, beam):

        ix = self.slices.slice_index_of_particle
        zeroth_moment = self.slices.n_macroparticles

        longitudinal_kick = self.convolution(beam, self.wake_functions['longitudinal'], zeroth_moment)

        beam.dp += longitudinal_kick[ix]

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

        self._wakefield_kick_x(beam)
        self._wakefield_kick_y(beam)
        self._wakefield_kick_z(beam)
