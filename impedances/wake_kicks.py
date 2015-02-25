"""
@class WakeKick
@author Kevin Li, Michael Schenk
@date July 2014
@brief Implementation of the wake kicks, i.e. of the elementary objects
       describing the effects of a wake field.
@copyright CERN
"""
from __future__ import division

import numpy as np
from scipy.constants import c
from abc import ABCMeta, abstractmethod

from . import Printing

class WakeKick(Printing):
    """ Abstract base class for wake kick classes, like e.g. the
    DipoleWakeKickX.
    Provides the basic and universal methods to calculate the strength
    of a wake kick. Two implementations of the convolution are
    available. Based on what slicer mode (uniform_bin, uniform_charge)
    is used, the self._convolution method is bound to one or the other.
    The self.apply(bunch, slice_set) method calculates and applies the
    corresponding kick to the particles of the bunch that are located
    inside the slicing region defined by a slice_set. This should be
    the only method to be implemented for a child class inheriting from
    the WakeKick class. """

    __metaclass__ = ABCMeta

    def __init__(self, wake_function, slicer, n_turns_wake,
                 *args, **kwargs):
        """ Universal constructor for WakeKick objects. The slicer_mode
        is passed only to decide about which of the two implementations
        of the convolution the self._convolution method is bound to. """
        self.wake_function = wake_function

        if (slicer.mode == 'uniform_bin' and
            (n_turns_wake == 1 or slicer.z_cuts)):
            self._convolution = self._convolution_numpy
            self.warns('*** WARNING: Acceleration not handled properly' +
                       ' by this kind of convolution due to changing' +
                       ' bunch length!')
        else:
            self._convolution = self._convolution_dot_product

        self.n_turns_wake = n_turns_wake

    @abstractmethod
    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies the corresponding wake kick to the
        bunch conjugate momenta using the given slice_set. Only
        particles within the slicing region, i.e particles_within_cuts
        (defined by the slice_set) experience a kick. """
        pass

    @staticmethod
    def _wake_factor(bunch):
        """ Universal scaling factor for the strength of a wake field
        kick. """
        wake_factor = (-(bunch.charge)**2 / (bunch.mass * bunch.gamma *
                       (bunch.beta * c)**2) * bunch.particlenumber_per_mp)
        return wake_factor

    def _convolution_dot_product(self, target_times, source_times,
                                 source_moments):
        """ Implementation of the convolution of wake and source_moments
        (beam profile) using the numpy dot product. To be used with the
        'uniform_charge' slicer mode. """
        dt_to_target_slice = (
            [target_times] - np.transpose([source_times]))
        wake = self.wake_function(dt_to_target_slice)

        return np.dot(source_moments, wake)

    def _convolution_numpy(self, target_times, source_times, source_moments):
        """ Implementation of the convolution of wake and source_moments
        (longitudinal beam profile) using the numpy built-in
        numpy.convolve method. Recommended use with the 'uniform_bin'
        slicer mode (in case of multiturn wakes, additional conditions
        must be fulfilled: fixed z_cuts and no acceleration!) for
        higher performance. """
        dt_to_target_slice = np.concatenate(
            (target_times - source_times[-1],
            (target_times - source_times[0])[1:]))
        wake = self.wake_function(dt_to_target_slice)

        return np.convolve(source_moments, wake, 'valid')

    def _accumulate_source_signal(self, bunch, times_list, moments_list,
                                  ages_list):
        """ Accumulate (multiturn-)wake signals left by source slices.
        Takes a list of slice set attributes and adds up all
        convolutions weighted by the respective moments. Also updates
        the age of each slice set."""
        target_times = times_list[0]
        accumulated_signal = 0

        if len(ages_list) < self.n_turns_wake:
            n_turns = len(ages_list)
        else:
            n_turns = self.n_turns_wake

        for i in range(n_turns):
            source_times = times_list[i] + ages_list[i]
            source_moments = moments_list[i]
            accumulated_signal += self._convolution(target_times, source_times,
                                                    source_moments)

        return self._wake_factor(bunch) * accumulated_signal


""" Constant wake kicks """

class ConstantWakeKickX(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a constant wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        constant_kick = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.xp[p_idx] += constant_kick.take(s_idx)


class ConstantWakeKickY(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a constant wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        constant_kick = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.yp[p_idx] += constant_kick.take(s_idx)


class ConstantWakeKickZ(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a constant wake kick to bunch.dp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        constant_kick = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.dp[p_idx] += constant_kick.take(s_idx)


""" Dipolar wake kicks """

class DipoleWakeKickX(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_x
                        for s in slice_set_list]
        dipole_kick_x = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.xp[p_idx] += dipole_kick_x.take(s_idx)


class DipoleWakeKickXY(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar (cross term x-y) wake kick
        to bunch.xp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_y
                        for s in slice_set_list]
        dipole_kick_xy = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.xp[p_idx] += dipole_kick_xy.take(s_idx)


class DipoleWakeKickY(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_y
                        for s in slice_set_list]
        dipole_kick_y = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.yp[p_idx] += dipole_kick_y.take(s_idx)


class DipoleWakeKickYX(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar (cross term y-x) wake kick
        to bunch.yp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_x
                        for s in slice_set_list]
        dipole_kick_yx = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.yp[p_idx] += dipole_kick_yx.take(s_idx)


""" Quadrupolar wake kicks """

class QuadrupoleWakeKickX(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_x = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.xp[p_idx] += quadrupole_kick_x.take(s_idx) * bunch.x.take(p_idx)


class QuadrupoleWakeKickXY(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar (cross term x-y) wake
        kick to bunch.xp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_xy = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.xp[p_idx] += quadrupole_kick_xy.take(s_idx) * bunch.y.take(p_idx)


class QuadrupoleWakeKickY(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_y = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.yp[p_idx] += quadrupole_kick_y.take(s_idx) * bunch.y.take(p_idx)


class QuadrupoleWakeKickYX(WakeKick):

    def apply(self, bunch, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar (cross term y-x) wake
        kick to bunch.yp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_yx = self._accumulate_source_signal(
            bunch, times_list, moments_list, slice_set_age_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.yp[p_idx] += quadrupole_kick_yx.take(s_idx) * bunch.x.take(p_idx)
