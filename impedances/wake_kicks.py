"""
@copyright CERN
"""
from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.constants import c
from scipy.signal import fftconvolve

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
            if slicer.n_slices < 400:
                self._convolution = self._convolution_numpy
            else:
                self._convolution = self._convolution_scipy
            self.warns('Acceleration not handled properly' +
                       ' by this kind of convolution due to changing' +
                       ' bunch length!')
        else:
            self._convolution = self._convolution_dot_product

        self.n_turns_wake = n_turns_wake

    @abstractmethod
    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies the corresponding wake kick to the
        bunch conjugate momenta using the given slice_set. Only
        particles within the slicing region, i.e particles_within_cuts
        (defined by the slice_set) experience a kick. """
        pass

    @staticmethod
    def _wake_factor(bunch):
        """ Universal scaling factor for the strength of a wake field
        kick. """
        wake_factor = (-(bunch.charge)**2 /
                       (bunch.mass * bunch.gamma * (bunch.beta*c)**2)
                       * bunch.particlenumber_per_mp)
        return wake_factor

    def _get_slice_set_data(self, slice_set_list, moments=None):
        """
        slice_set is of dimensionality n_turns x n_bunches each with arrays of n_slices
        """
        ages_list = None #[s[0].age for s in slice_set_list]
        betas_list = [s[0].beta for s in slice_set_list]
        times_list = [[b.convert_to_time(b.z_centers) for b in s] for s in slice_set_list]
        if moments is not None:
            moments_list = [[b.n_macroparticles_per_slice*getattr(b, moments) for b in s]
                            for s in slice_set_list]
        else:
            moments_list = [[b.n_macroparticles_per_slice for b in s]
                            for s in slice_set_list]

        return np.array(times_list), np.array(ages_list), np.array(moments_list), np.array(betas_list)

    def _convolution_dot_product(self, target_times,
                                 source_times, source_moments, source_beta):
        """ Implementation of the convolution of wake and source_moments
        (beam profile) using the numpy dot product. To be used with the
        'uniform_charge' slicer mode. """
        dt_to_target_slice = (
            [target_times] - np.transpose([source_times]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)

        return np.dot(source_moments, wake)

    def _convolution_numpy(self, target_times,
                           source_times, source_moments, source_beta):
        """ Implementation of the convolution of wake and source_moments
        (longitudinal beam profile) using the numpy built-in
        numpy.convolve method. Recommended use with the 'uniform_bin'
        slicer mode (in case of multiturn wakes, additional conditions
        must be fulfilled: fixed z_cuts and no acceleration!) for
        higher performance. Question: how about interpolation to avoid
        expensive dot product in most cases? """
        tmin, tmax = source_times[0], source_times[-1]
        dt_to_target_slice = np.concatenate(
            (target_times-tmax, (target_times - tmin)[1:]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)

        return np.convolve(source_moments, wake, 'valid')

    def _convolution_scipy(self, target_times,
                           source_times, source_moments, source_beta):
        """Implementation of the convolution of wake and source_moments (longitudinal
        beam profile) using the scipy.signal built-in fftconvolve method. This
        should be faster than the numpy version since it exploits the fft for
        perforing the convolution. It starts paying off for n_slices>400
        (empirically) e.g. factor 2 for n_slices=600.

        """
        tmin, tmax = source_times[0], source_times[-1]
        dt_to_target_slice = np.concatenate(
            (target_times-tmax, (target_times-tmin)[1:]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)

        return fftconvolve(wake, source_moments, 'valid')

    def _convolve_multibunch(self, times, moments, dt=None,
                             f_convolution=None):
        """Multibunch convolution according to Notebook implementation.

        Takes a list of times and moments from all bunches and performs the
        (multibunch) convolution accordingly.

        """
        kick = []

        if dt is None: dt = 0.*np.array(times)
        if f_convolution: f_convolution = self._convolution

        for i in xrange(len(times)):
            z = 0.*times[i]
            target_times = times[i]
            for j in range(i+1):
                source_times = times[j] + (dt[i] - dt[j])
                source_moments = moments[j]
                z += f_convolution(target_times, source_times, source_moments)

            kick.append(z)

        return kick

    def _accumulate_source_signal(self, bunch, times_list, ages_list,
                                  moments_list, betas_list):
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

        # Source beta is not needed?!?!
        for i in range(n_turns):
            source_times = times_list[i] + ages_list[i]
            source_beta = betas_list[i]
            source_moments = moments_list[i]
            accumulated_signal += self._convolution(
                target_times, source_times, source_moments, source_beta)

        return self._wake_factor(bunch) * accumulated_signal

    def _accumulate_source_signal_multibunch(self,
            bunches, times_list, ages_list, moments_list, betas_list):
        """

        Arguments:

        bunch -- bunch or list of bunches - the order is important; index 0 is
                 assumed to be the front most bunch i.e., the head of the b
        ages_list -- list with delay in [s] for each slice set since wake generation
        times_list, moments_list, betas_list -- 2d array (turns x bunches) with history for each bunch

        """

        accumulated_signal_list = []
        bunches = np.atleast_1d(bunches)
        dt_list = [b.dt for b in bunches]

        # Check for strictly ascending order
        try:
            assert(all(earlier <= later for earlier, later in zip(dt_list, dt_list[1:])))
        except AssertionError as err:
            print err.message()
            print "Bunches are not consecutive. I will re-arrange them now..."
            bunches = sorted(bunches, key=dt_list)

        if len(ages_list) < self.n_turns_wake:
            n_turns = len(ages_list)
        else:
            n_turns = self.n_turns_wake

        for i, b in enumerate(bunches):
            n_bunches_infront = i+1
            accumulated_signal = 0
            target_times = times_list[0,i]

            # Accumulate all bunches over all turns
            for k in xrange(n_turns):
                if k>0:
                    n_bunches_infront = len(bunches)
                for j in xrange(n_bunches_infront): # run over all bunches and take into account wake in front - test!
                    source_times = times_list[k,j] + ages_list[k] + dt_list[i] - dt_list[j]
                    source_moments = moments_list[k,j]
                    source_beta = betas_list[k]

                    accumulated_signal += self._convolution(
                        target_times, source_times, source_moments, source_beta)

            accumulated_signal_list.append(self._wake_factor(b) * accumulated_signal)

        self.dxp = accumulated_signal_list # For test and debugging purpose only
        return accumulated_signal_list


'''
==============================================================
Below we are to put the implemetation of any order wake kicks.
==============================================================
'''

class ConstantWakeKickX(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a constant wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        ages_list = slice_set_age_list#[s.age for s in slice_set_list]
        # times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        # betas_list = [s.beta for s in slice_set_list]
        # moments_list = [s.n_macroparticles_per_slice
        #                 for s in slice_set_list]
        times_list, ages_list, moments_list, betas_list = self._get_slice_set_data(slice_set_list)
        # import ipdb
        # ipdb.set_trace()

        constant_kick = self._accumulate_source_signal_multibunch(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        return times_list, slice_set_age_list, moments_list, betas_list
        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.xp[p_idx] += constant_kick.take(s_idx)


class ConstantWakeKickY(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a constant wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        constant_kick = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.yp[p_idx] += constant_kick.take(s_idx)


class ConstantWakeKickZ(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a constant wake kick to bunch.dp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        constant_kick = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.dp[p_idx] += constant_kick.take(s_idx)


class DipoleWakeKickX(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_x
                        for s in slice_set_list]
        dipole_kick_x = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunch.xp[p_idx] += dipole_kick_x.take(s_idx)


class DipoleWakeKickXY(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar (cross term x-y) wake kick
        to bunch.xp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_y
                        for s in slice_set_list]
        dipole_kick_xy = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunches.xp[p_idx] += dipole_kick_xy.take(s_idx)


class DipoleWakeKickY(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_y
                        for s in slice_set_list]
        dipole_kick_y = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunches.yp[p_idx] += dipole_kick_y.take(s_idx)


class DipoleWakeKickYX(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a dipolar (cross term y-x) wake kick
        to bunch.yp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice*s.mean_x
                        for s in slice_set_list]
        dipole_kick_yx = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunches.yp[p_idx] += dipole_kick_yx.take(s_idx)


class QuadrupoleWakeKickX(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_x = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunches.xp[p_idx] += quadrupole_kick_x.take(s_idx) * bunches.x.take(p_idx)


class QuadrupoleWakeKickXY(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar (cross term x-y) wake
        kick to bunch.xp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_xy = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunches.xp[p_idx] += quadrupole_kick_xy.take(s_idx) * bunches.y.take(p_idx)


class QuadrupoleWakeKickY(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_y = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunches.yp[p_idx] += quadrupole_kick_y.take(s_idx) * bunches.y.take(p_idx)


class QuadrupoleWakeKickYX(WakeKick):

    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """ Calculates and applies a quadrupolar (cross term y-x) wake
        kick to bunch.yp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick. """
        times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
        betas_list = [s.beta for s in slice_set_list]
        moments_list = [s.n_macroparticles_per_slice
                        for s in slice_set_list]
        quadrupole_kick_yx = self._accumulate_source_signal(
            bunches, times_list, slice_set_age_list, moments_list, betas_list)

        p_idx = slice_set_list[0].particles_within_cuts
        s_idx = slice_set_list[0].slice_index_of_particle.take(p_idx)
        bunches.yp[p_idx] += quadrupole_kick_yx.take(s_idx) * bunches.x.take(p_idx)
