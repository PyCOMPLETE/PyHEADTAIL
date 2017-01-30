""".. copyright:: CERN"""
from __future__ import division

import numpy as np
from scipy.constants import c
from scipy.signal import fftconvolve

from abc import ABCMeta, abstractmethod

from . import Printing


class WakeKick(Printing):
    """Abstract base class for wake kick classes, like e.g. the DipoleWakeKickX.

    Provides the basic and universal methods to calculate magnitude and shape
    of a wake kick via numerical convolution.

    Several implementations of the convolutions are provided each with their
    own advantages in terms of flexiility and performance. The fast convolutios
    such as the numpy and scipy convolution functions require uniform
    sampling. Based on the input slicer configuration the self._convolution
    method is bound to one or the other convolution function.

    The self.apply(bunch, slice_set) method calculates and applies the
    corresponding kick to all macroparticles located within the slicing region
    defined by a slice_set of a bunch or several bunches. This should be the
    only method to be implemented for any child class inheriting from the
    WakeKick class.

    """

    __metaclass__ = ABCMeta

    def __init__(self, wake_function, slicer, n_turns_wake, *args, **kwargs):
        """Universal constructor for WakeKick objects. The slicer_mode is passed only
        to decide about which of the two implementations of the convolution the
        self._convolution method is bound to.

        """
        self.wake_function = wake_function

        if (slicer.mode == 'uniform_bin' and
           (n_turns_wake == 1 or slicer.z_cuts)):
            if slicer.n_slices < 400:
                self._convolution = self._convolution_numpy
            else:
                self._convolution = self._convolution_scipy
            # self.warns('Acceleration not handled properly' +
            #            ' by this kind of convolution due to changing' +
            #            ' bunch length!')
        else:
            self._convolution = self._convolution_dot_product

        self.n_turns_wake = n_turns_wake
        self.slicer = slicer

    @abstractmethod
    def apply(self, bunches, slice_set_list, slice_set_age_list):
        """Calculates and applies the corresponding wake kick to the bunch conjugate
        momenta using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set) experience
        a kick.

        """
        pass

    @staticmethod
    def _wake_factor(bunch):
        """Universal scaling factor for the strength of a wake field kick.

        """
        wake_factor = (-(bunch.charge)**2 /
                       (bunch.mass * bunch.gamma * (bunch.beta*c)**2) *
                       bunch.particlenumber_per_mp)
        return wake_factor

    # def _extract_slice_set_data(self, slice_set_list, moments='zero'):
    #     """Convenience function called by wake kicks to return slice_set
    #     quantities. Slice set list and slice set quantities are of
    #     dimensionality n_turns x n_bunches each with arrays of n_slices

    #     """

    #     slice_set_list = np.array(slice_set_list)
    #     ages_list = slice_set_list[:, 0, :, :]
    #     betas_list = slice_set_list[:, 1, :, :]
    #     times_list = slice_set_list[:, 2, :, :]
    #     if moments == 'zero':
    #         moments_list = slice_set_list[:, 3, :, :]
    #     elif moments == 'mean_x':
    #         moments_list = slice_set_list[:, 4, :, :]
    #     elif moments == 'mean_y':
    #         moments_list = slice_set_list[:, 5, :, :]
    #     else:
    #         raise ValueError("Please specify moments as either " +
    #                          "'zero', 'mean_x' or 'mean_y'!")

    #     return times_list, ages_list, moments_list, betas_list

    def _convolution_dot_product(self, target_times,
                                 source_times, source_moments, source_beta):
        """Implementation of the convolution of wake and source_moments (beam profile)
        using the numpy dot product. To be used with the 'uniform_charge'
        slicer mode.

        """
        dt_to_target_slice = (
            [target_times] - np.transpose([source_times]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)

        return np.dot(source_moments, wake)

    def _convolution_numpy(self, target_times,
                           source_times, source_moments, source_beta):
        """Implementation of the convolution of wake and source_moments (longitudinal
        beam profile) using the numpy built-in numpy.convolve
        method. Recommended use with the 'uniform_bin' slicer mode (in case of
        multiturn wakes, additional conditions must be fulfilled: fixed z_cuts
        and no acceleration!) for higher performance. Question: how about
        interpolation to avoid expensive dot product in most cases?

        """
        tmin, tmax = source_times[0], source_times[-1]
        dt_to_target_slice = np.concatenate(
            (target_times-tmax, (target_times-tmin)[1:]))
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

    def _accumulate_source_signal(self, bunch, times_list, ages_list,
                                  moments_list, betas_list):
        """Accumulate (multiturn-)wake signals left by source slices.  Takes a list of
        slice set attributes and adds up all convolutions weighted by the
        respective moments. Also updates the age of each slice set.

        """
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

    def _accumulate_source_signal_multibunch(
            self, bunches, slice_set_list, moments='zero', bunch_offset=0):
        """Args:

            bunches: bunch or list of bunches - the order is important; index 0
                is assumed to be the front most bunch i.e., the head of the b

            ages_list: list with delay in [s] for each slice set since
                 wake generation

            times_list, moments_list, betas_list: 2d array (turns x bunches)
                with history for each bunch

        """

        # Isolate the local_bunch_indexes from slice_set_list. After that, it
        # can be treated as homogeneous ndarray
        bunches = np.atleast_1d(bunches)
        local_bunch_indexes = slice_set_list[0][-1]
        slice_set_list = np.array([s[:-1] for s in slice_set_list])

        ages_list = slice_set_list[:, 0, :, :]
        betas_list = slice_set_list[:, 1, :, :]
        times_list = slice_set_list[:, 2, :, :]
        charge_list = slice_set_list[:, 3, :, :]
        mean_x_list = slice_set_list[:, 4, :, :]
        mean_y_list = slice_set_list[:, 5, :, :]
        if moments == 'zero':
            moments_list = charge_list
        elif moments == 'mean_x':
            moments_list = charge_list * mean_x_list
        elif moments == 'mean_y':
            moments_list = charge_list * mean_y_list
        else:
            raise ValueError("Please specify moments as either " +
                             "'zero', 'mean_x' or 'mean_y'!")

        accumulated_signal_list = []
        # print moments_list
        # print betas_list

        # Check for strictly descending order
        z_delays = [b.mean_z() for b in bunches]
        assert all(earlier <= later
                   for later, earlier in zip(z_delays, z_delays[1:])), \
            ("Bunches are not ordered. Make sure that bunches are in" +
             " descending order in time.")

        # Here, we need to take into account the fact that source slice
        # and target bunch lists are different
        n_turns, n_sources, n_slices = times_list.shape
        if n_turns > self.n_turns_wake:
            n_turns = self.n_turns_wake  # limit to this particular wake length

        # Tricky... this assumes one set of bunches - bunch 'delay' is missing
        for i, b in enumerate(bunches):
            n_bunches_infront = n_sources  # i+1  # <-- usually n_sources!
            # not strictly needed, should be solved automatically
            # by wake function decaying fast in front
            accumulated_signal = 0
            target_times = times_list[0, local_bunch_indexes[i]]

            # Accumulate all bunches over all turns
            for k in xrange(n_turns):
                if k > 0:
                    n_bunches_infront = n_sources
                    # run over all bunches and take into account wake in front
                    # - test!
                for j in xrange(n_bunches_infront):
                    source_beta = betas_list[k, j]
                    source_times = (times_list[k, j] + ages_list[k, j])
                    source_moments = moments_list[k, j]

                    accumulated_signal += self._convolution(
                        target_times, source_times,
                        source_moments, source_beta)

            # accumulated_signal_list.append(
            #     self._wake_factor(b) * accumulated_signal)
            accumulated_signal_list.append(
                1 * accumulated_signal)

        return accumulated_signal_list


# ==============================================================
# Below we are to put the implemetation of any order wake kicks.
# ==============================================================
class ConstantWakeKickX(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a constant wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        constant_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list)

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += constant_kick[i].take(s_idx)


class ConstantWakeKickY(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a constant wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        constant_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list)

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += constant_kick[i].take(s_idx)


class ConstantWakeKickZ(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a constant wake kick to bunch.dp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        constant_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list)

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.dp[p_idx] += constant_kick[i].take(s_idx)


class DipoleWakeKickX(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a dipolar wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        dipole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list, moments='mean_x')

        # And then get slices of actual bunches list
        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += dipole_kick[i].take(s_idx)


class DipoleWakeKickXY(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a dipolar (cross term x-y) wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        dipole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list, moments='mean_y')

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += dipole_kick[i].take(s_idx)


class DipoleWakeKickY(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a dipolar wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        dipole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list, moments='mean_y')

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += dipole_kick[i].take(s_idx)


class DipoleWakeKickYX(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a dipolar (cross term y-x) wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        dipole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list, moments='mean_x')

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += dipole_kick[i].take(s_idx)


class QuadrupoleWakeKickX(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a quadrupolar wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        quadrupole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list)

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.x.take(p_idx))


class QuadrupoleWakeKickXY(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a quadrupolar (cross term x-y) wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        quadrupole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list)

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.y.take(p_idx))


class QuadrupoleWakeKickY(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a quadrupolar wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        quadrupole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list)

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.y.take(p_idx))


class QuadrupoleWakeKickYX(WakeKick):

    def apply(self, bunches, slice_set_list):
        """Calculates and applies a quadrupolar (cross term y-x) wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        quadrupole_kick = self._accumulate_source_signal_multibunch(
            bunches, slice_set_list)

        for i, b in enumerate(bunches):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.x.take(p_idx))
