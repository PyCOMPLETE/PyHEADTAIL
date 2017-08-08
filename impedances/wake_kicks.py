""".. copyright:: CERN"""
from __future__ import division

import numpy as np
import math

from scipy.constants import c
from scipy.signal import fftconvolve

from abc import ABCMeta, abstractmethod

from . import Printing

from PyHEADTAIL.mpi import mpi_data

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
    def apply(self, bunch_list, all_slice_sets, local_slice_sets,
              local_bunch_indexes, optimization_method,
              circumference):
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
        assert all(later <= earlier
                   for later, earlier in zip(z_delays, z_delays[1:])), \
            ("Bunches are not ordered. Make sure that bunches are in" +
             " descending order in time.")
        # assert all(earlier <= later
        #            for later, earlier in zip(z_delays, z_delays[1:])), \
        #     ("Bunches are not ordered. Make sure that bunches are in" +
        #      " descending order in time.")

# ========================================================================

        # Here, we need to take into account the fact that source slice
        # and target bunch lists are different
        n_turns, n_sources, n_slices = times_list.shape # includes all bunches
        if n_turns > self.n_turns_wake:
            n_turns = self.n_turns_wake  # limit to this particular wake length

        # HEADTAIL convention: bunch[0] is tail - bunch[-1] is head
        for i, b in enumerate(bunches):
            accumulated_signal = 0
            target_times = times_list[0, local_bunch_indexes[i]]

            # Accumulate all bunches over all turns
            n_bunches_infront= n_sources
            for k in range(n_turns):
                # if k > 0:
                #     n_bunches_infront = i + 1
                # else:
                #     n_bunches_infront = n_sources
                for j in range(n_bunches_infront)[::-1]:
                    source_beta = betas_list[k, j]
                    source_times = (times_list[k, j] + ages_list[k, j])
                    source_moments = moments_list[k, j]

                    accumulated_signal += self._convolution(
                        target_times, source_times,
                        source_moments, source_beta)

# ========================================================================

        # # Here, we need to take into account the fact that source slice
        # # and target bunch lists are different
        # n_turns, n_sources, n_slices = times_list.shape
        # if n_turns > self.n_turns_wake:
        #     n_turns = self.n_turns_wake  # limit to this particular wake length

        # # Tricky... this assumes one set of bunches - bunch 'delay' is missing
        # for i, b in enumerate(bunches):
        #     n_bunches_infront = n_sources  # i+1  # <-- usually n_sources!
        #     # not strictly needed, should be solved automatically
        #     # by wake function decaying fast in front
        #     accumulated_signal = 0
        #     target_times = times_list[0, local_bunch_indexes[i]]

        #     # Accumulate all bunches over all turns
        #     for k in xrange(n_turns):
        #         if k > 0:
        #             n_bunches_infront = n_sources
        #             # run over all bunches and take into account wake in front
        #             # - test!
        #         for j in xrange(n_bunches_infront):
        #             source_beta = betas_list[k, j]
        #             source_times = (times_list[k, j] + ages_list[k, j])
        #             source_moments = moments_list[k, j]

        #             accumulated_signal += self._convolution(
        #                 target_times, source_times,
        #                 source_moments, source_beta)

# ========================================================================

            accumulated_signal_list.append(
                self._wake_factor(b) * accumulated_signal)
            # accumulated_signal_list.append(
            #     1* accumulated_signal)

        return accumulated_signal_list

    def _init_loop_minimized(self, all_slice_sets, local_slice_sets, bunch_list,
                                local_bunch_indexes, circumference, h_rf, h_bunch):

        bunch_spacing = circumference/float(h_bunch)
        # total number of bunches
        self._n_target_bunches = len(local_bunch_indexes)
        n_source_bunches = len(all_slice_sets)

        # number of slices per bunch
        n_slices = len(all_slice_sets[0].mean_x)

        # Valid convolution of the noncontinous (partially defined) data requires that
        # the length of the wake function is at lest twice of the length of the bunch.
        # Thus, extra 'slices' have been added to the both sides of the bunch to
        # the dashed_wake_function

        # number of extra slices added to each side of the bunch
        empty_space_per_side = int(math.ceil(n_slices/2.))

        # total number of bins per bunch in dashed_wake_functions
        self._n_bins_per_kick = (n_slices + 2*empty_space_per_side)
        # total length of the dashed_wake_functions
        total_array_length = self.n_turns_wake * self._n_target_bunches * self._n_bins_per_kick

        self._n_bins_per_turn = self._n_target_bunches * self._n_bins_per_kick

        self._dashed_wake_functions = []

        # we have only one list
        self._accumulated_signal_list = []
        self._accumulated_kick = np.zeros(total_array_length)

        for i in xrange(n_source_bunches):
            self._dashed_wake_functions.append(np.zeros(total_array_length))


        # calculates the mid points of the bunches from the z_bins
        # the bunch_id could be used here
        bunch_mids = []
        for slice_set in all_slice_sets:
            bunch_mids.append(((slice_set.z_bins[0]+slice_set.z_bins[-1])/2.))

        # calculates normalized bin coordinates for a bin set in the dashed_wake_functions
        raw_z_bins = local_slice_sets[0].z_bins
        raw_z_bins = raw_z_bins - ((raw_z_bins[0]+raw_z_bins[-1])/2.)
        bin_width = np.mean(raw_z_bins[1:]-raw_z_bins[:-1])
        original_z_bin_mids = (raw_z_bins[1:]+raw_z_bins[:-1])/2.
        z_bin_mids = original_z_bin_mids[0] - np.linspace(empty_space_per_side, 1,
                                        empty_space_per_side)*bin_width
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids)
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids[-1] + np.linspace(1,
                               empty_space_per_side, empty_space_per_side)*bin_width)

        # loop over source buches
        for i, mid_i in enumerate(bunch_mids):
            z_values = np.zeros(total_array_length)

            # loop of target bunches
            for j,target_bunch_idx in enumerate(local_bunch_indexes):

                source_mid = mid_i
                target_mid = bunch_mids[target_bunch_idx]

                # Calculates the distance difference between the source and the target bunches
                delta_mid = target_mid-source_mid
                delta_mid = -1.*delta_mid
                if delta_mid < 0.:
                    # the target bunch must be after the source bunch
                    delta_mid += circumference

#               This must be uncommented if the traces are compared to those given by
#               the different approaches. The lack of small rounding gives a notable difference.
#                delta_mid = bunch_spacing*round(delta_mid/bunch_spacing)

                if i == 0:
                    kick_from = empty_space_per_side + j * self._n_bins_per_kick
                    kick_to = empty_space_per_side + j * self._n_bins_per_kick + n_slices
                    self._accumulated_signal_list.append(np.array(self._accumulated_kick[kick_from:kick_to], copy=False))

                # z values for the wake functions
                for k in xrange(self.n_turns_wake):
                    idx_from = k * self._n_bins_per_turn + j * self._n_bins_per_kick
                    idx_to = k * self._n_bins_per_turn + (j + 1) * self._n_bins_per_kick

                    offset = (float(k) * circumference + delta_mid)
                    temp_mids = -z_bin_mids+offset
                    np.copyto(z_values[idx_from:idx_to],temp_mids)

            # calculates wake function values for each source bunch
            wake = np.zeros(len(z_values))
            value_map = (z_values>0.)
            wake[value_map] =  self.wake_function(-z_values[value_map]/c, beta=local_slice_sets[0].beta)
            np.copyto(self._dashed_wake_functions[i], wake)


    def _accumulate_loop_minimized(self, all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments, h_rf, h_bunch):

        if not hasattr(self,'_dashed_wake_functions'):
            self. _init_loop_minimized(all_slice_sets, local_slice_sets,
                                         bunch_list, local_bunch_indexes,
                                         circumference, h_rf, h_bunch)

        # Copies the perivous turn data forward
        np.copyto(self._accumulated_kick[:-1*self._n_bins_per_turn], self._accumulated_kick[self._n_bins_per_turn:])
        np.copyto(self._accumulated_kick[-1*self._n_bins_per_turn:], np.zeros(self._n_bins_per_turn))

        # Acculumates kick data from the previous turn data and new kicks caused from each circulating bunch
        for i, wake in enumerate(self._dashed_wake_functions):

            if moments == 'zero':
                moment = all_slice_sets[i].n_macroparticles_per_slice
            elif moments == 'mean_x':
                moment = all_slice_sets[i].mean_x*all_slice_sets[i].n_macroparticles_per_slice
            elif moments == 'mean_y':
                moment = all_slice_sets[i].mean_y*all_slice_sets[i].n_macroparticles_per_slice
            else:
                raise ValueError("Please specify moments as either " +
                                 "'zero', 'mean_x' or 'mean_y'!")

            np.copyto(self._accumulated_kick, self._accumulated_kick+np.convolve(wake, moment, 'same'))

        kick_list = []
        for i, value in enumerate(self._accumulated_signal_list):
            kick_list.append(value*self._wake_factor(bunch_list[i]))

        return kick_list

    def _init_memory_optimized(self, all_slice_sets, local_slice_sets, bunch_list,
                                local_bunch_indexes, circumference, h_rf, h_bunch):

        bunch_spacing = circumference/float(h_bunch)
        every_n_bucket_fillted = int(h_rf/h_bunch)

        # total number of bunches
        self._n_target_bunches = len(local_bunch_indexes)

        # number of slices per bunch
        n_slices = len(all_slice_sets[0].mean_x)

        # number of extra slices added to each side of the bunch
        empty_space_per_side = int(math.ceil(n_slices/2.))

        # total number of bins per bunch
        self._n_bins_per_kick = (n_slices + 2*empty_space_per_side)
        # total number of bins per turn

        self._n_bins_per_turn = self._n_target_bunches * self._n_bins_per_kick

        # determines normalized zbins for a wake function of a bunch
        raw_z_bins = local_slice_sets[0].z_bins
        raw_z_bins = raw_z_bins - ((raw_z_bins[0]+raw_z_bins[-1])/2.)
        bin_width = np.mean(raw_z_bins[1:]-raw_z_bins[:-1])
        original_z_bin_mids = (raw_z_bins[1:]+raw_z_bins[:-1])/2.
        z_bin_mids = original_z_bin_mids[0] - np.linspace(empty_space_per_side, 1,
                                        empty_space_per_side)*bin_width
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids)
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids[-1] + np.linspace(1,
                               empty_space_per_side, empty_space_per_side)*bin_width)

        self._wake = np.zeros(self._n_bins_per_turn*self.n_turns_wake)
        self._accumulated_kick = np.zeros(self._n_bins_per_turn*self.n_turns_wake)
        self._accumulated_signal_list = []
        self._idx_data = [] #

        self._wake_database = [] # A database of wake function values for all bunches in all turns

        bunch_mids = []
        for slice_set in all_slice_sets:
            bunch_mids.append(((slice_set.z_bins[0]+slice_set.z_bins[-1])/2.))

        for i, slice_set in enumerate(all_slice_sets):
            # loop of target bunches
            self._idx_data.append([])
            for j,target_bunch_idx in enumerate(local_bunch_indexes):

                source_id = slice_set.bunch_id
                target_id = all_slice_sets[target_bunch_idx].bunch_id

                # Calculates the distance difference between the source and the target bunches
                delta_id = target_id - source_id
                delta_id = delta_id/every_n_bucket_fillted
                if delta_id < 0:
                    # the target bunch must be after the source bunch
                    delta_id = delta_id + h_bunch
                self._idx_data[i].append(int(delta_id))

                if i == 0:
                    kick_from = empty_space_per_side + j * self._n_bins_per_kick
                    kick_to = empty_space_per_side + j * self._n_bins_per_kick + n_slices
                    self._accumulated_signal_list.append(np.array(self._accumulated_kick[kick_from:kick_to], copy=False))



        for k in xrange(self.n_turns_wake):
            self._wake_database.append([None]*h_bunch)
#            print 'len(self._wake_database[k]): ' + str(len(self._wake_database[k]))
            for i in np.unique(np.concatenate(self._idx_data)):
                i = int(i)
                offset = (float(k) * circumference + i*bunch_spacing)

#                if (i==0) and (k==0):
#                    wake = np.zeros(len(z_bin_mids))
#                    pos_values = (z_bin_mids>=0)
#                    z_values = z_bin_mids[pos_values]+offset
#                    wake[pos_values] = self.wake_function(-z_values/c, beta=local_slice_sets[0].beta)
#                    self._wake_database[k][i] = wake
#                else:
                z_values = -z_bin_mids+offset
                self._wake_database[k][i] = self.wake_function(-z_values/c, beta=local_slice_sets[0].beta)

            self._wake_database[k] = np.array(self._wake_database[k])

    def _accumulate_memory_optimized(self, all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments, h_rf, h_bunch):

        if not hasattr(self,'_wake_database'):
            self. _init_memory_optimized(all_slice_sets, local_slice_sets,
                                         bunch_list, local_bunch_indexes,
                                         circumference, h_rf, h_bunch)

        # Copies the perivous turn data forward
        np.copyto(self._accumulated_kick[:-1*self._n_bins_per_turn], self._accumulated_kick[self._n_bins_per_turn:])
        np.copyto(self._accumulated_kick[-1*self._n_bins_per_turn:], np.zeros(self._n_bins_per_turn))

        # Acculumates kick data from the previous turn data and new kicks caused from each circulating bunch
        for i, slice_set in enumerate(all_slice_sets):

            if moments == 'zero':
                moment = slice_set.n_macroparticles_per_slice
            elif moments == 'mean_x':
                moment = slice_set.mean_x*slice_set.n_macroparticles_per_slice
            elif moments == 'mean_y':
                moment = slice_set.mean_y*slice_set.n_macroparticles_per_slice
            else:
                raise ValueError("Please specify moments as either " +
                                 "'zero', 'mean_x' or 'mean_y'!")
            for k in xrange(self.n_turns_wake):
                i_from = k * self._n_bins_per_kick*self._n_target_bunches
                i_to = (k + 1) * self._n_bins_per_kick*self._n_target_bunches
                np.copyto(self._wake[i_from:i_to], np.concatenate(self._wake_database[k][self._idx_data[i]]))
            np.copyto(self._accumulated_kick, self._accumulated_kick+np.convolve(self._wake, moment, 'same'))
#            np.copyto(self._accumulated_kick, self._accumulated_kick+fftconvolve(self._wake, moment[::-1], 'same'))

        kick_list = []
        for i, value in enumerate(self._accumulated_signal_list):
            kick_list.append(value*self._wake_factor(bunch_list[i]))

        return kick_list


    def _init_full_ring_fft(self, all_slice_sets, local_slice_sets, bunch_list,
                                local_bunch_indexes, circumference, h_rf, h_bunch):

        bunch_spacing = circumference/float(h_bunch)

        # total number of bunches
        self._n_target_bunches = len(local_bunch_indexes)

        # number of slices per bunch
        n_slices = len(all_slice_sets[0].mean_x)

        # number of extra slices added to each side of the bunch
        empty_space_per_side = int(math.ceil(n_slices/2.))

        # total number of bins per bunch
        self._n_bins_per_kick = (n_slices + 2*empty_space_per_side)
        # total number of bins per turn
        n_bins_per_turn = h_bunch * self._n_bins_per_kick

        # determines normalized zbins for a wake function of a bunch
        raw_z_bins = local_slice_sets[0].z_bins
        raw_z_bins = raw_z_bins - ((raw_z_bins[0]+raw_z_bins[-1])/2.)
        bin_width = np.mean(raw_z_bins[1:]-raw_z_bins[:-1])
        original_z_bin_mids = (raw_z_bins[1:]+raw_z_bins[:-1])/2.
        z_bin_mids = original_z_bin_mids[0] - np.linspace(empty_space_per_side, 1,
                                        empty_space_per_side)*bin_width
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids)
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids[-1] + np.linspace(1,
                               empty_space_per_side, empty_space_per_side)*bin_width)

        self._accumulated_signal_list = [] # Standard lick list to the kick objects
        self._kick_data = [] #
        self._idx_data = [] #
        self._accumulated_data = [] # Raw turn by turn data from the convolutions

        self._dashed_wake_functions = [] # Turn by turn wake functions for the convolution
        self._moment = np.zeros(n_bins_per_turn)

        # calculates the mid points of the bunches from the z_bins
        # the bunch_id could be used here
        bunch_mids = []
        for slice_set in all_slice_sets:
            bunch_mids.append(((slice_set.z_bins[0]+slice_set.z_bins[-1])/2.))


        z_values = np.zeros(n_bins_per_turn)
        for i in xrange(h_bunch):
            idx_from = i * self._n_bins_per_kick
            idx_to = (i + 1) * self._n_bins_per_kick

            offset = (i*bunch_spacing)

            temp_mids = z_bin_mids+offset

            np.copyto(z_values[idx_from:idx_to],temp_mids)


        for k in xrange(self.n_turns_wake):
            self._accumulated_data.append(np.zeros(n_bins_per_turn))
            turn_offset = (float(k) * circumference)
            temp_z = np.copy(z_values)
            if k==0:
                np.copyto(temp_z[:-int(self._n_bins_per_kick/2)],z_values[int(self._n_bins_per_kick/2):])
                np.copyto(temp_z[-int(self._n_bins_per_kick/2):],np.zeros(len(temp_mids)/2))
            else:
                np.copyto(temp_z[:-int(self._n_bins_per_kick/2)],z_values[int(self._n_bins_per_kick/2):])
                np.copyto(temp_z[-int(self._n_bins_per_kick/2):],z_values[:int(self._n_bins_per_kick/2)])

            self._dashed_wake_functions.append(self.wake_function(-(temp_z+turn_offset)/c, beta=local_slice_sets[0].beta))


        for j,local_bunch_idx in enumerate(local_bunch_indexes):
            local_mid = bunch_mids[local_bunch_idx]
            local_idx = -int(round(local_mid/bunch_spacing))
            kick_from = empty_space_per_side + local_idx * self._n_bins_per_kick
            kick_to = empty_space_per_side + local_idx * self._n_bins_per_kick + n_slices
            self._accumulated_signal_list.append(np.array(self._accumulated_data[0][kick_from:kick_to], copy=False))

        for j, bunch_mid in enumerate(bunch_mids):
            idx = -int(round(bunch_mid/bunch_spacing))
            kick_from = empty_space_per_side + idx * self._n_bins_per_kick
            kick_to = empty_space_per_side + idx * self._n_bins_per_kick + n_slices
            temp_idx_data = (kick_from, kick_to)
            self._idx_data.append(temp_idx_data)




    def _accumulate_full_fft_ring(self, all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments, h_rf, h_bunch):

        if not hasattr(self,'_dashed_wake_functions'):
            self. _init_full_ring_fft(all_slice_sets, local_slice_sets,
                                         bunch_list, local_bunch_indexes,
                                         circumference, h_rf, h_bunch)

        # processes moment data for the convolutions
        self._moment.fill(0.)
        for i  in xrange(len(all_slice_sets)):
            i_from = self._idx_data[i][0]
            i_to = self._idx_data[i][1]

            if moments == 'zero':
                moment = all_slice_sets[i].n_macroparticles_per_slice
            elif moments == 'mean_x':
                moment = all_slice_sets[i].mean_x*all_slice_sets[i].n_macroparticles_per_slice
            elif moments == 'mean_y':
                moment = all_slice_sets[i].mean_y*all_slice_sets[i].n_macroparticles_per_slice
            else:
                raise ValueError("Please specify moments as either " +
                                 "'zero', 'mean_x' or 'mean_y'!")
            # because of the historical reasons, moment data must be flipped
            np.copyto(self._moment[i_from:i_to],moment[::-1])
#            np.copyto(self._moment[i_from:i_to],moment)

        # calculates the convolutions and moves previous turn data one turn forward
        for k in xrange(self.n_turns_wake):
            if k < (self.n_turns_wake-1):
                np.copyto(self._accumulated_data[k], self._accumulated_data[k+1] + np.real(np.fft.ifft(np.fft.fft(self._dashed_wake_functions[k]) * np.fft.fft(self._moment))))
            else:
                np.copyto(self._accumulated_data[k], np.real(np.fft.ifft(np.fft.fft(self._dashed_wake_functions[k]) * np.fft.fft(self._moment))))

        # flips the accumulated kicks back to original order and
        # multiplies them by a wake factor
        kick_list = []
        for i, value in enumerate(self._accumulated_signal_list):
            kick_list.append(value[::-1]*self._wake_factor(bunch_list[i]))
#            real_values.append(value)*self._wake_factor(bunch_list[i]))

        return kick_list


    def _init_mpi_full_ring_fft(self, all_slice_sets, local_slice_sets, bunch_list,
                                local_bunch_indexes, circumference, h_rf, h_bunch):

        bunch_spacing = circumference/float(h_bunch)
        every_n_bucket_fillted = int(h_rf/h_bunch)

        # total number of bunches
        self._n_target_bunches = len(local_bunch_indexes)

        # number of slices per bunch
        n_slices = len(all_slice_sets[0].mean_x)

        # number of extra slices to be added to each side of the bunch
        empty_space_per_side = int(math.ceil(n_slices/2.))

        # total number of bins per bunch
        self._n_bins_per_kick = (n_slices + 2*empty_space_per_side)

        # total number of bins per turn
        self._n_bins_per_turn = h_bunch * self._n_bins_per_kick

        # a buffer array for moment data
        self._moment = np.zeros(self._n_bins_per_turn)

        # a buffer for wake data, which are gathered from all processors
        self._new_wake_data = np.zeros(self._n_bins_per_turn*self.n_turns_wake)

        # Raw accumulated data from turn by turn convolutions
        self._accumulated_data = np.zeros(self._n_bins_per_turn*self.n_turns_wake)

        # Splitted accumulated data for local bunches
        self._accumulated_signal_list = []
        for j,local_bunch_idx in enumerate(local_bunch_indexes):
            idx = all_slice_sets[local_bunch_idx].bunch_id/every_n_bucket_fillted
            kick_from = empty_space_per_side + idx * self._n_bins_per_kick
            kick_to = empty_space_per_side + idx * self._n_bins_per_kick + n_slices
            self._accumulated_signal_list.append(np.array(self._accumulated_data[kick_from:kick_to], copy=False))


        # A list of indexes, which indicate locations of the moment data from
        # individual slice sets in the total moment data array
        self._idx_data = []
        for j, slice_set in enumerate(all_slice_sets):
            idx = slice_set.bunch_id/every_n_bucket_fillted
            kick_from = empty_space_per_side + idx * self._n_bins_per_kick
            kick_to = empty_space_per_side + idx * self._n_bins_per_kick + n_slices
            temp_idx_data = (kick_from, kick_to)
            self._idx_data.append(temp_idx_data)

        # initializes an object for data sharing through mpi and splits wake
        # convolutions to processors
        self._mpi_array_share = mpi_data.MpiArrayShare()
        all_wake_turns = np.arange(self.n_turns_wake)
        my_wake_turns = mpi_data.my_tasks(all_wake_turns)

        if len(my_wake_turns) > 0:
            # if convulations are calculated in this processor
            # wake functions are initiliazed

            # a buffer for the data calculated in this processor
            self._my_data = np.zeros(self._n_bins_per_turn*len(my_wake_turns))

            # calculates normalized mid points z-bins, i.e. z-bins for a bunch
            # are bunch_id * bunch_spacing * z_bin_mids
            raw_z_bins = local_slice_sets[0].z_bins
            raw_z_bins = raw_z_bins - ((raw_z_bins[0]+raw_z_bins[-1])/2.)
            bin_width = np.mean(raw_z_bins[1:]-raw_z_bins[:-1])
            original_z_bin_mids = (raw_z_bins[1:]+raw_z_bins[:-1])/2.
            z_bin_mids = original_z_bin_mids[0] - np.linspace(empty_space_per_side, 1,
                                            empty_space_per_side)*bin_width
            z_bin_mids = np.append(z_bin_mids, original_z_bin_mids)
            z_bin_mids = np.append(z_bin_mids, original_z_bin_mids[-1] + np.linspace(1,
                                   empty_space_per_side, empty_space_per_side)*bin_width)

            # determines z-bins for the entire ring
            z_values = np.zeros(self._n_bins_per_turn)
            for i in xrange(h_bunch):
                idx_from = i * self._n_bins_per_kick
                idx_to = (i + 1) * self._n_bins_per_kick
                offset = (i*bunch_spacing)
                temp_mids = z_bin_mids+offset
                np.copyto(z_values[idx_from:idx_to],temp_mids)

            # for FFT convolution time of the array must start from zero.
            # Thus, rolls z-bins in order put negative z-bins to the end of the array
            n_roll = sum((z_bin_mids<0.))
            z_values = np.roll(z_values,-n_roll)

            # calculates wake function values for convolution for different turns
            self._dashed_wake_functions = []
            for k in my_wake_turns:
                turn_offset = (float(k) * circumference)
                temp_z = np.copy(z_values)
                if k==0:
                    np.copyto(temp_z[-n_roll:],np.zeros(n_roll))

                self._dashed_wake_functions.append(self.wake_function(-(temp_z+turn_offset)/c, beta=local_slice_sets[0].beta))

        else:
            # if convolutions are not calculated in this procecessors,
            # zero length arrays are initiliazed
            self._dashed_wake_functions = []
            self._my_data = np.array([])

    def _accumulate_mpi_full_ring_fft(self, all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments,
                                                 h_rf, h_bunch):

        if not hasattr(self,'_dashed_wake_functions'):
            self. _init_mpi_full_ring_fft(all_slice_sets, local_slice_sets,
                                         bunch_list, local_bunch_indexes,
                                         circumference, h_rf, h_bunch)

        # processes moment data for the convolutions
        self._moment.fill(0.)
        for i  in xrange(len(all_slice_sets)):
            i_from = self._idx_data[i][0]
            i_to = self._idx_data[i][1]

            if moments == 'zero':
                moment = all_slice_sets[i].n_macroparticles_per_slice
            elif moments == 'mean_x':
                moment = all_slice_sets[i].mean_x*all_slice_sets[i].n_macroparticles_per_slice
            elif moments == 'mean_y':
                moment = all_slice_sets[i].mean_y*all_slice_sets[i].n_macroparticles_per_slice
            else:
                raise ValueError("Please specify moments as either " +
                                 "'zero', 'mean_x' or 'mean_y'!")

            # because of the historical reasons, moment data must be flipped
            np.copyto(self._moment[i_from:i_to],moment[::-1])
#            np.copyto(self._moment[i_from:i_to],moment)

        # convolution calculations are distributed to different processors
        for i, wake in enumerate(self._dashed_wake_functions):
            i_from = i*self._n_bins_per_turn
            i_to = (i+1) * self._n_bins_per_turn
            np.copyto(self._my_data[i_from:i_to], np.real(np.fft.ifft(np.fft.fft(wake) * np.fft.fft(self._moment))))

        # gathers total wake data from all processors
        self._mpi_array_share.share(self._my_data, self._new_wake_data)

        # copies the old wake data
        old_data_from = self._n_bins_per_turn
        old_data_to = self.n_turns_wake*self._n_bins_per_turn
        old_data = np.append(self._accumulated_data[old_data_from:old_data_to],
                             np.zeros(self._n_bins_per_turn))

        # accumulates new wake data from the old and new data
        np.copyto(self._accumulated_data,
                  self._new_wake_data+old_data)


        # flips the accumulated kicks back to original order and
        # multiplies by the wake factor
        kick_list = []
        for i, value in enumerate(self._accumulated_signal_list):
            kick_list.append(value[::-1]*self._wake_factor(bunch_list[i]))
#            real_values.append(value)*self._wake_factor(bunch_list[i]))

        return kick_list


    def _accumulate_optimized(self, all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 moments = 'zero', h_rf=None, h_bunch=None):

        if optimization_method == 'loop_minimized':
            # This version mimizes the number of inner loops, by calculating kicks for all target bunches
            # in one go. This is done by calculating a convolution between one source bunch and a wake function
            # covering all target bunches in all turns. The size of the wake function is minimized by removing
            # the values from the empty spaces between the target bunches.
            #
            # Assumptions:
            #   -slicing identical for each bunch, but bunch spacing can vary arbitrarily
            #
            # Drawbacks:
            #   -requires a lot of memory, which limits this solution to ~100 bunches (for 100 slices per bunch)

            return  self._accumulate_loop_minimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments,
                                                 h_rf, h_bunch)
            pass
        elif optimization_method == 'memory_optimized':
            # Similar to the loop_minimized version, but wake functions for each source bunch are not
            # keep in memory, but they are reconstructed during accumulation from precalculated
            # wake functions by assuming constant bunch spacing over the ring. By using this,
            # the memory limitations of the previous solutions can be avoided
            #
            # Assumptions:
            #   - slicing identical for each bunch, but bunch spacing can vary arbitrarily
            #   - bunch spacing is an integer times the minimum bunch spacing
            #   (determined by the harmonic number of bunches, h_bunch)
            #
            # Drawbacks:
            #   - more assumtpions
            #   - a maximum number of simulated bunces is limited by the computing power
            #   (practical limit probably between 100-1000 bunches for 100 slices per bunch,
            #     depending on the number of processors available)

            return  self._accumulate_memory_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments,
                                                 h_rf, h_bunch)
            pass
        elif optimization_method == 'full_ring_fft':
            # Follows the idea of the previous solutions, but the convolution is calculated over
            # each (bunch) bucket in the accelerator (even if they are not filled). This allow
            # the use of the circular fft convolution (ifft(fft(moment)*fft(wake))), which
            # extremely fast. The computing time does not depend on the number of simulated
            # bunches, but this solution is practical only when the accelerator is small (<LHC)
            # or there are more than 50 bunches simulated (>=LHC)
            #
            # Assumptions:
            #   - slicing identical for each bunch, but bunch spacing can vary arbitrarily
            #   - bunch spacing is an integer times the minimum bunch spacing
            #   (determined by the harmonic number of bunches, h_bunch)
            #
            # Drawbacks:
            #   - more assumtpions
            #   - calculation time does not depend on the number of bunches, which prefers
            #   use of the memory_optimized version for a small number of bunches in large accelerators

            return  self._accumulate_full_fft_ring(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments,
                                                 h_rf, h_bunch)
        elif optimization_method == 'mpi_full_ring_fft':
            # Same as the 'full_ring_fft', but the wake calculations are parallelized by calculating
            # convolutions for different turns in different processors byt using MPI. One turn convultion
            # for 3564 buckets (100 slices per bucket), takes ~100-150 ms, so in principle a 10 turn
            # wake kick for the entire accelerator can be calculated below 200 ms, if over 10 processors
            # are available. However, there are sometimes problems performance issues with np.fft
            # with the MPI environment, which slows down this solution (numpy, HDF5, mpi4py, etc.
            # should be probably carefully compiled with all the optimization flags)
            #
            # Assumptions:
            #   - slicing identical for each bunch, but bunch spacing can vary arbitrarily
            #   - bunch spacing is an integer times the minimum bunch spacing
            #   (determined by the harmonic number of bunches, h_bunch)
            #
            # Drawbacks:
            #   - more assumtpions
            #   - calculation time does not depend on the number of bunches, which prefers
            #   use of the memory_optimized version for a small number of bunches in large accelerators

            return  self._accumulate_mpi_full_ring_fft(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference, moments,
                                                 h_rf, h_bunch)
        elif optimization_method == 'dummy':
            if not hasattr(self,'_dummy_values'):
                self._dummy_values = []
                n_slices = len(local_slice_sets[0].mean_x)
                for i in xrange(len(local_slice_sets)):
                    self._dummy_values.append(np.zeros(n_slices))

            return  self._dummy_values
        else:
            raise ValueError('Unknown optimization method')



# ==============================================================
# Below we are to put the implemetation of any order wake kicks.
# ==============================================================
class ConstantWakeKickX(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a constant wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        if optimization_method is None:
            constant_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets)
        else:
            constant_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += constant_kick[i].take(s_idx)


class ConstantWakeKickY(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a constant wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        if optimization_method is None:
            constant_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets)
        else:
            constant_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 h_rf=h_rf, h_bunch=h_bunch)
#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += constant_kick[i].take(s_idx)


class ConstantWakeKickZ(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a constant wake kick to bunch.dp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        if optimization_method is None:
            constant_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets)
        else:
            constant_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 h_rf=h_rf, h_bunch=h_bunch)
#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.dp[p_idx] += constant_kick[i].take(s_idx)


class DipoleWakeKickX(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a dipolar wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        if optimization_method is None:
            dipole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets, moments='mean_x')
        else:
            dipole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 moments='mean_x',
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += dipole_kick[i].take(s_idx)


class DipoleWakeKickXY(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a dipolar (cross term x-y) wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        if optimization_method is None:
            dipole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets, moments='mean_y')
        else:
            dipole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 moments='mean_y',
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += dipole_kick[i].take(s_idx)


class DipoleWakeKickY(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a dipolar wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        if optimization_method is None:
            dipole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets, moments='mean_y')
        else:
            dipole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 moments='mean_y',
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += dipole_kick[i].take(s_idx)


class DipoleWakeKickYX(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a dipolar (cross term y-x) wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        if optimization_method is None:
            dipole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets, moments='mean_x')
        else:
            dipole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 moments='mean_x',
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += dipole_kick[i].take(s_idx)


class QuadrupoleWakeKickX(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a quadrupolar wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        if optimization_method is None:
            quadrupole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets)
        else:
            quadrupole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.x.take(p_idx))


class QuadrupoleWakeKickXY(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a quadrupolar (cross term x-y) wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        if optimization_method is None:
            quadrupole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets)
        else:
            quadrupole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.xp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.y.take(p_idx))


class QuadrupoleWakeKickY(WakeKick):

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a quadrupolar wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """
        if optimization_method is None:
            quadrupole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets)
        else:
            quadrupole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.y.take(p_idx))


class QuadrupoleWakeKickYX(WakeKick):

     def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None,
              circumference=None, h_rf=None, h_bunch=None):
        """Calculates and applies a quadrupolar (cross term y-x) wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """
        if optimization_method is None:
            quadrupole_kick = self._accumulate_source_signal_multibunch(
                    bunch_list, all_slice_sets)
        else:
            quadrupole_kick = self._accumulate_optimized(all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, circumference,
                                                 h_rf=h_rf, h_bunch=h_bunch)

#        for i, (b, s) in enumerate(zip(bunch_list, local_slice_sets)):
        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = s.slice_index_of_particle.take(p_idx)
            b.yp[p_idx] += (quadrupole_kick[i].take(s_idx) * b.x.take(p_idx))
