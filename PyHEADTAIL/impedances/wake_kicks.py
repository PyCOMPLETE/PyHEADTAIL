"""
@class WakeKick
@author Kevin Li, Michael Schenk, Jani Komppula
@date July 2014
@brief Implementation of the wake kicks, i.e. of the elementary objects
       describing the effects of a wake field.
@copyright CERN
"""



import numpy as np
from scipy.constants import c
from scipy.signal import fftconvolve

from abc import ABCMeta, abstractmethod

from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.general.element import Printing

from PyHEADTAIL.mpi import mpi_data

class WakeKick(Printing, metaclass=ABCMeta):
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

    def __init__(self, wake_function, slicer, n_turns_wake,
                 *args, **kwargs):
        """Universal constructor for WakeKick objects. The slicer_mode
        is passed only to decide about which of the two implementations
        of the convolution the self._convolution method is bound to.
        """
        self.slicer = slicer
        self.n_turns_wake = n_turns_wake
        self.wake_function = wake_function

        self.h_bunch = self.slicer.h_bunch
        self.circumference = self.slicer.circumference

        self._moments =  kwargs['moments']
        self._target_plane = kwargs['target_plane']

        if (slicer.mode == 'uniform_bin' and
                (n_turns_wake == 1 or slicer.z_cuts)):
            self._convolution = self._convolution_numpy
        else:
            self._convolution = self._convolution_dot_product
            if n_turns_wake > 1:
                self.warns(
                    'You use multiturn wakes. The stored '
                    'SliceSet instances from previous turns are converted '
                    'from z to dt according to the current turn\'s gamma. '
                    'If you accelerate this is only an approximation which '
                    'becomes invalid when gamma changes much over '
                    'n_turns_wake.')

    @abstractmethod
    def apply(self, bunch_list, all_slice_sets, local_slice_sets,
              local_bunch_indexes, optimization_method):
        """
        Calculates and applies the corresponding wake kick to the bunch conjugate momenta using the given slice_set.
        Only particles within the slicing region, i.e particles_within_cuts (defined by the slice_set) experience
        a kick.
        """
        pass

    @staticmethod
    def _wake_factor(bunch):
        """Universal scaling factor for the strength of a wake field
        kick.
        """
        wake_factor = (-(bunch.charge)**2 / (bunch.mass * bunch.gamma *
                       (bunch.beta * c)**2) * bunch.particlenumber_per_mp)
        return wake_factor

    def _convolution_dot_product(self, target_times, source_times,
                                 source_moments, source_beta):
        """Implementation of the convolution of wake and source_moments
        (beam profile) using the numpy dot product. To be used with the
        'uniform_charge' slicer mode.
        """
        dt_to_target_slice = (
            [target_times] - np.transpose([source_times]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)

        return np.dot(source_moments, wake)

    def _convolution_numpy(self, target_times, source_times,
                           source_moments, source_beta):
        """Implementation of the convolution of wake and source_moments
        (longitudinal beam profile) using the numpy built-in
        numpy.convolve method. Recommended use with the 'uniform_bin'
        slicer mode (in case of multiturn wakes, additional conditions
        must be fulfilled: fixed z_cuts and no acceleration!) for
        higher performance. Question: how about interpolation to avoid
        expensive dot product in most cases?
        """
        # Currently target_times/source_times are on the GPU --> np.concatenate
        # doesnt work. Temporary fix before checking if rewrite of
        # np.concatenate is required on GPU (if this is bottleneck), is to
        # get the arrays to the cpu via .get()
        try:
            target_times = target_times.get()
        except AttributeError:
            pass # is already on CPU
        try:
            source_times = source_times.get()
        except AttributeError:
            pass #is already on CPU
        dt_to_target_slice = np.concatenate(
            (target_times - source_times[-1],
            (target_times - source_times[0])[1:]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)
        return pm.convolve(source_moments, wake, 'valid')

    def _compute_kick(self, all_slice_sets, bunch_list, local_bunch_indexes, local_slice_sets,
                     optimization_method, moments):
        # if h_bunch is None:
        #     wake_kick = self._accumulate_source_signal(bunch_list, all_slice_sets, moments=moments)

        if optimization_method is None:
            wake_kick = self._accumulate_source_signal_multibunch(
                bunch_list, all_slice_sets, self.circumference, self.h_bunch, moments=moments)
        else:
            wake_kick = self._accumulate_optimized(all_slice_sets,
                                                       local_slice_sets,
                                                       bunch_list,
                                                       local_bunch_indexes,
                                                       optimization_method,
                                                       moments)
        return wake_kick

#     def _accumulate_source_signal(self, bunch_list, all_slice_sets, moments='zero'):
# # , bunch, times_list, ages_list,
# #                                   moments_list, betas_list):
#         """Accumulate (multiturn-)wake signals left by source slices.
#         Takes a list of slice set attributes and adds up all
#         convolutions weighted by the respective moments. Also updates
#         the age of each slice set.
#         """
#         print('\nThis is the old accumulate_source_signal function!!\n')
#
# #         ##########
# #         times_list = [s.convert_to_time(s.z_centers) for s in slice_set_list]
# #         betas_list = [s.beta for s in slice_set_list]
# #         moments_list = [s.n_macroparticles_per_slice
# # for s in slice_set_list]
# #         ############
#
# #         target_times = times_list[0]
# #         accumulated_signal = 0
#
# #         if len(ages_list) < self.n_turns_wake:
# #             n_turns = len(ages_list)
# #         else:
# #             n_turns = self.n_turns_wake
#
#         # # Source beta is not needed?!?!
#         # for i in range(n_turns):
#         #     source_times = times_list[i] + ages_list[i]
#         #     source_beta = betas_list[i]
#         #     source_moments = moments_list[i]
#         #     accumulated_signal += self._convolution(
#         #         target_times, source_times, source_moments, source_beta)
#
#         # accumulated_signal_list = np.atleast_1d(accumulated_signal)
#
#         return self._accumulate_source_signal_multibunch(
#             bunch_list, all_slice_sets, moments=moments,
#             circumference=0, h_bunch=1) # called explicitly here to enforce single bunch

    def _accumulate_source_signal_multibunch(
            self, bunches, slice_set_list, circumference, h_bunch, moments='zero'):
        """Args:

            bunches: bunch or list of bunches - the order is important; index 0
                is assumed to be the front most bunch i.e., the head of the b

            ages_list: list with delay in [s] for each slice set since
                 wake generation

            times_list, moments_list, betas_list: 2d array (turns x bunches)
                with history for each bunch

        """

        bunch_offset = []
        for b in bunches:
            bunch_offset.append(-b.bucket_id[0]*circumference/float(h_bunch))

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

        # Check for strictly descending order
        z_delays = [b.mean_z()+offset for b, offset in zip(bunches, bunch_offset)]
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
                    source_beta = betas_list[k, j, 0]
                    source_times = (times_list[k, j, :] + ages_list[k, j, :])
                    source_moments = moments_list[k, j, :]

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

    def _init_memory_optimized(self, all_slice_sets, local_slice_sets, bunch_list, local_bunch_indexes):

        circumference = local_slice_sets[0].circumference
        h_bunch = local_slice_sets[0].h_bunch

        bunch_spacing = circumference/float(h_bunch)

        # total number of bunches
        self._n_target_bunches = len(local_bunch_indexes)

        # number of slices per bunch
        n_slices = len(all_slice_sets[0].mean_x)

        # number of extra slices added to each side of the bunch
        empty_space_per_side = int(np.ceil(n_slices/2.))

        # total number of bins per bunch
        self._n_bins_per_kick = (n_slices + 2*empty_space_per_side)
        # total number of bins per turn
        empty_left = empty_space_per_side
        if n_slices%2 == 0:
            empty_right = empty_space_per_side + 1
            self._n_bins_per_kick += 1
        else:
            empty_right = empty_space_per_side

        self._n_bins_per_turn = self._n_target_bunches * self._n_bins_per_kick

        # determines normalized zbins for a wake function of a bunch
        raw_z_bins = local_slice_sets[0].z_bins
        raw_z_bins = raw_z_bins - ((raw_z_bins[0]+raw_z_bins[-1])/2.)
        bin_width = np.mean(raw_z_bins[1:]-raw_z_bins[:-1])
        original_z_bin_mids = (raw_z_bins[1:]+raw_z_bins[:-1])/2.
        z_bin_mids = original_z_bin_mids[0] - np.linspace(empty_left, 1,
                                        empty_left)*bin_width
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids)
        z_bin_mids = np.append(z_bin_mids, original_z_bin_mids[-1] + np.linspace(1,
                               empty_right, empty_right)*bin_width)
        if n_slices%2 == 0:
            z_bin_mids = z_bin_mids + bin_width/2.
        self._wake = np.zeros(self._n_bins_per_turn*self.n_turns_wake)
        self._accumulated_kick = np.zeros(self._n_bins_per_turn*self.n_turns_wake)
        self._accumulated_signal_list = []

        # Creates a database which contains the distance differeneces for all
        # the source and target bunch combinations
        self._idx_data = [] #
        for k in range(self.n_turns_wake):
            self._idx_data.append([])
            for i, slice_set in enumerate(all_slice_sets):
                # loop of target bunches
                self._idx_data[-1].append([])
                for j,target_bucket_idx in enumerate(local_bunch_indexes):

                    source_id = slice_set.bucket_id
                    target_id = all_slice_sets[target_bucket_idx].bucket_id

                    # Calculates the distance between the source and the target
                    # bunches in the units of harmonic bunch spacing
                    delta_id = target_id - source_id
                    delta_id = delta_id + k*(h_bunch)

                    if delta_id >= 0:
                        self._idx_data[k][i].append(int(delta_id))
                    else:
                        self._idx_data[k][i].append(-1)


                    # Generates memory views for all the target bunch wake kicks
                    if i == 0 and k == 0:
                        kick_from = empty_space_per_side + j * self._n_bins_per_kick
                        kick_to = empty_space_per_side + j * self._n_bins_per_kick + n_slices
                        self._accumulated_signal_list.append(np.array(self._accumulated_kick[kick_from:kick_to], copy=False))


        # Generates a wakedabase, whick contains all the wake function values for the calculations
        self._wake_database = [None]*(self.n_turns_wake*h_bunch+1)
        for k in range(self.n_turns_wake):

            idxs = np.concatenate(self._idx_data[k])

            for i in idxs:
                if i != -1:
                    offset = i*bunch_spacing
                    z_values = z_bin_mids+offset

                    self._wake_database[i] = self.wake_function(-z_values/c, beta=local_slice_sets[0].beta)
                else:
                        self._wake_database[-1] = np.zeros(len(z_bin_mids))

        self._wake_database = np.array(self._wake_database)

    def _accumulate_memory_optimized(self, all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, moments):
        if not hasattr(self,'_wake_database'):
            self. _init_memory_optimized(all_slice_sets, local_slice_sets,
                                         bunch_list, local_bunch_indexes)

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

            moment = moment[::-1]
            for k in range(self.n_turns_wake):
                i_from = k * self._n_bins_per_kick*self._n_target_bunches
                i_to = (k + 1) * self._n_bins_per_kick*self._n_target_bunches
                np.copyto(self._wake[i_from:i_to], np.concatenate(self._wake_database[self._idx_data[k][i]]))

            np.copyto(self._accumulated_kick, self._accumulated_kick+np.convolve(self._wake, moment, 'same'))

        # flips the accumulated kicks back to original order and
        # multiplies by the wake factor
        kick_list = []
        for i, value in enumerate(self._accumulated_signal_list):
            kick_list.append(value[::-1]*self._wake_factor(bunch_list[i]))

        return kick_list


    def _init_mpi_full_ring_fft(self, convolution, all_slice_sets, local_slice_sets,
                                bunch_list, local_bunch_indexes,
                                turns_on_this_proc, Q):
        # initializes an object for data sharing through mpi and splits wake
        # convolutions to processors
        self._mpi_array_gather = mpi_data.MpiArrayGather()
        self._mpi_array_broadcast = mpi_data.MpiArrayBroadcast()
        self._my_rank = mpi_data.my_rank()
        if turns_on_this_proc is None:
            all_wake_turns = np.arange(self.n_turns_wake)
            my_wake_turns = mpi_data.my_tasks(all_wake_turns)
        else:
            my_wake_turns = turns_on_this_proc

        circumference = local_slice_sets[0].circumference
        h_bunch = local_slice_sets[0].h_bunch

        bunch_spacing = circumference/float(h_bunch)

        # total number of bunches
        self._n_target_bunches = len(local_bunch_indexes)

        # number of slices per bunch
        n_slices = len(all_slice_sets[0].mean_x)

        # number of extra slices to be added to each side of the bunch
        empty_space_per_side = int(np.ceil(n_slices/2.))

        # total number of bins per bunch
        self._n_bins_per_kick = (n_slices + 2*empty_space_per_side)

        # total number of bins per turn
        self._n_bins_per_turn = h_bunch * self._n_bins_per_kick


        # a buffer array for moment data
        self._moment = np.zeros(self._n_bins_per_turn)

        # Raw accumulated data from turn by turn convolutions
        if convolution == 'linear':
            if self._my_rank == 0:
                self._accumulated_data = np.zeros(self._n_bins_per_turn*(self.n_turns_wake+1))
            else:
                self._accumulated_data = np.zeros(0)
            self._wake_kick_data = np.zeros(self._n_bins_per_turn)
            # a buffer for wake data, which are gathered from all processors
            self._new_real_wake_data = np.zeros(0)

        elif convolution == 'circular':

            if self._my_rank == 0:
                self._accumulated_data = np.zeros(self._n_bins_per_turn*self.n_turns_wake,dtype=complex)
            else:
                self._accumulated_data = np.zeros(0,dtype=complex)
            self._wake_kick_data = np.zeros(self._n_bins_per_turn,dtype=complex)
            self._wake_kick_data_real = np.zeros(self._n_bins_per_turn)
            self._wake_kick_data_imag = np.zeros(self._n_bins_per_turn)
            # a buffer for wake data, which are gathered from all processors
            self._new_real_wake_data = np.zeros(0)
            self._new_imag_wake_data = np.zeros(0)
        else:
            raise ValueError('Unknown convolution')

        # Splitted accumulated data for local bunches
        self._accumulated_signal_list = []
        for j,local_bucket_idx in enumerate(local_bunch_indexes):
            idx = int(all_slice_sets[local_bucket_idx].bucket_id)

            kick_from = empty_space_per_side + idx * self._n_bins_per_kick
            kick_to = empty_space_per_side + idx * self._n_bins_per_kick + n_slices

            self._accumulated_signal_list.append(np.array(self._wake_kick_data[kick_from:kick_to], copy=False))


        # A list of indexes, which indicate locations of the moment data from
        # individual slice sets in the total moment data array
        self._idx_data = []
        for j, slice_set in enumerate(all_slice_sets):
            idx = int(slice_set.bucket_id)
            kick_from = empty_space_per_side + idx * self._n_bins_per_kick
            kick_to = empty_space_per_side + idx * self._n_bins_per_kick + n_slices
            temp_idx_data = (kick_from, kick_to)
            self._idx_data.append(temp_idx_data)

        if len(my_wake_turns) > 0:
            # if convulations are calculated in this processor
            # wake functions are initiliazed

            # a buffer for the data calculated in this processor
            if convolution == 'linear':
                self._my_data = np.zeros(2*self._n_bins_per_turn*len(my_wake_turns))
            elif convolution == 'circular':
                self._my_data = np.zeros(self._n_bins_per_turn*len(my_wake_turns),dtype=complex)
            else:
                raise ValueError('Unknown convolution')

            # calculates normalized mid points z-bins, i.e. z-bins for a bunch
            # are bucket_id * bunch_spacing * z_bin_mids
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
            for i in range(h_bunch):
                idx_from = i * self._n_bins_per_kick
                idx_to = (i + 1) * self._n_bins_per_kick
                offset = (i*bunch_spacing)
                temp_mids = z_bin_mids+offset
                np.copyto(z_values[idx_from:idx_to],temp_mids)

            # FFT convolution requires that the values of z-bins start from
            # zero. Thus, rolls negative z-bins to the end of the array.
            roll_threshold = -1. * bin_width/2.
            n_roll = sum((z_bin_mids < roll_threshold))
            z_values[:n_roll] = z_values[:n_roll]+circumference
            z_values = np.roll(z_values,-n_roll)

            # sets z values to start from zero also with even number of slices
            z_values = z_values-z_values[0]

            # calculates wake function values for convolution for different turns
            self._dashed_wake_functions = []
            for k in my_wake_turns:
                turn_offset = (float(k) * circumference)
                temp_z = np.copy(z_values)
#                if k==0:
#                    np.copyto(temp_z[-n_roll:],np.zeros(n_roll))
#                np.copyto(temp_z[:n_roll],np.zeros(n_roll))

                if convolution == 'linear':
                    self._dashed_wake_functions.append(self.wake_function(-(temp_z+turn_offset)/c, beta=local_slice_sets[0].beta))
                elif convolution == 'circular':
                    rotation_angle = -2.*np.pi*(Q%1.)*z_values/circumference
                    self._dashed_wake_functions.append(np.exp(1j*rotation_angle)*self.wake_function(-(temp_z+turn_offset)/c, beta=local_slice_sets[0].beta))
                else:
                    raise ValueError('Unknown convolution')


        else:
            # if convolutions are not calculated in this procecessors,
            # zero length arrays are initiliazed
            self._dashed_wake_functions = []
            self._my_data = np.array([])

    def _calculate_field_mpi_full_ring_fft(self, convolution, all_slice_sets, local_slice_sets,
                                                 bunch_list, local_bunch_indexes,
                                                 optimization_method, moments,
                                                 turns_on_this_proc, Q, beta):
        if not hasattr(self,'_dashed_wake_functions'):
            self. _init_mpi_full_ring_fft(convolution, all_slice_sets, local_slice_sets,
                                         bunch_list, local_bunch_indexes,
                                         turns_on_this_proc, Q)

        # processes moment data for the convolutions
        self._moment.fill(0.)
        for i  in range(len(all_slice_sets)):
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


        # calculates distributed convolutions on each processor
        if convolution == 'linear':
            for i, wake in enumerate(self._dashed_wake_functions):
                i_from = 2*i*self._n_bins_per_turn
                i_to = 2*(i+1) * self._n_bins_per_turn-1
                np.copyto(self._my_data[i_from:i_to], fftconvolve(wake,self._moment,'full'))

        elif convolution == 'circular':

            for i, wake in enumerate(self._dashed_wake_functions):
                i_from = i*self._n_bins_per_turn
                i_to = (i+1) * self._n_bins_per_turn
                np.copyto(self._my_data[i_from:i_to], np.fft.ifft(np.fft.fft(wake) * np.fft.fft(self._moment+ 0*1j)))

            if len(self._dashed_wake_functions) > 0:
                self._my_data.imag = beta*self._my_data.imag
        else:
            raise ValueError('Unknown convolution')


    def _accumulate_mpi_full_ring_fft(self, convolution, all_slice_sets,local_slice_sets,
                                      bunch_list, local_bunch_indexes,
                                      optimization_method, moments):
        # Gathers all the calculated wake data in the method
        # _calculate_field_mpi_full_ring_fft() to the rank 0
        if convolution == 'linear':
            old_data_from = self._n_bins_per_turn
            old_data_to = (self.n_turns_wake+1)*self._n_bins_per_turn
            self._new_real_wake_data = self._mpi_array_gather.gather(np.copy(self._my_data), self._new_real_wake_data)

        elif convolution == 'circular':
            old_data_from = self._n_bins_per_turn
            old_data_to = self.n_turns_wake*self._n_bins_per_turn
            self._new_real_wake_data = self._mpi_array_gather.gather(np.copy(self._my_data.real), self._new_real_wake_data)
            self._new_imag_wake_data = self._mpi_array_gather.gather(np.copy(self._my_data.imag), self._new_imag_wake_data)
        else:
            raise ValueError('Unknown convolution')

        # combines the old wake data from previous turns with the new data on the rank 0
        if self._my_rank == 0:
            old_data = np.append(self._accumulated_data[old_data_from:old_data_to],
                                 np.zeros(self._n_bins_per_turn))



            # accumulates new wake data from the old and new data
            if convolution == 'linear':
                np.copyto(self._accumulated_data,old_data)
                for i in range(self.n_turns_wake):
                    j_from = i*self._n_bins_per_turn
                    j_to = (i+2)*self._n_bins_per_turn
                    k_from = 2*i*self._n_bins_per_turn
                    k_to = 2*(i+1)*self._n_bins_per_turn
                    self._accumulated_data[j_from:j_to] = self._accumulated_data[j_from:j_to] + self._new_real_wake_data[k_from:k_to]

            elif convolution == 'circular':
                np.copyto(self._accumulated_data,
                          self._new_real_wake_data + 1j*self._new_imag_wake_data+old_data)
            else:
                raise ValueError('Unknown convolution')



        # shares only the relevan wake kick data with all the processors
        if convolution == 'linear':
            if self._my_rank == 0:
                i_from = 0
                i_to = self._n_bins_per_turn
                np.copyto(self._wake_kick_data, self._accumulated_data[i_from:i_to])

            self._wake_kick_data = self._mpi_array_broadcast.broadcast(self._wake_kick_data)

        elif convolution == 'circular':
            if self._my_rank == 0:
                i_from = 0
                i_to = self._n_bins_per_turn
                np.copyto(self._wake_kick_data_real, self._accumulated_data[i_from:i_to].real)
                np.copyto(self._wake_kick_data_imag, self._accumulated_data[i_from:i_to].imag)


            self._wake_kick_data_real = self._mpi_array_broadcast.broadcast(self._wake_kick_data_real)
            self._wake_kick_data_imag = self._mpi_array_broadcast.broadcast(self._wake_kick_data_imag)

            np.copyto(self._wake_kick_data, self._wake_kick_data_real + 1j*self._wake_kick_data_imag)

        else:
            raise ValueError('Unknown convolution')

        # flips the accumulated kicks back to original order and
        # multiplies by the wake factor
        kick_list = []
        for i, value in enumerate(self._accumulated_signal_list):
            kick_list.append(value[::-1]*self._wake_factor(bunch_list[i]))
#            real_values.append(value)*self._wake_factor(bunch_list[i]))
        return kick_list


    def _accumulate_optimized(self, all_slice_sets, local_slice_sets,
                              bunch_list, local_bunch_indexes,
                              optimization_method, moments='zero'):

        import time
        t0  = time.perf_counter()

        if optimization_method == 'memory_optimized':
            # Similar to the loop_minimized version, but wake functions for each source bunch are not
            # keep in memory, but they are reconstructed during accumulation from precalculated
            # wake functions by assuming constant bunch spacing over the ring. By using this,
            # the memory limitations of the previous solution can be avoided
            #
            # Assumptions:
            #   - slicing identical for each bunch
            #   - bunch spacing is a multiple of the minimum bunch spacing
            #   (determined by the harmonic number of bunches, h_bunch)
            #
            # Drawbacks:
            #   - a maximum number of simulated bunces is limited by the computing power
            #   (practical limit probably between 100-1000 bunches for 100 slices per bunch,
            #    depending on the number of processors available)
            res = self._accumulate_memory_optimized(all_slice_sets, local_slice_sets,
                                                     bunch_list, local_bunch_indexes,
                                                     optimization_method, moments)
            t1 = time.perf_counter()
            print('using memory optimized')
            self.time_last_accumulate = t1-t0
            return res

        elif optimization_method == 'circular_mpi_full_ring_fft':
            # Follows the idea from the previous solutions, but the convolution is calculated over
            # each (bunch) bucket in the accelerator (even if they are not filled). This allow
            # the use of the circular fft convolution (ifft(fft(moment)*fft(wake))), which is
            # extremely fast. The wake calculations for different turns are parallelized by
            # using MPI.  The computing time does not depend on the number of simulated
            # bunches, but this solution is practical only when the accelerator is small (<LHC)
            # or there are more than 50 bunches simulated (>=LHC)
            #
            # Assumptions:
            #   - slicing identical for each bunch
            #   - bunch spacing is a multiple of the minimum bunch spacing
            #   (determined by the harmonic number of bunches, h_bunch)
            #
            # Drawbacks:
            #   - calculation time does not depend on the number of bunches, which prefers
            #   use of the memory_optimized version for a small number of bunches in large accelerators
            res = self._accumulate_mpi_full_ring_fft('circular', all_slice_sets, local_slice_sets,
                                                       bunch_list, local_bunch_indexes,
                                                       optimization_method, moments)
            t1 = time.perf_counter()
            self.time_last_accumulate = t1-t0
            return res

        elif optimization_method == 'linear_mpi_full_ring_fft':
            res = self._accumulate_mpi_full_ring_fft('linear',all_slice_sets, local_slice_sets,
                                                       bunch_list, local_bunch_indexes,
                                                       optimization_method, moments)
            t1 = time.perf_counter()
            print('Using linear mpi full ring fft')
            self.time_last_accumulate = t1-t0
            return res
        elif optimization_method == 'dummy':
            if not hasattr(self,'_dummy_values'):
                self._dummy_values = []
                n_slices = len(local_slice_sets[0].mean_x)
                for i in range(len(local_slice_sets)):
                    self._dummy_values.append(np.zeros(n_slices))
            t1 = time.perf_counter()
            self.time_last_accumulate = t1-t0
            return  self._dummy_values
        else:
            raise ValueError('Unknown optimization method')



    def calculate_field(self, all_slice_sets, local_slice_sets,
                              bunch_list, local_bunch_indexes,
                              optimization_method, turns_on_this_proc,
                              circular_conv_params):

        if optimization_method == 'memory_optimized':
            pass

        elif optimization_method == 'circular_mpi_full_ring_fft':
            if self._target_plane == 'x':
                Q = circular_conv_params['Q_x']
                beta = circular_conv_params['beta_x']
            elif self._target_plane == 'y':
                Q = circular_conv_params['Q_y']
                beta = circular_conv_params['beta_y']
            elif self._target_plane is None:
                Q = 0.
                beta = 1.
            else:
                raise ValueError('Unknown wake target plane')

            self._calculate_field_mpi_full_ring_fft('circular', all_slice_sets,
                                                    local_slice_sets,
                                                    bunch_list,
                                                    local_bunch_indexes,
                                                    optimization_method,
                                                    self._moments,
                                                    turns_on_this_proc,
                                                    Q, beta)
        elif optimization_method == 'linear_mpi_full_ring_fft':
            self._calculate_field_mpi_full_ring_fft('linear', all_slice_sets,
                                                    local_slice_sets,
                                                    bunch_list,
                                                    local_bunch_indexes,
                                                    optimization_method,
                                                    self._moments,
                                                    turns_on_this_proc,
                                                    None, None)
        elif optimization_method == 'dummy':
            pass
        else:
            raise ValueError('Unknown optimization method')

# ==============================================================
# Below we are to put the implemetation of any order wake kicks.
# ==============================================================
class ConstantWakeKickX(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='zero',
                                             target_plane='x',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a constant wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """

        constant_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                          local_slice_sets, optimization_method, self._moments)

        for i, bunch in enumerate(bunch_list):
            slices = bunch.get_slices(self.slicer)
            p_idx = slices.particles_within_cuts
            s_idx = pm.take(slices.slice_index_of_particle, p_idx)
            bunch.xp[p_idx] += pm.take(constant_kick[i].real, s_idx)
            bunch.x[p_idx] += pm.take(constant_kick[i].imag, s_idx)




class ConstantWakeKickY(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='zero',
                                             target_plane='y',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a constant wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """

        constant_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                           local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.yp[p_idx] += pm.take(constant_kick[i].real, s_idx)
            b.y[p_idx] += pm.take(constant_kick[i].imag, s_idx)


class ConstantWakeKickZ(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='zero',
                                             target_plane=None,**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a constant wake kick to bunch.dp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """

        constant_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                           local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.dp[p_idx] += pm.take(constant_kick[i], s_idx)



""" Dipolar wake kicks """

class DipoleWakeKickX(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='mean_x',
                                             target_plane='x',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a dipolar wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """

        dipole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                       local_slice_sets, optimization_method, self._moments)
        self._last_dipole_kick = dipole_kick

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.xp[p_idx] += pm.take(dipole_kick[i].real, s_idx)
            b.x[p_idx] += pm.take(dipole_kick[i].imag, s_idx)


class DipoleWakeKickXY(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='mean_y',
                                             target_plane='x',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a dipolar (cross term x-y) wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """

        dipole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                         local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.xp[p_idx] += pm.take(dipole_kick[i].real, s_idx)
            b.x[p_idx] += pm.take(dipole_kick[i].imag, s_idx)


class DipoleWakeKickY(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='mean_y',
                                             target_plane='y',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a dipolar wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """

        dipole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                         local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.yp[p_idx] += pm.take(dipole_kick[i].real, s_idx)
            b.y[p_idx] += pm.take(dipole_kick[i].imag, s_idx)


class DipoleWakeKickYX(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='mean_x',
                                             target_plane='y',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a dipolar (cross term y-x) wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """

        dipole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                         local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.yp[p_idx] += pm.take(dipole_kick[i].real, s_idx)
            b.y[p_idx] += pm.take(dipole_kick[i].imag, s_idx)


""" Quadrupolar wake kicks """

class QuadrupoleWakeKickX(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='zero',
                                             target_plane='x',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a quadrupolar wake kick to bunch.xp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """

        quadrupole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                         local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.xp[p_idx] += pm.take(quadrupole_kick[i].real, s_idx) * pm.take(b.x, p_idx)
            b.x[p_idx] += pm.take(quadrupole_kick[i].imag, s_idx) * pm.take(b.x, p_idx)


class QuadrupoleWakeKickXY(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='zero',
                                             target_plane='x',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a quadrupolar (cross term x-y) wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """

        quadrupole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                         local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.xp[p_idx] += pm.take(quadrupole_kick[i].real, s_idx) * pm.take(b.y, p_idx)
            b.x[p_idx] += pm.take(quadrupole_kick[i].imag, s_idx) * pm.take(b.y, p_idx)


class QuadrupoleWakeKickY(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='zero',
                                             target_plane='y',**kwargs)

    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
              local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a quadrupolar wake kick to bunch.yp using the given
        slice_set. Only particles within the slicing region, i.e
        particles_within_cuts (defined by the slice_set) experience the kick.

        """

        quadrupole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                         local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.yp[p_idx] += pm.take(quadrupole_kick[i].real, s_idx) * pm.take(b.y, p_idx)
            b.y[p_idx] += pm.take(quadrupole_kick[i].imag, s_idx) * pm.take(b.y, p_idx)


class QuadrupoleWakeKickYX(WakeKick):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(*args, moments='zero',
                                             target_plane='y',**kwargs)


    def apply(self, bunch_list, all_slice_sets, local_slice_sets=None,
               local_bunch_indexes=None, optimization_method=None):
        """Calculates and applies a quadrupolar (cross term y-x) wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing region,
        i.e particles_within_cuts (defined by the slice_set) experience the
        kick.

        """

        quadrupole_kick = self._compute_kick(all_slice_sets, bunch_list, local_bunch_indexes,
                                         local_slice_sets, optimization_method, self._moments)

        for i, b in enumerate(bunch_list):
            s = b.get_slices(self.slicer)
            p_idx = s.particles_within_cuts
            s_idx = pm.take(s.slice_index_of_particle, p_idx)
            b.yp[p_idx] += pm.take(quadrupole_kick[i].real, s_idx) * pm.take(b.x, p_idx)
            b.y[p_idx] += pm.take(quadrupole_kick[i].imag, s_idx) * pm.take(b.x, p_idx)
