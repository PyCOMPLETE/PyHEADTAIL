import numpy as np
import math
from scipy.constants import c
from PyHEADTAIL.mpi import mpi_data

# import matplotlib.pyplot as plt

def get_local_slice_sets(bunch, slicer):
    all_slice_sets = [bunch.get_slices(slicer, statistics=['mean_z','mean_x','mean_y'])]
    local_slice_sets = all_slice_sets
    bunch_list = [bunch]

    return all_slice_sets, local_slice_sets, bunch_list


def get_mpi_slice_sets(superbunch, mpi_gatherer):
    mpi_gatherer.gather(superbunch)
    all_slice_sets = mpi_gatherer.bunch_by_bunch_data
    local_slice_sets = mpi_gatherer.slice_set_list
    bunch_list = mpi_gatherer.bunch_list

    return all_slice_sets, local_slice_sets, bunch_list


class HackedDipoleWake(object):
    def __init__(self, data_t, data_x, data_y, slicer, n_turns_wakes, circumference, mpi=False):
        self._data_t = data_t # time column from the normal wake file [ns]
        self._data_x = data_x # dipole x column from the normal wake file
        self._data_y = data_y # dipole y column from the normal wake file
        self._slicer = slicer

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      ['mean_z','mean_x','mean_y',
                                                       'n_macroparticles_per_slice'])

        self._n_turns_wakes = n_turns_wakes
        self._circumference = circumference

        # a list indexes of the bunches
        self._local_bunch_indexes = None

        self._n_bins_per_turn = None

        # A list of interpolated values of the wake functions. The name "dashed" refers to the
        #
        self._dashed_wake_functions_x = None
        self._dashed_wake_functions_y = None

        # Accumulated total kicks. Each element in the array corresponds to the total kick caused
        # by the corresponding bunch
        self._accumulated_kick_x = None
        self._accumulated_kick_y = None
        self._temp_kick = None

        # 2D arrays of the references to the accumulated kicks, Each element of the array is a list,
        # which contains wake kicks from all bunches to one bunch, i.e. the total wake kick to the
        # bunch is a sum over the list.
        self._kick_lists_x = None
        self._kick_lists_y = None

    @staticmethod
    def _wake_factor(bunch):
        """ From PyHEADTAIL. Universal scaling factor for the strength of a wake field kick.
        """
        wake_factor = (-(bunch.charge)**2 /
                       (bunch.mass * bunch.gamma * (bunch.beta*c)**2) *
                       bunch.particlenumber_per_mp)
        return wake_factor


    def __init_variables(self, all_slice_sets, local_slice_sets, bunch_list,local_bunch_indexes):
        # total number of bunches
        n_target_bunches = len(local_bunch_indexes)
        n_source_bunches = len(all_slice_sets)

        # number of slices per bunch
        n_slices = len(all_slice_sets[0].mean_x)

        # correct convolution requires that the wake function continous at least half way outside
        # the bunch. Thus, extra bins are added to the wake functions in dashed_wake_functions

        # number of extra slices added to each side of the bunch
        empty_space_per_side = int(math.ceil(n_slices/2.))

        # total number of bins per bunch in dashed_wake_functions
        n_bins_per_kick = (n_slices + 2*empty_space_per_side)
        # total length of the dashed_wake_functions
        total_array_length = self._n_turns_wakes * n_target_bunches * n_bins_per_kick

        self._n_bins_per_turn = n_target_bunches * n_bins_per_kick

        # initializes the arrays of arrays

        self._temp_kick = np.zeros(total_array_length)


        self._dashed_wake_functions_x = []
        self._dashed_wake_functions_y = []

        self._accumulated_kick_x = []
        self._accumulated_kick_y = []
        self._kick_lists_x = []
        self._kick_lists_y = []

        for i in xrange(n_source_bunches):

            self._dashed_wake_functions_x.append(np.zeros(total_array_length))
            self._dashed_wake_functions_y.append(np.zeros(total_array_length))

            self._accumulated_kick_x.append(np.zeros(total_array_length))
            self._accumulated_kick_y.append(np.zeros(total_array_length))

            self._kick_lists_x.append([])
            self._kick_lists_y.append([])



        # calculates the mid points of the bunches from the z_bins
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


        for i, mid_i in enumerate(bunch_mids):
            z_values = np.zeros(total_array_length)
#            for j in xrange(len(bunch_mids)):
            for j,target_bunch_idx in enumerate(local_bunch_indexes):
                # index of the target bunch depends on index of the source bunch, because all
                # the bunches are added after the source bunch. In the case of low beta_beam wakes
                # this must be hacked
                #
                # Because the bunch spacing is always a multiplier of the minimum bunch spacing, 
		# memory usage could be reduced further by producing only one dashed_wake_function 
		# for all bunches, but it would cost in the computing time for the convolution.

                source_mid = mid_i
                target_mid = bunch_mids[target_bunch_idx]

                delta_mid = target_mid-source_mid
                if delta_mid < 0.:
                    # the target bunch is after the source bunch
                    delta_mid += self._circumference

                kick_from = empty_space_per_side + j * n_bins_per_kick
                kick_to = empty_space_per_side + j * n_bins_per_kick + n_slices

                self._kick_lists_x[target_bunch_idx].append(np.array(self._accumulated_kick_x[i][kick_from:kick_to], copy=False))
                self._kick_lists_y[target_bunch_idx].append(np.array(self._accumulated_kick_y[i][kick_from:kick_to], copy=False))

                # adds kicks over the multiple turns
                for k in xrange(self._n_turns_wakes):
                    idx_from = k * self._n_bins_per_turn + j * n_bins_per_kick
                    idx_to = k * self._n_bins_per_turn + (j + 1) * n_bins_per_kick

                    offset = (float(k) * self._circumference + delta_mid)
                    temp_mids = z_bin_mids+offset
                    np.copyto(z_values[idx_from:idx_to],temp_mids)

            # interpolates the wake functions for the corresponding source bunch
            convert_to_V_per_Cm = -1e15
            np.copyto(self._dashed_wake_functions_x[i],np.interp(z_values, self._data_t*1e-9*c, self._data_x*convert_to_V_per_Cm))
            np.copyto(self._dashed_wake_functions_y[i],np.interp(z_values, self._data_t*1e-9*c, self._data_y*convert_to_V_per_Cm))


##            for i, z_values in enumerate(self._messy_wake_z_values):
##                # print 'z_values: ' + str(z_values)
##                # print 'len(z_values): ' + str(len(z_values))
##                convert_to_V_per_Cm = -1e15
##                np.copyto(self._dashed_wake_functions_x[i],np.interp(z_values, self._data_t*1e-9*c, self._data_x*convert_to_V_per_Cm))
##                np.copyto(self._dashed_wake_functions_y[i],np.interp(z_values, self._data_t*1e-9*c, self._data_y*convert_to_V_per_Cm))
#
#            fig1 = plt.figure(figsize=(9, 5))
#            ax1 = fig1.add_subplot(111)
#            y1 = 0.8 * np.ones(len(original_z_bin_mids))
#            y2 = 1.2 * np.ones(len(z_bin_mids))
#            ax1.plot(original_z_bin_mids,y1, 'k.')
#            ax1.plot(z_bin_mids,y2, 'b.')
#            ax1.set_ylim(0,1.5)
#
#            fig2 = plt.figure(figsize=(9, 5))
#            ax2 = fig2.add_subplot(111)
#            y1 = 1.0 * np.ones(len(self._dashed_z_values[0]))
#            y2 = 0.9 * np.ones(len(self._dashed_z_values[1]))
#            y3 = 0.8 * np.ones(len(self._dashed_z_values[2]))
#            y4 = 0.7 * np.ones(len(self._dashed_z_values[3]))
#            y5 = 0.6 * np.ones(len(self._dashed_z_values[4]))
#            ax2.plot(self._dashed_z_values[0],y1, 'k.')
#            ax2.plot(self._dashed_z_values[1],y2, 'b.')
#            ax2.plot(self._dashed_z_values[2],y3, 'b.')
#            ax2.plot(self._dashed_z_values[3],y4, 'b.')
#            ax2.plot(self._dashed_z_values[4],y5, 'b.')
#            ax2.set_ylim(0,1.1)
#
#            fig3 = plt.figure(figsize=(9, 5))
#            ax3 = fig3.add_subplot(111)
#            y1 = self._dashed_wake_functions_x[0]
#            y2 = self._dashed_wake_functions_x[1]
#            ax3.plot(self._data_t*1e-9*c,self._data_x, 'r-')
#            ax3.plot(self._dashed_z_values[0],y1, 'k.')
#            ax3.plot(self._dashed_z_values[1],y2, 'b.')
#            ax3.set_ylim(min(y1), max(y1))
#            ax3.set_xlim(min(self._dashed_wake_functions_x[0]), max(self._dashed_wake_functions_x[0]))
#            plt.show()


    def __update_wakes(self, all_slice_sets, local_slice_sets, bunch_list):

        for i, wake in enumerate(self._dashed_wake_functions_x):
            self._temp_kick.fill(0.)
            # removes the previous turn from the old kick and copies it to the temp array
            np.copyto(self._temp_kick[:-1*self._n_bins_per_turn], self._accumulated_kick_x[i][self._n_bins_per_turn:])

            # the new accumulated kick is a sum of the convolution and the old accumulated
            # kick moved one turn forward
            moment = all_slice_sets[i].mean_x*all_slice_sets[i].n_macroparticles_per_slice
            np.copyto(self._accumulated_kick_x[i], np.convolve(wake, moment, 'same') + self._temp_kick)

        for i, wake in enumerate(self._dashed_wake_functions_y):
            self._temp_kick.fill(0.)
            np.copyto(self._temp_kick[:-1*self._n_bins_per_turn], self._accumulated_kick_y[i][self._n_bins_per_turn:])
            moment = all_slice_sets[i].mean_y*all_slice_sets[i].n_macroparticles_per_slice
            np.copyto(self._accumulated_kick_y[i], np.convolve(wake, moment, 'same') + self._temp_kick)

    def track(self, superbunch):
        if self._mpi:
            all_slice_sets, local_slice_sets, bunch_list \
            = get_mpi_slice_sets(superbunch, self._mpi_gatherer)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes
                self.__init_variables(all_slice_sets, local_slice_sets, bunch_list,
                                      self._local_bunch_indexes)

        else:
            all_slice_sets, local_slice_sets, bunch_list \
            = get_local_slice_sets(superbunch, self._slicer, self._required_variables)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = [0]
                self.__init_variables(all_slice_sets, local_slice_sets, bunch_list,
                                      self._local_bunch_indexes)


        self.__update_wakes(all_slice_sets, local_slice_sets, bunch_list)


        for slice_set, bunch_idx, bunch in zip(local_slice_sets,
                                               self._local_bunch_indexes, bunch_list):

            p_idx = slice_set.particles_within_cuts
            s_idx = slice_set.slice_index_of_particle.take(p_idx)

            # the total kick is a sum over the partial kicks in the kick list
            kick_x = self._wake_factor(bunch) * np.sum(self._kick_lists_x[bunch_idx], axis=0)
            kick_y = self._wake_factor(bunch) * np.sum(self._kick_lists_y[bunch_idx], axis=0)

            bunch.xp[p_idx] += np.take(kick_x, s_idx)
            bunch.yp[p_idx] += np.take(kick_y, s_idx)


        if self._mpi:
            self._mpi_gatherer.rebunch(superbunch)





