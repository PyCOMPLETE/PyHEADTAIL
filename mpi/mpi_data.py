import numpy as np
import copy
from mpi4py import MPI
from abc import ABCMeta, abstractmethod


# TODO: total slice data monitor map?


class MpiSniffer(object):
    """
        Sniffer object, which can be used for getting information about the number of processors and the rank of the
        processors. The separated object has been implemented, because it might offer a way to avoid importing mpi4py
        without a real need.
    """
    def __init__(self):
        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_size = self._mpi_comm.Get_size()
        self._mpi_rank = self._mpi_comm.Get_rank()

    @property
    def comm(self):
        return self._mpi_comm

    @property
    def size(self):
        return self._mpi_size

    @property
    def rank(self):
        return self._mpi_rank


class MpiGatherer(object):
    """MpiGather is an object, which gathers slice data from other processors. The
        data between processors are shared by using MPI message-passing
        system. The data can be accessed by calling the total_data or
        bunch_by_bunch_data objects. The object total_data includes slice set
        data from all bunches in all processors in one array, whereas the
        object bunch_by_bunch_data is a list of objects, which includes slice
        set data from individual bunches.  Those objects emulates orginal
        slice_set objects, data can be accessed by calling same variable names
        (e.g. mean_x, sigma_x and n_macroparticles_per_slice) than in the slice
        set object. For example, by calling total_data.mean_x returns a single
        array, which includes mean_x values from all slices in all bunches and
        bunch_by_bunch_data[0].mean_x returns mean_x values from the slice set
        of the first bunch in the filling scheme

    """

    def __init__(self, slicer, required_variables):
        """:param slicer: PyHEADTAIL slicer object

        :param required_variables: a list of variable names in a slice set
        object which are shared (e.g. n_macro_particles, mean_x, sigma_y, etc)

        """

        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_size = self._mpi_comm.Get_size()
        self._mpi_rank = self._mpi_comm.Get_rank()

        self._n_bunches = None  # total number of bunches simulated in all
                                # processors
        self._n_local_bunches = None  # number of bunches simulated in this
                                      # rank
        self._local_bunch_indexes = None  # list indexes for bunches simulated
                                          # in this rank

        # lists of bunch and slice set object in this rank
        self.bunch_list = None
        self.slice_set_list = None

        self._n_slices = None

        # the names of the variables in the slice set, which are shared between
        # processors
        self._required_variables = required_variables
        self._n_variables = len(self._required_variables)

        # the names of the statistical variables for the slice set
        self._statistical_variables = self._parse_statistics()

        self._slicer = slicer

        # data, which are shared
        self._output_buffer = None

        # a number of bunches in different processors
        self._bunch_distribution = None

        # raw data gathered from the processors
        self._raw_bin_data = None
        self._raw_data = None

        # data locations for mpi.gatherv(...)
        self._slice_data_sizes = None
        self._slice_data_offsets = None

        # easily accessible data data from all bunches
        self.bunch_by_bunch_data = []
        self.total_data = None

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def mpi_size(self):
        return self._mpi_size

    @property
    def mpi_comm(self):
        return self._mpi_comm

    @property
    def n_bunches(self):
        return self._n_bunches

    @property
    def n_local_bunches(self):
        return self._n_local_bunches

    @property
    def bunch_distribution(self):
        return self._bunch_distribution

    @property
    def local_bunch_indexes(self):
        return self._local_bunch_indexes

    def gather(self, superbunch):
        self.bunch_list = superbunch.split()

        self.slice_set_list = []
        if self._slicer.config[3] is not None:
            # In this case, we need to bring bunches back to zero
            for i, b in enumerate(self.bunch_list):
                z_delay = b.mean_z()
                b.z -= z_delay
                s = b.get_slices(self._slicer,
                                 statistics=self._statistical_variables)
                b.z += z_delay
                s.z_bins += z_delay
                # Correct back mean_z which had delay removed
                if 'mean_z' in self._statistical_variables:
                    s.mean_z += z_delay

                self.slice_set_list.append(s)
        else:
            for i, b in enumerate(self.bunch_list):
                s = b.get_slices(self._slicer,
                                 statistics=self._statistical_variables)
                self.slice_set_list.append(s)

        if self.total_data is None:
            self._initialize_buffers(superbunch, self.slice_set_list)

        self._fill_output_buffer(superbunch, self.slice_set_list)

        # another way to implement this could be to use separated
        # Allgatherv(...) calls for each statistical variable. It would allow
        # to use memoryviews also in TotalDataSet object, but multiple
        # Allgatherv(...)  calls might cost too much in time.
        self._mpi_comm.Allgatherv(
            self._output_buffer,
            [self._raw_data, self._slice_data_sizes,
             self._slice_data_offsets, MPI.DOUBLE])

        # updates total data object
        self.total_data.update()

    def rebunch(self, superbunch):
        superbunch_new = sum(self.bunch_list)
        superbunch.x[:] = superbunch_new.x[:]
        superbunch.xp[:] = superbunch_new.xp[:]
        superbunch.y[:] = superbunch_new.y[:]
        superbunch.yp[:] = superbunch_new.yp[:]
        superbunch.z[:] = superbunch_new.z[:]
        superbunch.dp[:] = superbunch_new.dp[:]

    def _initialize_buffers(self, superbunch, slice_set_list):

        # BUNCH DISTRIBUTION ON PROCESSORS ####################################
        #######################################################################
        self._n_local_bunches = len(slice_set_list)
        self._n_slices = slice_set_list[0].n_slices

        self._bunch_distribution = np.zeros(self.mpi_size, dtype=np.uint32)

        bunches_in_this_rank = np.zeros(1, dtype=np.uint32)
        bunches_in_this_rank[0] = self._n_local_bunches
        bunch_distribution_sizes = np.ones(self._mpi_size, dtype=np.uint32) * 1
        bunch_distribution_offsets = np.zeros(self._mpi_size)
        bunch_distribution_offsets[1:] = np.cumsum(bunch_distribution_sizes)[:-1]

        bunches_temp_list = [self._bunch_distribution,
                             bunch_distribution_sizes,
                             bunch_distribution_offsets, MPI.INT32_T]

        self._mpi_comm.Allgatherv(bunches_in_this_rank, bunches_temp_list)

        self._n_bunches = np.sum(self._bunch_distribution)

        self._local_bunch_indexes = []
        for i in xrange(self._n_local_bunches):
            idx = int(np.sum(self._bunch_distribution[:self.mpi_rank]) + i)
            self._local_bunch_indexes.append(idx)

        # BUFFER INITIALIZATION ###############################################
        #######################################################################
        # FIXME: a problem in variable types because int(...) is required
        raw_data_len = int(self._n_variables * self._n_slices *
                           self._n_bunches)
        self._raw_data = np.zeros(raw_data_len)

        # FIXME: a problem in variable types because int(...) is required
        output_buffer_len = int(self._n_variables * self._n_slices *
                                self._n_local_bunches)
        self._output_buffer = np.zeros(output_buffer_len)

        # FIXME: a problem in variable types because int(...) is required
        values_per_bunch = int(self._n_variables * self._n_slices)

        self._slice_data_sizes = (np.ones(self._mpi_size) *
                                  self._bunch_distribution * values_per_bunch)
        self._slice_data_offsets = np.zeros(self._mpi_size)
        self._slice_data_offsets[1:] = np.cumsum(self._slice_data_sizes)[:-1]

        # BIN DATA SET ########################################################
        #######################################################################
        bin_data_length = len(slice_set_list[0].z_bins)
        local_bin_data = np.zeros(self._n_local_bunches*bin_data_length)

        for i, slice_set in enumerate(slice_set_list):
            local_bin_data[i*bin_data_length:(i+1)*bin_data_length] = slice_set.z_bins

        # FIXME: a problem in variable types because int(...) is required
        raw_bin_data_len = int(bin_data_length * self._n_bunches)
        self._raw_bin_data = np.zeros(raw_bin_data_len)

        bin_data_sizes = (np.ones(self._mpi_size) *
                          self._bunch_distribution * bin_data_length)
        bin_data_offsets = np.zeros(self._mpi_size)
        bin_data_offsets[1:] = np.cumsum(bin_data_sizes)[:-1]

        self._mpi_comm.Allgatherv(
            local_bin_data,
            [self._raw_bin_data, bin_data_sizes, bin_data_offsets, MPI.DOUBLE])

        self.total_data = TotalDataAccess(self._local_bunch_indexes,
                                          self._raw_data,
                                          self._raw_bin_data,
                                          self._required_variables,
                                          self._n_slices,
                                          self._n_bunches)

        for idx in xrange(self._n_bunches):
            self.bunch_by_bunch_data.append(
                BunchDataAccess(idx,
                                self._raw_data,
                                self._raw_bin_data,
                                self._required_variables,
                                self._n_slices))

    def _fill_output_buffer(self, superbunch, slice_set_list):

        for set_idx, slice_set in enumerate(slice_set_list):
            for idx, variable in enumerate(self._required_variables):
                idx_from = (set_idx * self._n_slices * self._n_variables +
                            idx * self._n_slices)
                idx_to = (set_idx * self._n_slices * self._n_variables +
                          (idx+1) * self._n_slices)
                self._output_buffer[idx_from:idx_to] = getattr(slice_set, variable)

    def _parse_statistics(self):
        statistical_variables = copy.copy(self._required_variables)

        if 'n_macroparticles_per_slice' in statistical_variables:
            statistical_variables.remove('n_macroparticles_per_slice')

        return statistical_variables


# TODO: Do we need both BunchData and AllData objects?
class BunchDataAccess(object):
    """An object, which emulates a slice set object by creating pointers to the
        shared data of the specific bunch. In other words, the data can be
        accessed by using same methods than in the case of slice set object
        (e.g. by calling SliceDataSetReference.mean_x or
        SliceDataSetReference.n_macroparticles_per_slice)

    """
    def __init__(self, bunch_idx, raw_data, raw_bin_data,
                 variables, n_slices):
        """:param bunch_idx: a list index of the bunch

        :param raw_data: raw slice set data from mpi.Allgatherv(...)

        :param raw_bin_data: raw bin set data from mpi.Allgatherv(...)

        :param variables: a list of variable names in the slice set, which are
        shared between bunches

        :param n_slices: a number of slices per bunch

        """
        self._v = memoryview(raw_data)
        self._v_bin = memoryview(raw_bin_data)
        self._bunch_idx = bunch_idx
        self._variables = variables
        self._n_variables = len(variables)
        self._n_slices = n_slices

        idx_from = self._bunch_idx * (self._n_slices + 1)
        idx_to = (self._bunch_idx+1) * (self._n_slices + 1)

        self.z_bins = np.array(self._v_bin[idx_from:idx_to], copy=False)

        for idx, variable in enumerate(self._variables):
            idx_from = (self._bunch_idx * self._n_slices * self._n_variables +
                        idx*self._n_slices)
            idx_to = (self._bunch_idx * self._n_slices * self._n_variables +
                      (idx+1)*self._n_slices)

            exec ('self.' + variable +
                  ' = np.array(self._v[idx_from:idx_to],copy = False)')


class TotalDataAccess(object):
    """Produces an object, which contains data from all bunches in a single
        list. For example, by calling TotalDataSet.mean_x returns a single
        list, which contains mean_x values of all slices in all bunches

    """

    def __init__(self, local_bunches, raw_data, raw_bin_data,
                 variables, n_slices, n_bunches):
        """:param local_bunches: a list of list indexes for bunches in this processor

        :param raw_data: raw slice set data from mpi.Allgatherv(...)

        :param raw_bin_data: raw bin set data from mpi.Allgatherv(...)

        :param variables: a list of variable names in the slice set, which are
        shared between bunches :param n_slices: a number of slices per bunch

        :param n_bunches: a total number of bunches in the all processors

        """

        self._local_bunches = local_bunches
        self.local_data_locations = []
        self.z_bins = raw_bin_data
        self._v = memoryview(raw_data)
        self._variables = variables
        self._n_variables = len(variables)
        self._n_slices = n_slices
        self._n_bunches = n_bunches
        v_bin = memoryview(raw_bin_data)
        self.bin_edges = []
        
        # print 'n_bunches:'
        # print n_bunches

        # print 'n_slices:'
        # print n_slices

        # print 'len(raw_bin_data):'
        # print len(raw_bin_data)
        # print raw_bin_data

        # TODO: Different format for the bin set is used here. Is it fine?
        for i in xrange(n_bunches):
            if i in local_bunches:
                self.local_data_locations.append((i * self._n_slices,
                                                  (i + 1) * self._n_slices))

            idx_from = i * (self._n_slices + 1)
            idx_to = (i + 1) * (self._n_slices + 1)

            bin_set = np.array(v_bin[idx_from:idx_to], copy=False)

            for i, j in zip(bin_set, bin_set[1:]):
                self.bin_edges.append((i, j))

        self.bin_edges = np.array(self.bin_edges)

        # print 'self.local_data_locations:' + str(self.local_data_locations)
        # self.local_data_position =
        # (self._bunch_idx*self._n_slices,(self._bunch_idx+1)*self._n_slices)

        self._data_length = int(self._n_slices * self._n_bunches)
        for idx, variable in enumerate(self._variables):
            exec 'self.' + variable + ' = np.zeros(self._data_length)'

    def update(self):
        # This method might be too slow (it copies data). It can be optimized
        # after the decision about the data transfer API between processors

        for idx, variable in enumerate(self._variables):
            temp_data = np.zeros(self._data_length)
            for bunch_idx in xrange(self._n_bunches):
                local_from = bunch_idx*self._n_slices
                local_to = (bunch_idx+1)*self._n_slices
                buffer_from = (bunch_idx*self._n_slices*self._n_variables +
                               idx*self._n_slices)
                buffer_to = (bunch_idx*self._n_slices*self._n_variables +
                             (idx+1)*self._n_slices)

                temp_data[local_from:local_to] = self._v[buffer_from:buffer_to]

            setattr(self, variable, temp_data)
