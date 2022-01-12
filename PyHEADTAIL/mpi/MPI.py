import numpy as np
import copy


class COMM(object):

    def __init__(self):
        pass

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Allgatherv(self, input_buffer, output_list):
        try:
            output_list[0][:] = input_buffer[:] 
        except ValueError:
            print('ValueError, doing nothing')

    def Gatherv(self, local_data, mpi_output, root=0):
        try:
            mpi_output[0][:] = local_data[:] 
        except ValueError:
            print('ValueError, doing nothing')

    def Allreduce(self, input_buffer, output_list):
        try:
            output_list[0][:] = input_buffer[:]
        except IndexError:
            output_list[0] = input_buffer
        except ValueError:
            print('ValueError, doing nothing')

    def Bcast(self, all_data, root=0):
        pass

class MPI(object):

    COMM_WORLD = COMM()

    FLOAT = np.float32
    DOUBLE = np.float64
    INT8_T = np.int8
    INT16_T = np.int16
    INT32_T = np.int32
    INT64_T = np.int64
    UINT8_T = np.uint8
    UINT16_T = np.uint16
    UINT32_T = np.uint32
    UINT64_T = np.uint64


def my_rank():
    """Returns the rank index of this processors.

    Note that this is a slow function, and it is only recommended to use for
    initializing things.

    Returns
    -------
    int
        Rank of this processor
    """
    mpi_comm = MPI.COMM_WORLD
    return mpi_comm.Get_rank()


def num_procs():
    """ Returns the total number of processors (ranks).

    Note that this is a slow function, and it is only recommended to use for
    initializing things.

    Returns
    -------
    int
        Total number of processors in parallel
    """
    mpi_comm = MPI.COMM_WORLD
    return mpi_comm.Get_size()


def split_tasks(tasks):
    """ Splits a list of tasks to sublists, which correspond to the tasks for
    different processors.

    Note that this is a slow function, and it is only recommended to use for
    initializing things.

    Parameters
    ----------
    tasks : list
        A list of tasks for all processors

    Returns
    -------
    list
        A list of lists, which correspond to the tasks for
        different processors.
    """

    n_procs = num_procs()
    n_tasks = len(tasks)

    n_tasks_on_rank = [n_tasks//n_procs + 1 if i < n_tasks % n_procs else
                       n_tasks//n_procs + 0 for i in range(n_procs)]

    n_tasks_cumsum = np.insert(np.cumsum(n_tasks_on_rank), 0, 0)

    return [tasks[n_tasks_cumsum[i]:n_tasks_cumsum[i+1]] for i in range(n_procs)]


def my_tasks(tasks):
    """ Picks tasks for this processor.

    Note that this is a slow function, and it is only recommended to use for
    initializing things.

    Parameters
    ----------
    tasks : list
        A list of tasks for all processors

    Returns
    -------
    list
        A list of tasks to be executed in this processor.
    """

    idx = my_rank()
    splitted_tasks = split_tasks(tasks)
    return splitted_tasks[idx]


def share_numbers(my_number):
    """ Shares numbers with all processors.

    Note that this is a slow function, and it is only recommended to use for
    initializing things.

    Parameters
    ----------
    my_number : number
        A number to be shared

    Returns
    -------
    NumPy array
        Numbers from all processors
    """

    mpi_comm = MPI.COMM_WORLD
    n_procs = mpi_comm.Get_size()

    local_number = np.array([my_number])
    data_type = local_number.dtype

    number_array = np.zeros(n_procs, dtype=data_type)

    segment_sizes = np.ones(n_procs, dtype=np.int32)
    segment_offsets = np.zeros(n_procs, dtype=np.int32)
    segment_offsets[1:] = np.cumsum(segment_sizes)[:-1]

    mpi_input = [number_array,
                 segment_sizes,
                 segment_offsets,
                 numpy_type_to_mpi_type(data_type)
                 ]

    mpi_comm.Allgatherv(local_number, mpi_input)

    return number_array

def share_arrays(my_array):
    """ Shares array data with all processors.

    Note that this is a slow function, and it is only recommended to use for
    initializing things.

    Parameters
    ----------
    my_array : NumPy array
        A array to be shared

    Returns
    -------
    NumPy array
        Data from all processors
    """
    mpi_comm = MPI.COMM_WORLD
    n_procs = mpi_comm.Get_size()

    data_type = my_array.dtype

    segment_sizes = share_numbers(len(my_array))

    all_data = np.zeros(sum(segment_sizes), dtype=data_type)
    segment_offsets = np.zeros(n_procs)
    segment_offsets[1:] = np.cumsum(segment_sizes)[:-1]

    mpi_input = [all_data,
                 segment_sizes,
                 segment_offsets,
                 numpy_type_to_mpi_type(data_type)
                 ]

    mpi_comm.Allgatherv(my_array, mpi_input)

    return all_data


def share_array_lists(my_arrays):
    """ Shares a list arrays with all processors.

    Note that this is a slow function, and it is only recommended to use for
    initializing things.

    Parameters
    ----------
    my_arrays : NumPy array
        A list of arrays to be shared

    Returns
    -------
    list
        A list of all arrays from all processors
    """
    mpi_comm = MPI.COMM_WORLD
    n_procs = mpi_comm.Get_size()

    data_type = my_arrays[0].dtype

    local_segment_sizes = np.zeros(len(my_arrays), dtype=np.int32)
    local_total_length = 0
    for i, array in enumerate(my_arrays):
        local_segment_sizes[i] = len(array)
        local_total_length += len(array)

    segment_sizes_for_splitting = share_arrays(local_segment_sizes)
    segment_sizes_for_data_sharing = share_numbers(local_total_length)

    local_data = np.zeros(local_total_length, dtype=data_type)

    counter = 0
    for i, array in enumerate(my_arrays):
        np.copyto(local_data[counter:(counter + len(array))], array,
                  casting='unsafe')
        counter += len(array)

    all_data = np.zeros(sum(segment_sizes_for_data_sharing), dtype=data_type)

    segment_offsets = np.zeros(n_procs)
    segment_offsets[1:] = np.cumsum(segment_sizes_for_data_sharing)[:-1]

    mpi_input = [all_data,
                 segment_sizes_for_data_sharing,
                 segment_offsets,
                 numpy_type_to_mpi_type(data_type)
                 ]

    mpi_comm.Allgatherv(local_data, mpi_input)

    all_arrays = []

    counter = 0
    for segment_size in segment_sizes_for_splitting:
        all_arrays.append(all_data[counter:(counter+segment_size)])
        counter += segment_size

    return all_arrays


def mpi_type_to_numpy_type(data_type):
    """ Converts mpi4py data type to NumPy data type

        Parameters
        ----------
        data_type : mpi4py data type, e.g. MPI.FLOAT, MPI.INT32_T, etc

        Returns
        -------
        NumPy data type
    """
    if data_type == MPI.FLOAT:
        return np.float32
    elif data_type == MPI.DOUBLE:
        return np.float64
    elif data_type == MPI.INT8_T:
        return np.int8
    elif data_type == MPI.INT16_T:
        return np.int16
    elif data_type == MPI.INT32_T:
        return np.int32
    elif data_type == MPI.INT64_T:
        return np.int64
    elif data_type == MPI.UINT8_T:
        return np.uint8
    elif data_type == MPI.UINT16_T:
        return np.uint16
    elif data_type == MPI.UINT32_T:
        return np.uint32
    elif data_type == MPI.UINT64_T:
        return np.uint64
    else:
        raise ValueError('Unknown data type.')
    pass


def numpy_type_to_mpi_type(data_type):
    """ Converts NumPy data type to mpi4py data type

        Parameters
        ----------
        data_type : NumPy data type, e.g. np.int32, np.int64, np.float, etc

        Returns
        -------
        mpi4py data type
    """
    if data_type == np.float32:
        return MPI.FLOAT
    elif data_type == np.float64:
        return MPI.DOUBLE
    elif data_type == np.int8:
        return MPI.INT8_T
    elif data_type == np.int16:
        return MPI.INT16_T
    elif data_type == np.int32:
        return MPI.INT32_T
    elif data_type == np.int64:
        return MPI.INT64_T
    elif data_type == np.uint8:
        return MPI.UINT8_T
    elif data_type == np.uint16:
        return MPI.UINT16_T
    elif data_type == np.uint32:
        return MPI.UINT32_T
    elif data_type == np.uint64:
        return MPI.UINT64_T
    else:
        raise ValueError('Unknown data type.')


class MpiArrayShare(object):
    """ Shares a NumpyArray with other processors.
    """
    def __init__(self):

        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_size = self._mpi_comm.Get_size()

        self._numpy_type = None
        self._mpi_type = None

        self._segment_sizes = None
        self._segment_offsets = None
        self._required_data_length = None
        
    @property
    def segment_sizes(self):
        return self._segment_sizes
        
    @property
    def segment_offsets(self):
        return self._segment_offsets

    def _init_sharing(self, local_data, all_data):
        self._numpy_type = local_data.dtype
        self._mpi_type = numpy_type_to_mpi_type(self._numpy_type)

        local_segment_size = len(local_data)
        self._segment_sizes = share_numbers(local_segment_size)
        self._segment_offsets = np.zeros(self._mpi_size)
        self._segment_offsets[1:] = np.cumsum(self._segment_sizes)[:-1]

        self._required_data_length = np.sum(self._segment_sizes)

    def share(self, local_data, all_data):
        """ A method which is called, when data is shared

            Parameters
            ----------
            local_data : NumPy array
                Data which are sent to the all processors
            all_data : NumPy array
                An array where the all data from all processors are stored
        """
        if self._segment_sizes is None:
            self._init_sharing(local_data, all_data)

        if len(all_data) < self._required_data_length:
            all_data = np.zeros(self._required_data_length)

        mpi_input = [all_data,
                     self._segment_sizes,
                     self._segment_offsets,
                     self._mpi_type
                     ]

        self._mpi_comm.Allgatherv(local_data, mpi_input)
        return all_data

class MpiArrayGather(object):
    """ Gathers a NumpyArray to one processor.
    """
    def __init__(self, root_rank=0):

        self._root_rank = root_rank
        
        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_size = self._mpi_comm.Get_size()
        self._my_rank = self._mpi_comm.Get_rank()

        self._numpy_type = None
        self._mpi_type = None

        self._segment_sizes = None
        self._segment_offsets = None
        
    @property
    def segment_sizes(self):
        return self._segment_sizes
        
    @property
    def segment_offsets(self):
        return self._segment_offsets

    def _init_sharing(self, local_data, all_data):
        self._numpy_type = local_data.dtype
        self._mpi_type = numpy_type_to_mpi_type(self._numpy_type)

        local_segment_size = len(local_data)
        self._segment_sizes = share_numbers(local_segment_size)
        self._segment_offsets = np.zeros(self._mpi_size)
        self._segment_offsets[1:] = np.cumsum(self._segment_sizes)[:-1]
        
        if self._root_rank == self._my_rank:
            self._required_data_length = np.sum(self._segment_sizes)
        else:
            self._required_data_length = 0

    def gather(self, local_data, all_data):
        """ A method which is called, when data is shared

            Parameters
            ----------
            local_data : NumPy array
                Data which are sent to the all processors
            all_data : NumPy array
                An array where the all data from all processors are stored
        """
        if self._segment_sizes is None:
            self._init_sharing(local_data, all_data)

        if len(all_data) != self._required_data_length: 
            all_data = np.zeros(self._required_data_length)

        mpi_input = [all_data,
                     self._segment_sizes,
                     self._segment_offsets,
                     self._mpi_type
                     ]

        self._mpi_comm.Gatherv(local_data, mpi_input, root = self._root_rank)
        
        return all_data

class MpiArrayBroadcast(object):
    """ Gathers a NumpyArray to one processor.
    """
    def __init__(self, root_rank=0):

        self._root_rank = root_rank
        
        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_size = self._mpi_comm.Get_size()
        self._my_rank = self._mpi_comm.Get_rank()

        self._numpy_type = None
        self._mpi_type = None

        self._segment_sizes = None
        self._segment_offsets = None
        
    @property
    def segment_sizes(self):
        return self._segment_sizes
        
    @property
    def segment_offsets(self):
        return self._segment_offsets

    def _init_sharing(self, local_data, all_data):
        self._numpy_type = local_data.dtype
        self._mpi_type = numpy_type_to_mpi_type(self._numpy_type)

        if self._root_rank == self._my_rank:
            local_segment_size = len(local_data)
        else:
            local_segment_size = 0
        self._segment_sizes = share_numbers(local_segment_size)
        self._segment_offsets = np.zeros(self._mpi_size)
        self._segment_offsets[1:] = np.cumsum(self._segment_sizes)[:-1]

        self._required_data_length = np.sum(self._segment_sizes)


    def broadcast(self, all_data):
        """ A method which is called, when data is shared

            Parameters
            ----------
            local_data : NumPy array
                Data which are sent to the all processors
            all_data : NumPy array
                An array where the all data from all processors are stored
        """
        if self._segment_sizes is None:
            self._init_sharing(all_data, all_data)

        if len(all_data) != self._required_data_length: 
            all_data = np.zeros(self._required_data_length)

        self._mpi_comm.Bcast(all_data, root = self._root_rank)
        
        return all_data

class MpiSniffer(object):
    """ Sniffer object, which can be used for getting information about
        the number of processors and the rank of the processors. The
        separated object has been implemented, because it might offer
        a way to avoid importing mpi4py without a real need.
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
    """An object, which gathers slice data from other processors. The
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
    of the first bunch in the filling scheme.
    """

    def __init__(self, slicer, required_variables):
        """
        Parameters
        ----------
        slicer : PyHEADTAIL slicer object
            A list index of the bunch
        required_variables : list
            A list of PyHEADTAIL slice_set variable names, which data
            are shared between the processors.
        """

        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_size = self._mpi_comm.Get_size()
        self._mpi_rank = self._mpi_comm.Get_rank()

        # total number of bunches simulated in all processors
        self._n_bunches = None

        # number of bunches simulated in this rank
        self._n_local_bunches = None

        # list indexes for the bunches simulated in this rank
        self._local_bunch_indexes = None

        # lists of bunch and slice set object in this rank
        self.bunch_list = None
        self.slice_set_list = None

        self._n_slices = None
        
        # accelerator parameters
        self._circumference = None
        self._h_bunch = None

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

        self._id_list = None
        self._local_id_list = None

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
        self.bunch_list = superbunch.split_to_views()

        self.slice_set_list = []
        if self._slicer.config[3] is not None:
            # In this case, we need to bring bunches back to zero
            for i, b in enumerate(self.bunch_list):
                
#                For particle by particle comparison to the original code,
#                the following lines must be uncommented
#
#                z_delay = b.mean_z()
#                b.z -= z_delay
                s = b.get_slices(self._slicer,
                                 statistics=self._statistical_variables)
#                b.z += z_delay
#                s.z_bins += z_delay
##                 Correct back mean_z which had delay removed
#                if 'mean_z' in self._statistical_variables:
#                    s.mean_z += z_delay

                self.slice_set_list.append(s)
        else:
            for i, b in enumerate(self.bunch_list):
                s = b.get_slices(self._slicer,
                                 statistics=self._statistical_variables)
                self.slice_set_list.append(s)

        if self.total_data is None:
            self._initialize_buffers(superbunch, self.slice_set_list)

        self._fill_output_buffer(superbunch, self.slice_set_list)

        self._mpi_comm.Allgatherv(
            self._output_buffer,
            [self._raw_data, self._slice_data_sizes,
             self._slice_data_offsets, MPI.DOUBLE])

        # updates total data object
        self.total_data.update()

    def rebunch(self, superbunch):
        pass
#        superbunch_new = sum(self.bunch_list)
#        superbunch.x[:] = superbunch_new.x[:]
#        superbunch.xp[:] = superbunch_new.xp[:]
#        superbunch.y[:] = superbunch_new.y[:]
#        superbunch.yp[:] = superbunch_new.yp[:]
#        superbunch.z[:] = superbunch_new.z[:]
#        superbunch.dp[:] = superbunch_new.dp[:]

    def _initialize_buffers(self, superbunch, slice_set_list):

        self._n_local_bunches = len(slice_set_list)
        self._n_slices = slice_set_list[0].n_slices
        
        self._circumference = slice_set_list[0].circumference
        self._h_bunch = slice_set_list[0].h_bunch

        # determines how many bunches per processor are simulated
        self._bunch_distribution = share_numbers(self._n_local_bunches)
        self._n_bunches = np.sum(self._bunch_distribution)

        # calculates list indexes for the local bunches
        self._local_bunch_indexes = []
        for i in range(self._n_local_bunches):
            idx = int(np.sum(self._bunch_distribution[:self.mpi_rank]) + i)
            self._local_bunch_indexes.append(idx)


        # determines ids (bucket numbers) for all the bunches
        self._local_id_list = list(set(superbunch.bucket_id))
        self._local_id_list = np.array(sorted(self._local_id_list, reverse=True), dtype=np.int32)
        self._id_list = share_arrays(self._local_id_list)

        # initializes buffers for the actual data sharing
        raw_data_len = int(self._n_variables * self._n_slices *
                           self._n_bunches)
        self._raw_data = np.zeros(raw_data_len)

        output_buffer_len = int(self._n_variables * self._n_slices *
                                self._n_local_bunches)
        self._output_buffer = np.zeros(output_buffer_len)

        values_per_bunch = int(self._n_variables * self._n_slices)

        self._slice_data_sizes = (np.ones(self._mpi_size) *
                                  self._bunch_distribution * values_per_bunch)
        self._slice_data_offsets = np.zeros(self._mpi_size)
        self._slice_data_offsets[1:] = np.cumsum(self._slice_data_sizes)[:-1]

        # gathers z_bin data from all the bunches
        bin_data_length = len(slice_set_list[0].z_bins)
        local_bin_data = np.zeros(self._n_local_bunches*bin_data_length)
        for i, slice_set in enumerate(slice_set_list):
            local_bin_data[i*bin_data_length:(i+1)*bin_data_length] = slice_set.z_bins

        self._raw_bin_data = share_arrays(local_bin_data)

        # creates objects for easier data acces for the shared data
        self.total_data = TotalDataAccess(self._local_bunch_indexes,
                                          self._raw_data,
                                          self._raw_bin_data,
                                          self._required_variables,
                                          self._n_slices,
                                          self._n_bunches,
                                          self._id_list,
                                          self._circumference,
                                          self._h_bunch)

        for idx in range(self._n_bunches):
            self.bunch_by_bunch_data.append(
                BunchDataAccess(idx, int(self._id_list[idx]),
                                self._raw_data,
                                self._raw_bin_data,
                                self._required_variables,
                                self._n_slices,
                                self._circumference,
                                self._h_bunch))

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


class BunchDataAccess(object):
    """An object, which emulates a PyHEADTAIL slice set object for shared data
    from an individual bunch. In the other words, the data can be accessed
    by using same methods as in the case of slice set object (e.g. by using
    bunchdataaccess.mean_x or bunchdataaccess.n_macroparticles_per_slice)
    """
    def __init__(self, bunch_idx, bucket_id, raw_data, raw_bin_data,
                 variables, n_slices, circumference, h_bunch):
        """
        Parameters
        ----------
        bunch_idx : int
            A list index of the bunch
        bucket_id : int
            A bucket index of the bunch
        raw_data : NumPy array
            Raw slice set data from mpi.Allgatherv(...), which includes all
            the data from the all bunches
        raw_bin_data : NumPy array
            z bin data from all bunches
        variables : list
            A list of slice set variable names, which data is shared between
            the bunches.
        n_slices : number
            A number of slices per bunch
        circumference : number
            A circumference of the simulated accelerator
        h_bunch : number
            A harmonic bunch number of the simulated accelerator, i.e.
            the maximum number bunches if every bucket is filled without gaps.

        """

        self._bunch_idx = bunch_idx
        self._variables = variables
        self._n_variables = len(variables)
        self._n_slices = n_slices
        self.bucket_id = bucket_id

        self.circumference = circumference
        self.h_bunch = h_bunch

        idx_from = self._bunch_idx * (self._n_slices + 1)
        idx_to = (self._bunch_idx+1) * (self._n_slices + 1)

        self.z_bins = np.array(raw_bin_data[idx_from:idx_to], copy=False)

        for idx, variable in enumerate(self._variables):
            idx_from = (self._bunch_idx * self._n_slices * self._n_variables +
                        idx*self._n_slices)
            idx_to = (self._bunch_idx * self._n_slices * self._n_variables +
                      (idx+1)*self._n_slices)
            setattr(self, variable,
                    np.array(raw_data[idx_from:idx_to], copy=False))


class TotalDataAccess(object):
    """An object, which emulates a PyHEADTAIL slice set object for data
    from all the bunches. In the other words, the data can be accessed
    by using same variable names as in the case of slice set object (e.g.
    by using bunchdataaccess.mean_x or bunchdataaccess.n_macroparticles_per_slice)
    and those variables includes data from all the bunches in a single array.
    """

    def __init__(self, local_bunches, raw_data, raw_bin_data,
                 variables, n_slices, n_bunches, id_list, circumference,
                 h_bunch):
        """
        Parameters
        ----------
        local_bunches : int

        bunch_id : int
            A bucket index of the bunch
        raw_data : NumPy array
            Raw slice set data from mpi.Allgatherv(...), which includes all
            the data from all the bunches
        raw_bin_data : NumPy array
            z bin data from all bunches
        variables : list
            A list of slice set variable names, which data is shared between
            the processors
        n_slices : number
            A number of slices per bunch
        n_bunches : int
            A total number of bunches in all the processors
        id_list : list
            A list of bucket indexes for all the bunches
        circumference : number
            A circumference of the simulated accelerator
        h_bunch : number
            A harmonic bunch number of the simulated accelerator, i.e.
            the maximum number bunches if every bucket is filled without gaps.
        """

        self.circumference = circumference
        self.h_bunch = h_bunch

        self._local_bunches = local_bunches
        self.local_data_locations = []
        self.z_bins = raw_bin_data
        self._raw_data = np.array(raw_data, copy=False)
        self._variables = variables
        self._n_variables = len(variables)
        self._n_slices = n_slices
        self._n_bunches = n_bunches
        self.bin_edges = []
        self.id_list = id_list

        # determines bin edges for all the bunches
        for i in range(n_bunches):
            if i in local_bunches:
                self.local_data_locations.append((i * self._n_slices,
                                                  (i + 1) * self._n_slices))
            idx_from = i * (self._n_slices + 1)
            idx_to = (i + 1) * (self._n_slices + 1)
            bin_set = np.array(self.z_bins[idx_from:idx_to], copy=False)
            for i, j in zip(bin_set, bin_set[1:]):
                self.bin_edges.append((i, j))

        self.bin_edges = np.array(self.bin_edges)

        self._data_length = int(self._n_slices * self._n_bunches)
        for idx, variable in enumerate(self._variables):
            setattr(self, variable, np.zeros(self._data_length))

    def update(self):

        for idx, variable in enumerate(self._variables):
            temp_data = np.zeros(self._data_length)
            for bunch_idx in range(self._n_bunches):
                local_from = bunch_idx*self._n_slices
                local_to = (bunch_idx+1)*self._n_slices
                buffer_from = (bunch_idx*self._n_slices*self._n_variables +
                               idx*self._n_slices)
                buffer_to = (bunch_idx*self._n_slices*self._n_variables +
                             (idx+1)*self._n_slices)

                np.copyto(temp_data[local_from:local_to],
                          self._raw_data[buffer_from:buffer_to])

            setattr(self, variable, temp_data)
