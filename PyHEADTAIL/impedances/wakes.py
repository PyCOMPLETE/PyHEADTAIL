"""
This module includes the description of a (multiturn) WakeField as well
as the implementation of the WakeSource objects.

A WakeField is defined as a composition of the elementary WakeKick
objects (see .wake_kicks module). They originate from WakeSources,
e.g. a WakeTable, Resonator and/or a ResistiveWall. The WakeField does
not directly accept the WakeKick objects, but takes a list of
WakeSources first (can be of different kinds), each of which knows how
to generate its WakeKick objects via the factory method
WakeSource.get_wake_kicks(..). The collection of WakeKicks from all the
WakeSources define the WakeField and are the elementary objects that are
stored, (i.e. the WakeField forgets about the origin of the WakeKicks
once they have been created).

@author Hannes Bartosik, Kevin Li, Giovanni Rumolo, Michael Schenk
@date March 2014
@brief Implementation of a WakeField as a composition of WakeKicks
       originating from different WakeSources.
@copyright CERN
"""



import numpy as np
from collections import deque
from scipy.constants import c, physical_constants
from scipy.interpolate import interp1d
from abc import ABCMeta, abstractmethod

from PyHEADTAIL.impedances.wake_kicks import *
from PyHEADTAIL.general.element import Element, Printing
from PyHEADTAIL.general.decorators import deprecated
from PyHEADTAIL.mpi import mpi_data
from functools import reduce

sin = np.sin
cos = np.cos


def check_wake_sampling(bunch, slicer, wakes, beta=1, wake_column=None, bins=False):
    '''
    Handy function for quick visual check of sampling of the wake functions.
    For now only implemented for wake table type wakes.
    '''
    from scipy.constants import c
    import matplotlib.pyplot as plt

    ss = bunch.get_slices(slicer).z_centers
    zz = bunch.get_slices(slicer).z_bins
    ss = ss[:-1]
    ll = bunch.get_slices(slicer).lambda_z(ss, sigma=100)
    # ss = np.concatenate((s.z_centers-s.z_centers[-1], (s.z_centers-s.z_centers[0])[1:]))

    A = [wakes.wake_table['time'] * beta*c*1e-9, wakes.wake_table[wake_column] * 1e15]
    W = [ss[::-1], wakes.function_transverse(wake_column)(beta, ss)]


    fig, (ax1, ax2) = plt.subplots(2, figsize=(16,12), sharex=True)

    ax1.plot(ss, ll)

    ax2.plot(A[0], (A[1]), 'b-+', ms=12)
    ax2.plot(W[0][:-1], (-1*W[1][1:]), 'r-x')
    if bins:
        [ax2.axvline(z, color='g') for z in zz]

    ax2.grid()
    lgd = ['Table', 'Interpolated']
    if bins:
        lgd += ['Bin edges']
    ax2.legend(lgd)

    print('\n--> Resulting number of slices: {:g}'.format(len(ss)))

    return ax1


# ==============================================================================
# BASIC WAKE FIELD OBJECT
# ==============================================================================
class WakeField(Element):
    """WakeField object to collect all wake kicks and apply them upon track.

    A WakeField is defined by elementary WakeKick objects that may originate
    from different WakeSource objects. Usually, there is no need for the user
    to define more than one instance of the WakeField class in a simulation -
    except if one wants to use different slicing configurations (one WakeField
    object is allowed to have exactly one slicing configuration, i.e. only one
    instance of the Slicer class). A WakeField also is able to calculate the
    wake forces coming from earlier turns (multiturn wakes) by archiving the
    longitudinal bunch distribution (SliceSet instances) a number of turns
    back.

    """

    def __init__(self, slicer, wake_sources_list, mpi=None, **kwargs):
        """Stores slicer and wake sources list and manages wake history.

        Obtains a slicer and a wake sources list. Owns a slice set deque of
        maximum length of wake sources list wake.

        Parallel version obtains a communicator.

        At each turn accepts a list of WakeSource objects. Each WakeSource object knows
		how to generate its corresponding WakeKick objects. The collection of all
        the WakeKick objects of each of the passed WakeSource objects defines
        the WakeField. When instantiating the WakeField object, the WakeKick
        objects for each WakeSource defined in wake_sources are requested. The
        returned WakeKick lists are all stored in the WakeField.wake_kicks
        list. The WakeField itself forgets about the origin (WakeSource) of the
        kicks as soon as they have been generated.

        Exactly one instance of the Slicer class must be passed to the
        WakeField constructor. All the wake field components (kicks)
        hence use the same slicing and thus the same slice_set to
        calculate the strength of the kicks.
        To calculate the contributions from multiturn wakes, the
        longitudinal beam distributions (SliceSet instances) are
        archived in a deque. In parallel to the slice_set_deque,
        there is a slice_set_age_deque to keep track of the age of
        each of the SliceSet instances.

        Args:

            slicer: slicer object which will be used for all wake kicks
                (convolutions).

            wake_sources_list: list of wake sources which will be internally
                unfolded into a list of wake kicks.

        Kwargs:

            circumference: optional argument - must be provided for multi-turn
                wakes to update ages of slice sets.

        """

        self.slicer = slicer
        self.h_bunch = self.slicer.h_bunch
        self.circumference = self.slicer.circumference

        wake_sources_list = np.atleast_1d(wake_sources_list)
        self.wake_kicks = reduce(lambda x,y: x+y, [source.get_wake_kicks(self.slicer)
                           for source in wake_sources_list])

        # Prepare the slice sets register ([turns, bunches])
        n_turns_wake_max = max([source.n_turns_wake
                                for source in wake_sources_list])
        self.slice_set_deque = deque([], maxlen=n_turns_wake_max)
        # self.slice_set_age_deque = deque([], maxlen=n_turns_wake_max)

        if ((n_turns_wake_max > 1) or (mpi is not None)) and ((self.circumference is None) or (self.h_bunch is None)):
            raise ValueError(
                """ The circumference and h_bunch must be given to the slicer
                as an input parameter if multi turn or multi bunch wakes are used!""")

        if mpi == 'circular_mpi_full_ring_fft':
            if (not 'Q_x' in kwargs) or (not 'Q_y' in kwargs) or\
            (not 'beta_x' in kwargs) or (not 'beta_y' in kwargs):
                raise ValueError("""The mpi option circular_mpi_full_ring_fft
                                 requires Q_x, Q_y, beta_x and beta_y to be
                                 given as input parameters""")
            else:
                self._circular_convolution_parameters = {
                        'Q_x': kwargs['Q_x'],
                        'Q_y': kwargs['Q_y'],
                        'beta_x': kwargs['beta_x'],
                        'beta_y': kwargs['beta_y'],
                        }
        else:
            self._circular_convolution_parameters = None

        self._mpi = mpi
        self._mpi_gatherer = None
        self._required_variables = ['age', 'beta', 't_centers',
                                    'n_macroparticles_per_slice',
                                    'mean_x', 'mean_y']

    def _serial_track(self, beam):

        bunches_list = beam.split_to_views()

        # Updates ages of bunches in slice_set_deque
        for i, turnbyturn in enumerate(self.slice_set_deque):
            beta = turnbyturn[1]
            for j, bunchbybunch in enumerate(beta):
                age = self.circumference/(bunchbybunch[0]*c)
                turnbyturn[0][j] += age
                # print ("\n\n--> age and beta: {:g}, {:g}".format(beta, 0))
                # print ("\n\n--> age and beta: {:g}, {:g}".format(b[0][0], b[1][0]))

        force_absolute = hasattr(self.slicer, 'force_absolute') and self.slicer.force_absolute
        # Fills wake register -
        slice_set_list = []
        n_bunches_total = len(bunches_list)
        n_slices = self.slicer.n_slices
        if self.slicer.config[3] is not None and not force_absolute: # slicer.z_cuts
            # In this case, we need to bring bunches back to zero
            for i, b in enumerate(bunches_list):
                z_delay = b.mean_z()
                b.z -= z_delay
                s = b.get_slices(self.slicer, statistics=['mean_x', 'mean_y'])
                b.z += z_delay
                s.z_bins += z_delay
                slice_set_list.append(s)
        else:
            for i, b in enumerate(bunches_list):
                s = b.get_slices(self.slicer, statistics=['mean_x', 'mean_y'])
                slice_set_list.append(s)

        self.slice_set_deque.appendleft(
            [np.array([np.ones(n_slices) * s.age for s in slice_set_list]),
             np.array([np.ones(n_slices) * s.beta for s in slice_set_list]),
             np.array([s.t_centers for s in slice_set_list]),
             np.array([s.n_macroparticles_per_slice for s in slice_set_list]),
             np.array([s.mean_x for s in slice_set_list]),
             np.array([s.mean_y for s in slice_set_list]),
             np.arange(n_bunches_total)]
        )

        for kick in self.wake_kicks:
            kick.apply(bunches_list, self.slice_set_deque)


    def _mpi_track(self, beam):

        # Creates a mpi gatherer, if it has not been created earlier
        if self._mpi_gatherer is None:
            self._mpi_gatherer = mpi_data.MpiGatherer(self.slicer,
                                                      self._required_variables)

        # Gathers data from all bunches
        self._mpi_gatherer.gather(beam)

        # Updates ages of bunches in slice_set_deque
        for i, turnbyturn in enumerate(self.slice_set_deque):
            beta = turnbyturn[1]
            for j, bunchbybunch in enumerate(beta):
                age = self.circumference/(bunchbybunch[0]*c)
                turnbyturn[0][j] += age
                # print ("\n\n--> age and beta: {:g}, {:g}".format(beta, 0))
                # print ("\n\n--> age and beta: {:g}, {:g}".format(b[0][0], b[1][0]))

        # Fills wake register - little trick here to include
        # local_bunch_indexes that will be used in wake kicks.apply. Makes
        # deque no longer convertible into an ndarray. Needs to be poped later.
        assert(self.slicer == self._mpi_gatherer._slicer)
        n_bunches_total = self._mpi_gatherer.n_bunches
        n_slices = self.slicer.n_slices
        self.slice_set_deque.appendleft(
            [self._mpi_gatherer.total_data.age,
             self._mpi_gatherer.total_data.beta,
             self._mpi_gatherer.total_data.t_centers,
             self._mpi_gatherer.total_data.n_macroparticles_per_slice,
             self._mpi_gatherer.total_data.mean_x,
             self._mpi_gatherer.total_data.mean_y,
             self._mpi_gatherer.local_bunch_indexes]
            )
        for i, v in enumerate(self.slice_set_deque[0][:-1]):
            self.slice_set_deque[0][i] = np.reshape(
                v, (n_bunches_total, n_slices))

        # if self._mpi_gatherer.mpi_rank == 0:
        #     print(self._mpi_gatherer.total_data.t_centers)
        # print(len(self.slice_set_deque[0]))
        # print(self.slice_set_deque[0][1].shape)
        # print(self._mpi_gatherer.total_data.beta)
        # wurstel


        for kick in self.wake_kicks:
            kick.apply(self._mpi_gatherer.bunch_list, self.slice_set_deque)

        # At the end the superbunch must be rebunched. Without that the kicks
        # do not apply to the next turn
        self._mpi_gatherer.rebunch(beam)

        # signal_x = np.array([])
        # signal_y = np.array([])

        # # slice set data from all bunches in all processors can be found from under mpi_gatherer.total_data object
        # if self._axis == 'divergence':
        #     signal_x = np.array([s for s in self._mpi_gatherer.total_data.mean_xp])
        #     signal_y = np.array([s for s in self._mpi_gatherer.total_data.mean_yp])

        # elif self._axis == 'displacement':
        #     signal_x = np.array([s for s in self._mpi_gatherer.total_data.mean_x])
        #     signal_y = np.array([s for s in self._mpi_gatherer.total_data.mean_y])


        # # the object mpi_gatherer.total_data can be used as a normal slice_set object expect that bin_set is slightly different
        # for processor in self._processors_x:
        #     signal_x = processor.process(signal_x,self._mpi_gatherer.total_data, None, mpi = True)

        # for processor in self._processors_y:
        #     signal_y = processor.process(signal_y,self._mpi_gatherer.total_data, None, mpi = True)

        # # mpi_gatherer.gather(...) splits the superbunch, so it is efficient to use same bunch list
        # for i, b in enumerate(self._mpi_gatherer.bunch_list):

        #     # the slice set data from all bunches in all processors pass the signal processors. Here, the correction
        #     # signals for the bunches tracked in this processors are picked by using indexes found from
        #     # mpi_gatherer.total_data.local_data_locations
        #     idx_from = self._mpi_gatherer.total_data.local_data_locations[i][0]
        #     idx_to = self._mpi_gatherer.total_data.local_data_locations[i][1]

        #     correction_x = self._gain_x*signal_x[idx_from:idx_to]
        #     correction_y = self._gain_y*signal_y[idx_from:idx_to]

        #     # mpi_gatherer has also slice set list, which can be used for applying the kicks
        #     p_idx = self._mpi_gatherer.slice_set_list[i].particles_within_cuts
        #     s_idx = self._mpi_gatherer.slice_set_list[i].slice_index_of_particle.take(p_idx)

        #     if self._axis == 'divergence':
        #         b.xp[p_idx] -= correction_x[s_idx]
        #         b.yp[p_idx] -= correction_y[s_idx]

        #     elif self._axis == 'displacement':
        #         b.x[p_idx] -= correction_x[s_idx]
        #         b.y[p_idx] -= correction_y[s_idx]


    def _mpi_track_optimized(self, beam, optimization_method):

        # Creates a mpi gatherer, if it has not been created earlier
        if self._mpi_gatherer is None:
            self._mpi_gatherer = mpi_data.MpiGatherer(self.slicer,
                                                      self._required_variables)

        # Gathers data from all bunches
        self._mpi_gatherer.gather(beam)
        all_slice_sets = self._mpi_gatherer.bunch_by_bunch_data
        local_slice_sets = self._mpi_gatherer.slice_set_list
        bunch_list = self._mpi_gatherer.bunch_list
        local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes


        if not hasattr(self, '_turns_on_this_proc'):
            kick_turn_data = []

            total_n_turns = 0
            for kick in self.wake_kicks:
                kick_turn_data.append(np.arange(total_n_turns,
                                                 total_n_turns+kick.n_turns_wake))
                total_n_turns += kick.n_turns_wake

            all_convolutions = np.arange(total_n_turns)
            my_convolutions = mpi_data.my_tasks(all_convolutions)

            self._turns_on_this_proc = []

            for i in range(len(self.wake_kicks)):
                calculate_on_this_proc = []
                for j in my_convolutions:
                    if j in kick_turn_data[i]:
                        for k, val  in enumerate(kick_turn_data[i]):
                            if val == j:
                                calculate_on_this_proc.append(k)

                self._turns_on_this_proc.append(calculate_on_this_proc)

        # Calculates wakes fields for different turns in parallel if possible.
        # Because a wake field calculation for one turn is difficult to
        # parallelize, the parallelization occurs by splitting different turns
        # in different kicks to different processors. The wake fields from
        # different tursn are gathered when the kicks are applied.

        for kick, turns in zip(self.wake_kicks, self._turns_on_this_proc):

            kick.calculate_field(all_slice_sets,local_slice_sets,bunch_list,
                                 local_bunch_indexes, optimization_method, turns,
                                 self._circular_convolution_parameters)

        # ensures that everything is calculated, i.e. synchronizes threads
        mpi_data.share_numbers(1)

        for kick in self.wake_kicks:
            kick.apply(bunch_list, all_slice_sets, local_slice_sets,
                       local_bunch_indexes, optimization_method)

    def track_classic(self, beam):
        """Update macroparticle momenta according to wake kick.

        First, splits up beam into a set of bunches which can be individually
        sliced. Extracts slice data and send all slice data to the register on
        master.


        The function iterates through all bunches in the list and calls the
        WakeKick.apply(bunch, slice_set) method of each of the WakeKick objects
        stored in self.wake_kicks. A slice_set is necessary to perform this
        operation. It is requested from the bunch (instance of the Particles
        class) using the Particles.get_slices(self.slicer) method, where
        self.slicer is the instance of the Slicer class used for this
        particluar WakeField object. A slice_set is returned according to the
        self.slicer configuration. The statistics mean_x and mean_y are
        requested to be calculated and saved in the SliceSet instance, too,
        s.t. the first moments x, y can be calculated by the WakeKick
        instances.

        Args:

            bunches: A bunch/beam or a list of bunches.

        """
        n_slices = self.slicer.n_slices
        stride = 2 + 4*n_slices

        bunches_list = beam.split_to_views()
        n_bunches_counts = self.comm.allgather(len(bunches_list))
        n_bunches_offsets = np.cumsum(n_bunches_counts)
        n_bunches_total = sum(n_bunches_counts)

        slice_data = self._get_slice_data(bunches_list)
        slice_data_counts = self.comm.allgather(len(slice_data))
        slice_data_offsets = np.insert(
            np.cumsum(slice_data_counts), 0, 0)[:-1]

        # The register is assembled and now sent to all processors
        register = np.zeros((stride * n_bunches_total))
        self.comm.Allgatherv(
            [slice_data, len(slice_data), MPI.DOUBLE],
            [register, slice_data_counts, slice_data_offsets, MPI.DOUBLE])

        # Update ages of bunches in slice_set_deque
        for i, t in enumerate(self.slice_set_deque):
            for j, b in enumerate(t):
                beta = b[1]
                age = self.circumference/(beta*c)
                b[0] += age

        self.slice_set_deque.appendleft(
            np.reshape(register, (n_bunches_total, stride)))
        print(self.slice_set_deque[-1][:, 0])

        for kick in self.wake_kicks:
            kick.apply(bunches_list, self.slice_set_deque)

        # Here, we need to put the values back into the reference!
        beam_new = sum(bunches_list)
        # beam.update({'x': beam_new.x,
        #              'y': beam_new.y,
        #              'z': beam_new.z,
        #              'xp': beam_new.xp,
        #              'yp': beam_new.yp,
        #              'dp': beam_new.dp
        beam.x[:] = beam_new.x[:]
        beam.xp[:] = beam_new.xp[:]
        beam.y[:] = beam_new.y[:]
        beam.yp[:] = beam_new.yp[:]
        beam.z[:] = beam_new.z[:]
        beam.dp[:] = beam_new.dp[:]

    def track(self, beam):
        if isinstance(self._mpi, str):
                self._mpi_track_optimized(beam, self._mpi)

        elif self._mpi:
            self._mpi_track(beam)
        else:
            self._serial_track(beam)


# ==============================================================================
# WAKE SOURCE BASE CLASS FOLLOWED BY COLLECTION OF DIFFERENT WAKE SOURCES
# ==============================================================================
class WakeSource(Printing, metaclass=ABCMeta):
    """Abstract base class for wake sources, such as WakeTable,
    Resonator or ResistiveWall.
    """

    @abstractmethod
    def get_wake_kicks(self, slicer_mode):
        """Factory method. Creates instances of the WakeKick objects for the given
        WakeSource and returns them as a list wake_kicks.  This method is
        usually only called by a WakeField object to collect and create all the
        WakeKick objects originating from the different sources. (The slicer
        mode Slicer.mode must be passed at instantiation of a WakeKick object
        only to set the appropriate convolution method. See docstrings of
        WakeKick class.)

        """
        pass


class WakeTable(WakeSource):
    """Class to define wake functions and WakeKick objects using wake data from a
    table.
    """

    def __init__(self, wake_file, wake_file_columns, n_turns_wake=1,
                 *args, **kwargs):
        """Load data from the wake_file and store them in a dictionary
        self.wake_table. Keys are the names specified by the user in
        wake_file_columns and describe the names of the wake field components
        (e.g. dipole_x or dipole_yx). The dict values are given by the
        corresponding data read from the table. The nomenclature of the wake
        components must be strictly obeyed.  Valid names for wake components
        are:

        'constant_x', 'constant_y', 'dipole_x', 'dipole_y', 'dipole_xy',
        'dipole_yx', 'quadrupole_x', 'quadrupole_y', 'quadrupole_xy',
        'quadrupole_yx', 'longitudinal'.

        The order of wake_file_columns is relevant and must correspond to the
        one in the wake_file. There is no way to check this here and it is in
        the responsibility of the user to ensure it is correct. Two checks made
        here are whether the length of wake_file_columns corresponds to the
        number of columns in the wake_file and whether a column 'time' is
        specified.

        The units and signs of the wake table data are assumed to follow
        the HEADTAIL conventions, i.e.
          time: [ns]
          transverse wake components: [V/pC/mm]
          longitudinal wake component: [V/pC].

        The parameter 'n_turns_wake' defines how many turns are considered for
        the multiturn wakes. It is 1 by default, i.e.  multiturn wakes are off.
        """
        super(WakeTable, self).__init__(*args, **kwargs)

        self.wake_table = {}

        wake_data = np.loadtxt(wake_file)
        if len(wake_file_columns) != wake_data.shape[1]:
            raise ValueError("Length of wake_file_columns list does not" +
                             " correspond to the number of columns in the" +
                             " specified wake_file. \n")
        if 'time' not in wake_file_columns:
            raise ValueError("No wake_file_column with name 'time' has" +
                             " been specified. \n")

        for i, column_name in enumerate(wake_file_columns):
            self.wake_table.update({column_name: wake_data[:, i]})

        self.n_turns_wake = n_turns_wake

        self.wake_type = {
            'constant_x':    ConstantWakeKickX,
            'constant_y':    ConstantWakeKickY,
            'longitudinal':  ConstantWakeKickZ,
            'dipole_x':      DipoleWakeKickX,
            'dipole_y':      DipoleWakeKickY,
            'dipole_xy':     DipoleWakeKickXY,
            'dipole_yx':     DipoleWakeKickYX,
            'quadrupole_x':  QuadrupoleWakeKickX,
            'quadrupole_y':  QuadrupoleWakeKickY,
            'quadrupole_xy': QuadrupoleWakeKickXY,
            'quadrupole_yx': QuadrupoleWakeKickYX}

    def get_wake_kicks(self, slicer):
        """Factory method. Creates instances of the appropriate WakeKick objects for
        all the wake components provided by the user (and the wake table
        data). The WakeKick objects are returned as a list wake_kicks.

        """
        wake_kicks = []

        for name, function in list(self.wake_type.items()):
            if self._is_provided(name):
                if name == 'longitudinal':
                    wake_function = self.function_longitudinal()
                else:
                    wake_function = self.function_transverse(name)
                wake_kicks.append(
                    function(wake_function, slicer, self.n_turns_wake))

        return wake_kicks

    def _is_provided(self, wake_component):
        """ Check whether wake_component is a valid name and available
        in wake table data. Return 'True' if yes and 'False' if no. """
        if wake_component in list(self.wake_table.keys()):
            return True
        else:
            # self.warns(wake_component + ' \n' +
            #       'Wake component is either not provided or does not \n'+
            #       'use correct nomenclature. See docstring of WakeTable \n' +
            #       'constructor to display valid names. \n')
            return False

    def function_transverse(self, wake_component):
        """ Defines and returns the wake(beta, dz) function for the
        given wake_component (transverse). Data from the wake table are
        used, but first converted to SI units assuming that time is
        specified in [ns] and transverse wake field strengths in
        [V/pC/mm]. Sign conventions are applied (HEADTAIL conventions).
        dz is related to wake table time data by dz = beta c dt (dz < 0
        for the ultrarelativistic case).

        The wake(dt) uses the scipy.interpolate.interp1d linear
        interpolation to calculate the wake strength at an arbitrary
        value of dt (provided it is in the valid range). The valid range
        of dt is given by the time range from the wake table. If values
        of wake(dt) are requested for dt outside the valid range, a
        ValueError is raised by interp1d.

        Very basic conformity checks for the wake table data are already
        performed at definition time of the wake(dt) method. E.g.
        whether the specified wake is valid only for ultrarelativistic
        cases or low beta cases. In the former case, the wake strength
        at time 0 must be defined by the user!
        """

        convert_to_s = 1e-9
        convert_to_V_per_Cm = 1e15

        time = convert_to_s * self.wake_table['time']
        wake_strength = -convert_to_V_per_Cm * self.wake_table[wake_component]
        interpolation_function = interp1d(time, wake_strength)

        if (time[0] == 0) and (wake_strength[0] == 0):
            def wake(dt, *args, **kwargs):
                dt = dt.clip(max=0)
                return interpolation_function(-dt)
            self.prints(wake_component +
                        ' Assuming ultrarelativistic wake.')
        elif (time[0] < 0):
            def wake(dt, *args, **kwargs):
                return interpolation_function(-dt)
            self.prints(wake_component +  ' Found low beta wake.')

        else:
            raise ValueError(wake_component +
                             ' does not meet requirements.')

        return wake

    def function_longitudinal(self):
        """Defines and returns the wake(dt, *args, **kwargs) function for the given
        wake_component (longitudinal). Data from the wake table are used, but
        first converted to SI units assuming that time is specified in [ns] and
        longitudinal wake field strength in [V/pC]. Sign conventions are
        applied (HEADTAIL conventions).  The wake(dt, *args, **kwargs) uses the
        scipy.interpolate.interp1d linear interpolation to calculate the wake
        strength at an arbitrary value of dt (provided it is in the valid
        range). The valid range of dt is given by the time range from the wake
        table. If values of wake(dt, *args, **kwargs) are requested for dt
        outside the valid range, a ValueError is raised by interp1d.  The beam
        loading theorem is respected and applied for dt=0.

        """
        convert_to_s = 1e-9
        convert_to_V_per_C = 1e12

        time = convert_to_s * self.wake_table['time']
        wake_strength = -convert_to_V_per_C * self.wake_table['longitudinal']
        interpolation_function = interp1d(time, wake_strength)

        def wake(dt, *args, **kwargs):
            wake_interpolated = interpolation_function(-dt)
            if time[0] == 0:
                # Beam loading theorem: Half value of wake strength at
                # dt = 0.
                return (np.sign(-dt) + 1.) / 2. * wake_interpolated
            elif time[0] < 0:
                return wake_interpolated
            else:
                raise ValueError('Longitudinal wake component does not meet' +
                                 ' requirements.')

        return wake


class Resonator(WakeSource):
    """ Class to describe the wake functions originating from a
    resonator impedance. Alex Chao's resonator model (Eq. 2.82) is used
    as well as the definitions from HEADTAIL. """

    def __init__(self, R_shunt, frequency, Q,
                 Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, switch_Z,
                 n_turns_wake=1, *args, **kwargs):
        """ General constructor to create a Resonator WakeSource object
        describing the wake functions of a resonator impedance. Alex
        Chao's resonator model (Eq. 2.82) is used as well as definitions
        from HEADTAIL.
        Note that it is no longer allowed to pass a LIST of parameters
        to generate a number of resonators with different parameters
        within the same Resonator object. Instead, create the Resonator
        objects and pass all of them to the WakeField constructor.
        The parameter 'n_turns_wake' defines how many turns are
        considered for the multiturn wakes. It is 1 by default, i.e.
        multiturn wakes are off. """
        super(Resonator, self).__init__(*args, **kwargs)

        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q
        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_Y2 = Yokoya_Y2
        self.switch_Z = switch_Z
        self.n_turns_wake = n_turns_wake

    def get_wake_kicks(self, slicer):
        """ Factory method. Creates instances of the appropriate
        WakeKick objects for a Resonator WakeSource with the specified
        parameters. A WakeKick object is instantiated only if the
        corresponding Yokoya factor is non-zero. The WakeKick objects
        are returned as a list wake_kicks. """
        wake_kicks = []

        # Dipole wake kick x.
        if self.Yokoya_X1:
            wake_function = self.function_transverse(self.Yokoya_X1)
            wake_kicks.append(DipoleWakeKickX(
                wake_function, slicer, self.n_turns_wake))

        # Quadrupole wake kick x.
        if self.Yokoya_X2:
            wake_function = self.function_transverse(self.Yokoya_X2)
            wake_kicks.append(QuadrupoleWakeKickX(
                wake_function, slicer, self.n_turns_wake))

        # Dipole wake kick y.
        if self.Yokoya_Y1:
            wake_function = self.function_transverse(self.Yokoya_Y1)
            wake_kicks.append(DipoleWakeKickY(
                wake_function, slicer, self.n_turns_wake))

        # Quadrupole wake kick y.
        if self.Yokoya_Y2:
            wake_function = self.function_transverse(self.Yokoya_Y2)
            wake_kicks.append(QuadrupoleWakeKickY(
                wake_function, slicer, self.n_turns_wake))

        # Constant wake kick z.
        if self.switch_Z:
            wake_function = self.function_longitudinal()
            wake_kicks.append(ConstantWakeKickZ(
                wake_function, slicer, self.n_turns_wake))

        return wake_kicks

    def function_transverse(self, Yokoya_factor):
        """ Define the wake function (transverse) of a resonator with
        the given parameters according to Alex Chao's resonator model
        (Eq. 2.82) and definitions of the resonator in HEADTAIL. """
        omega = 2 * np.pi * self.frequency
        alpha = omega / (2 * self.Q)
        omegabar = np.sqrt(np.abs(omega**2 - alpha**2))

        def wake(dt, *args, **kwargs):
            dt = dt.clip(max=0)
            if self.Q > 0.5:
                y = (Yokoya_factor * self.R_shunt * omega**2 / (self.Q *
                     omegabar) * np.exp(alpha*dt) * sin(omegabar*dt))
            elif self.Q == 0.5:
                y = (Yokoya_factor * self.R_shunt * omega**2 / self.Q *
                     np.exp(alpha * dt) * dt)
            else:
                y = (Yokoya_factor * self.R_shunt * omega**2 / (self.Q *
                     omegabar) * np.exp(alpha*dt) * np.sinh(omegabar*dt))
            return y
        return wake

    def function_longitudinal(self):
        """ Define the wake function (longitudinal) of a resonator with
        the given parameters according to Alex Chao's resonator model
        (Eq. 2.82) and definitions of the resonator in HEADTAIL. """
        omega = 2 * np.pi * self.frequency
        alpha = omega / (2 * self.Q)
        omegabar = np.sqrt(np.abs(omega**2 - alpha**2))

        def wake(dt, *args, **kwargs):
            if self.Q > 0.5:
                y = (-(np.sign(dt) - 1) * self.R_shunt * alpha *
                     np.exp(alpha * dt) * (cos(omegabar * dt) +
                     alpha / omegabar * sin(omegabar*dt)))
            elif self.Q == 0.5:
                y = (-(np.sign(dt) - 1) * self.R_shunt * alpha *
                     np.exp(alpha * dt) * (1. + alpha * dt))
            elif self.Q < 0.5:
                y = (-(np.sign(dt) - 1) * self.R_shunt * alpha *
                     np.exp(alpha * dt) * (np.cosh(omegabar * dt) +
                     alpha / omegabar * np.sinh(omegabar * dt)))
            return y
        return wake


class CircularResonator(Resonator):
    '''Circular Resonator.'''
    def __init__(self, R_shunt, frequency, Q, n_turns_wake=1,
                 *args, **kwargs):
        """ Special case of circular resonator. """
        Yokoya_X1 = 1.
        Yokoya_Y1 = 1.
        Yokoya_X2 = 0.
        Yokoya_Y2 = 0.
        switch_Z = False

        super(CircularResonator, self).__init__(
            R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1,
            Yokoya_X2, Yokoya_Y2, switch_Z, n_turns_wake, *args, **kwargs)


class ParallelHorizontalPlatesResonator(Resonator):
    '''Broad-band resonator for horizontal parallel plates.'''
    def __init__(self, R_shunt, frequency, Q, n_turns_wake=1,
                 *args, **kwargs):
        """ Special case of parallel plate resonator. """
        Yokoya_X1 = np.pi**2 / 24.
        Yokoya_Y1 = np.pi**2 / 12.
        Yokoya_X2 = -np.pi**2 / 24.
        Yokoya_Y2 = np.pi**2 / 24.
        switch_Z = False

        super(ParallelHorizontalPlatesResonator, self).__init__(
            R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1,
            Yokoya_X2, Yokoya_Y2, switch_Z, n_turns_wake, *args, **kwargs)


@deprecated('--> "ParallelPlatesResonator" will be removed '
            'in the near future. '
            'Use "ParallelHorizontalPlatesResonator" instead.\n')
class ParallelPlatesResonator(ParallelHorizontalPlatesResonator):
    pass


class ParallelVerticalPlatesResonator(Resonator):
    '''Broad-band resonator for vertical parallel plates.'''
    def __init__(self, R_shunt, frequency, Q, n_turns_wake=1,
                 *args, **kwargs):
        """ Special case of parallel plate resonator. """
        Yokoya_X1 = np.pi**2 / 12.
        Yokoya_Y1 = np.pi**2 / 24.
        Yokoya_X2 = np.pi**2 / 24.
        Yokoya_Y2 = -np.pi**2 / 24.
        switch_Z  = False

        super(ParallelVerticalPlatesResonator, self).__init__(
            R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1,
            Yokoya_X2, Yokoya_Y2, switch_Z, n_turns_wake, *args, **kwargs)


class ResistiveWall(WakeSource):
    """ Class to describe the wake functions originating from a
    resistive wall impedance. """

    def __init__(self, pipe_radius, resistive_wall_length, conductivity,
                 dt_min, Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2,
                 n_turns_wake=1, *args, **kwargs):
        """Resistive wall wake fied contructor

        General constructor to create a ResistiveWall WakeSource object
        describing the wake functions of a resistive wall impedance. The wake
        function is implemented according to A. Chao eq.(2.53) in SI
        units. Since the function diverges at t=0, a cut-off time dt_min must
        be specified below which the wake is assumed to be constant. The
        parameter 'n_turns_wake' defines how many turns are considered for the
        multiturn wakes. It is 1 by default, i.e.  multiturn wakes are off.

        Arguments:
        pipe_radius -- the resistive wall pipe radius in [m]
        resistive_wall_length -- the resistive wall pipe length in [m]
        conductivity -- conductivity in [?]
        dt_min -- the minimum slice width in [s]
        n-turns_wake -- number of turns for multiturn wake (1 by default)

        """
        super(ResistiveWall, self).__init__(*args, **kwargs)

        self.pipe_radius = np.array([pipe_radius]).flatten()
        self.resistive_wall_length = resistive_wall_length
        self.conductivity = conductivity
        self.dt_min = dt_min

        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y2 = Yokoya_Y2

        self.n_turns_wake = n_turns_wake

    def get_wake_kicks(self, slicer):
        """ Factory method. Creates instances of the appropriate
        WakeKick objects for the ResistiveWall WakeSource with the
        specified parameters. A WakeKick object is instantiated only if
        the corresponding Yokoya factor is non-zero. The WakeKick
        objects are returned as a list wake_kicks. """
        wake_kicks = []

        # Dipole wake kick x.
        if self.Yokoya_X1:
            wake_function = self.function_transverse(self.Yokoya_X1)
            wake_kicks.append(DipoleWakeKickX(
                wake_function, slicer, self.n_turns_wake))

        # Quadrupole wake kick x.
        if self.Yokoya_X2:
            wake_function = self.function_transverse(self.Yokoya_X2)
            wake_kicks.append(QuadrupoleWakeKickX(
                wake_function, slicer, self.n_turns_wake))

        # Dipole wake kick y.
        if self.Yokoya_Y1:
            wake_function = self.function_transverse(self.Yokoya_Y1)
            wake_kicks.append(DipoleWakeKickY(
                wake_function, slicer, self.n_turns_wake))

        # Quadrupole wake kick y.
        if self.Yokoya_Y2:
            wake_function = self.function_transverse(self.Yokoya_Y2)
            wake_kicks.append(QuadrupoleWakeKickY(
                wake_function, slicer, self.n_turns_wake))

        return wake_kicks

    def function_transverse(self, Yokoya_factor):
        """Define the wake function (transverse) of a resistive wall with the given
        parameters. The function explicitly depends on the relativistic beta of
        the beam. This is a peculiarity of the resistive wall wake. And in
        particular for multi-turn wakes the history of the beam beta needs to
        be passed. For this reason, beta needs to be provided at the wake
        function level and as a consequence all wake functions need to be
        adapted not to violate the interface.

        """
        mu_r = 1
        # The impedance of free space [Ohm]
        Z_0 = 119.9169832 * np.pi

        def wake(dt, *args, **kwargs):
            y = (Yokoya_factor * (np.sign(dt + np.abs(self.dt_min)) - 1) / 2. *
                 np.sqrt(kwargs['beta']) * self.resistive_wall_length / np.pi /
                 self.pipe_radius**3 * np.sqrt(-mu_r / np.pi /
                 self.conductivity / dt.clip(max=-abs(self.dt_min))))*np.sqrt(Z_0*c)
            return y
        return wake


class CircularResistiveWall(ResistiveWall):
    '''Circular resistive wall.'''
    def __init__(self, pipe_radius, resistive_wall_length, conductivity,
                 dt_min, n_turns_wake=1, *args, **kwargs):
        """ Special case of a circular resistive wall. """
        Yokoya_X1 = 1.
        Yokoya_Y1 = 1.
        Yokoya_X2 = 0.
        Yokoya_Y2 = 0.

        super(CircularResistiveWall, self).__init__(
            pipe_radius, resistive_wall_length, conductivity, dt_min,
            Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, n_turns_wake,
            *args, **kwargs)


class ParallelHorizontalPlatesResistiveWall(ResistiveWall):
    '''Resistive wall impedance for horizontal parallel plates.'''
    def __init__(self, pipe_radius, resistive_wall_length, conductivity,
                 dt_min, n_turns_wake=1, *args, **kwargs):
        """ Special case of a parallel plates resistive wall. """
        Yokoya_X1 = np.pi**2 / 24.
        Yokoya_Y1 = np.pi**2 / 12.
        Yokoya_X2 = -np.pi**2 / 24.
        Yokoya_Y2 = np.pi**2 / 24.

        super(ParallelHorizontalPlatesResistiveWall, self).__init__(
            pipe_radius, resistive_wall_length, conductivity, dt_min,
            Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, n_turns_wake,
            *args, **kwargs)

@deprecated('--> "ParallelPlatesResistiveWall" will be removed '
            'in the near future. '
            'Use "ParallelHorizontalPlatesResistiveWall" instead.\n')
class ParallelPlatesResistiveWall(ParallelHorizontalPlatesResistiveWall):
    pass


class ParallelVerticalPlatesResistiveWall(Resonator):
    '''Resistive wall impedance for vertical parallel plates.'''
    def __init__(self, pipe_radius, resistive_wall_length, conductivity,
                 dt_min, n_turns_wake=1, *args, **kwargs):
        """ Special case of a parallel plates resistive wall. """
        Yokoya_X1 = np.pi**2 / 12.
        Yokoya_Y1 = np.pi**2 / 24.
        Yokoya_X2 = np.pi**2 / 24.
        Yokoya_Y2 = -np.pi**2 / 24.

        super(ParallelVerticalPlatesResistiveWall, self).__init__(
            pipe_radius, resistive_wall_length, conductivity, dt_min,
            Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, n_turns_wake,
            *args, **kwargs)


def check_wake_sampling(bunch, slicer, wakes, beta=1, wake_column=None, bins=False):
    '''
    Handy function for quick visual check of sampling of the wake functions.
    For now only implemented for wake table type wakes.
    '''
    from scipy.constants import c
    import matplotlib.pyplot as plt

    ss = bunch.get_slices(slicer).z_centers
    zz = bunch.get_slices(slicer).z_bins
    ss = ss[:-1]
    ll = bunch.get_slices(slicer).lambda_z(ss, sigma=100)
    # ss = np.concatenate((s.z_centers-s.z_centers[-1], (s.z_centers-s.z_centers[0])[1:]))

    A = [wakes.wake_table['time'] * beta*c*1e-9, wakes.wake_table[wake_column] * 1e15]
    W = [ss[::-1], wakes.function_transverse(wake_column)(beta, ss)]


    fig, (ax1, ax2) = plt.subplots(2, figsize=(16,12), sharex=True)

    ax1.plot(ss, ll)

    ax2.plot(A[0], (A[1]), 'b-+', ms=12)
    ax2.plot(W[0][:-1], (-1*W[1][1:]), 'r-x')
    if bins:
        [ax2.axvline(z, color='g') for z in zz]

    ax2.grid()
    lgd = ['Table', 'Interpolated']
    if bins:
        lgd += ['Bin edges']
    ax2.legend(lgd)

    print('\n--> Resulting number of slices: {:g}'.format(len(ss)))

    return ax1



