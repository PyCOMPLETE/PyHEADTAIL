import numpy as np
import collections
import itertools
import copy
from PyHEADTAIL.mpi import mpi_data
"""
    This file contains modules, which can be used as a feedback module/object in PyHEADTAIL. Actual signal processing is
    done by using signal processors written to files processors.py and digital_processors.py. A list of signal
    processors is given as a argument for feedback elements.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN
"""

def get_processor_variables(processors, required_variables = None):
    """Function which checks statistical variables required by signal processors

    :param processors: a list of signal processors
    :param variables: a list of statistical variables determined earlier
    :return: a list of statistical variables, which is a sum of variables from input list and those found from
    the signal processors
    """

    if required_variables is None:
        required_variables = []

    for processor in processors:
        required_variables.extend(processor.required_variables)

    required_variables = list(set(required_variables))

    if 'z_bins' in required_variables:
        required_variables.remove('z_bins')

    statistical_variables = copy.deepcopy(required_variables)

    if 'n_macroparticles_per_slice' in statistical_variables:
        statistical_variables.remove('n_macroparticles_per_slice')

    return required_variables, statistical_variables


class IdealBunchFeedback(object):
    """ The simplest possible feedback. It corrects a gain fraction of a mean xp/yp value of the bunch.
    """
    def __init__(self,gain):
        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

    def track(self,bunch):
        bunch.xp -= self._gain_x *bunch.mean_xp()
        bunch.yp -= self._gain_y*bunch.mean_yp()


class IdealSliceFeedback(object):
    """Corrects a gain fraction of a mean xp/yp value of each slice in the bunch."""
    def __init__(self,gain,slicer):

        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

        self._slicer = slicer

    def track(self,bunch):
        slice_set = bunch.get_slices(self._slicer, statistics = ['mean_xp', 'mean_yp'])

        # Reads a particle index and a slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] -= self._gain_x * slice_set.mean_xp[s_idx]
        bunch.yp[p_idx] -= self._gain_y * slice_set.mean_yp[s_idx]


class OneboxFeedback(object):
    """ General class for a simple feedback, where a pick up and a kicker is located in the same place. It takes
        mean_xp/yp or mean_x/y values of each slice and pass them through signal processor chains given in parameters
        processors_x and processors_y. The final correction for x/y or xp/yp values is a gain times the signals through
        the signal processors. Axes (xp/yp or x/y) can be chosen by giving input parameter axis='divergence' for xp/yp
        and axis='displacement' for x/y. The default axis is divergence.
    """
    def __init__(self, gain, slicer, processors_x, processors_y, axis='divergence', mpi = False, extra_statistics = None):
        """

        :param gain:
        :param slicer:
        :param processors_x:
        :param processors_y:
        :param axis:
        :param mpi: if True, data are gathered from the processors
        :param extra_statistics: variable for debugging mpi and plotting the gathered data. It will be removed
        """

        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y

        self._axis = axis

        if self._axis == 'divergence':
            self._required_variables = ['mean_xp', 'mean_yp']
        elif self._axis == 'displacement':
            self._required_variables = ['mean_x', 'mean_y']

        if extra_statistics is not None:
            self._required_variables += extra_statistics

        self._required_variables, self._statistical_variables = get_processor_variables(self._processors_x, self._required_variables)
        self._required_variables, self._statistical_variables = get_processor_variables(self._processors_y, self._required_variables)

        self._mpi = mpi
        self._mpi_gatherer = None

    def track(self,bunch):

        if self._mpi:
            self._mpi_track(bunch)
        else:
            self._normal_track(bunch)

    def _mpi_track(self,superbunch):
        # this is the new code required by the multi bunch feedback

        # creates a mpi gatherer, if it has not been created earlier
        if self._mpi_gatherer is None:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer, self._required_variables)

        # gathers data from all bunches
        self._mpi_gatherer.gather(superbunch)

        signal_x = np.array([])
        signal_y = np.array([])

        # slice set data from all bunches in all processors can be found from under mpi_gatherer.total_data object
        if self._axis == 'divergence':
            signal_x = np.array([s for s in self._mpi_gatherer.total_data.mean_xp])
            signal_y = np.array([s for s in self._mpi_gatherer.total_data.mean_yp])

        elif self._axis == 'displacement':
            signal_x = np.array([s for s in self._mpi_gatherer.total_data.mean_x])
            signal_y = np.array([s for s in self._mpi_gatherer.total_data.mean_y])


        # the object mpi_gatherer.total_data can be used as a normal slice_set object expect that bin_set is slightly different
        for processor in self._processors_x:
            signal_x = processor.process(signal_x,self._mpi_gatherer.total_data, None, mpi = True)

        for processor in self._processors_y:
            signal_y = processor.process(signal_y,self._mpi_gatherer.total_data, None, mpi = True)

        # mpi_gatherer.gather(...) splits the superbunch, so it is efficient to use same bunch list
        for i, b in enumerate(self._mpi_gatherer.bunch_list):

            # the slice set data from all bunches in all processors pass the signal processors. Here, the correction
            # signals for the bunches tracked in this processors are picked by using indexes found from
            # mpi_gatherer.total_data.local_data_locations
            idx_from = self._mpi_gatherer.total_data.local_data_locations[i][0]
            idx_to = self._mpi_gatherer.total_data.local_data_locations[i][1]

            correction_x = self._gain_x*signal_x[idx_from:idx_to]
            correction_y = self._gain_y*signal_y[idx_from:idx_to]

            # mpi_gatherer has also slice set list, which can be used for applying the kicks
            p_idx = self._mpi_gatherer.slice_set_list[i].particles_within_cuts
            s_idx = self._mpi_gatherer.slice_set_list[i].slice_index_of_particle.take(p_idx)

            if self._axis == 'divergence':
                b.xp[p_idx] -= correction_x[s_idx]
                b.yp[p_idx] -= correction_y[s_idx]

            elif self._axis == 'displacement':
                b.x[p_idx] -= correction_x[s_idx]
                b.y[p_idx] -= correction_y[s_idx]

        # at the end the superbunch must be rebunched. Without that the kicks do not apply to the next turn
        self._mpi_gatherer.rebunch(superbunch)

    def _normal_track(self, bunch):


        slice_set = bunch.get_slices(self._slicer, statistics=self._statistical_variables)

        signal_x = np.array([])
        signal_y = np.array([])

        if self._axis == 'divergence':
            signal_x = np.array([s for s in slice_set.mean_xp])
            signal_y = np.array([s for s in slice_set.mean_yp])

        elif self._axis == 'displacement':
            signal_x = np.array([s for s in slice_set.mean_x])
            signal_y = np.array([s for s in slice_set.mean_y])

        for processor in self._processors_x:
            signal_x = processor.process(signal_x,slice_set, None)

        for processor in self._processors_y:
            signal_y = processor.process(signal_y,slice_set, None)

        correction_x = self._gain_x*signal_x
        correction_y = self._gain_y*signal_y

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        if self._axis == 'divergence':
            bunch.xp[p_idx] -= correction_x[s_idx]
            bunch.yp[p_idx] -= correction_y[s_idx]

        elif self._axis == 'displacement':
            bunch.x[p_idx] -= correction_x[s_idx]
            bunch.y[p_idx] -= correction_y[s_idx]

# class PickUp(object):
#     """ General class for a pickup. It takes mean_x and mean_y values of each slice and pass them through signal processor
#         chains given in input parameters signal_processors_x and signal_processors_y. Note that the signals are
#         stored only to registers in the signal processor chains!
#     """
#     def __init__(self,slicer,processors_x,processors_y, phase_advance_x, phase_advance_y, mpi = False):
#         """
#         :param slicer: PyHEADTAIL slicer object
#         :param processors_x: a list of signal processors for x plane
#         :param processors_y: a list of signal processors for y plane
#         :param phase_advance_x: a location of the pickup in the units of betatron phase in x plane
#         :param phase_advance_y: a location of the pickup in the units of betatron phase in y plane
#         """
#
#         self._slicer = slicer
#
#         self._processors_x = processors_x
#         self._processors_y = processors_y
#
#         self._phase_advance_x = phase_advance_x
#         self._phase_advance_y = phase_advance_y
#
#         self._statistical_variables = None
#
#         self._mpi = mpi
#
#     def track(self,bunch):
#
#         if self._statistical_variables is None:
#             self._statistical_variables = ['mean_x', 'mean_y']
#             self._statistical_variables = get_statistical_variables(self._processors_x, self._statistical_variables)
#             self._statistical_variables = get_statistical_variables(self._processors_y, self._statistical_variables)
#
#         slice_set = bunch.get_slices(self._slicer, statistics=self._statistical_variables)
#
#         signal_x = np.array([s for s in slice_set.mean_x])
#         signal_y = np.array([s for s in slice_set.mean_y])
#
#         for processor in self._processors_x:
#             signal_x = processor.process(signal_x,slice_set,self._phase_advance_x)
#
#         for processor in self._processors_y:
#             signal_y = processor.process(signal_y,slice_set,self._phase_advance_y)
#
#
# class Kicker(object):
#     """ General class for a kicker. It takes signals from variable number of registers given in lists registers_x and
#         registers_y. The total correction kick is produced by combining those signals and passing that signal through
#         a signal processor chain (input parameters signal_processors_x and signal_processors_y) and multiplying that
#         by gain.
#
#         If the signal doesn't change in signal processors (e.g. by using only Bypass processoes), the correction kick
#         is a gain fraction of the displacement of each slice of the bunch.
#
#         In order to take into account betatron phase differences between registers and the kicker, betatron
#         phase angles (from the reference point of the accelerator) in x and y plane must be given as a parameter
#         (input parameters phase_advance_x, phase_advance_y).
#     """
#
#     def __init__(self,gain,slicer,processors_x,processors_y, phase_advance_x, phase_advance_y,
#                  registers_x,registers_y,xp_per_x, yp_per_y, mpi = False):
#         """
#         :param gain: gain coefficient for kicks. If two values are given (in tuple or list), separated values gain
#             values are used for x and y planes. If only single value is given, it is used both in x and y planes.
#         :param slicer: PyHEADTAIL slicer object for the bunch
#         :param processors_x: a list of signal processors for x plane
#         :param processors_y: a list of signal processors for y plane
#         :param phase_advance_x: location of the kicker in the units of betatron phase in x plane
#         :param phase_advance_y: location of the kicker in the units of betatron phase in y plane
#         :param registers_x: a list of register for x plane
#         :param registers_y: a list of register for y plane
#         :param xp_per_x: a conversion coefficient from displacement (x) to divergence (xp) for the signal
#         :param yp_per_y: a conversion coefficient from displacement (y) to divergence (yp) for the signal
#         """
#
#         if isinstance(gain, collections.Container):
#             self._gain_x = gain[0]
#             self._gain_y = gain[1]
#         else:
#             self._gain_x = gain
#             self._gain_y = gain
#
#         self._slicer = slicer
#
#         self._processors_x = processors_x
#         self._processors_y = processors_y
#         self._phase_advance_x = phase_advance_x
#         self._phase_advance_y = phase_advance_y
#         self._registers_x = registers_x
#         self._registers_y = registers_y
#         self._xp_per_x = xp_per_x
#         self._yp_per_y = yp_per_y
#
#         self._statistical_variables = None
#
#         self._mpi = mpi
#
#     def track(self,bunch):
#
#         if self._statistical_variables is None:
#             self._statistical_variables = ['mean_xp', 'mean_yp']
#             self._statistical_variables = get_statistical_variables(self._processors_x, self._statistical_variables)
#             self._statistical_variables = get_statistical_variables(self._processors_y, self._statistical_variables)
#
#         slice_set = bunch.get_slices(self._slicer, statistics=self._statistical_variables)
#         # Reads a particle index and a slice index for each macroparticle
#         p_idx = slice_set.particles_within_cuts
#         s_idx = slice_set.slice_index_of_particle.take(p_idx)
#
#         signal_x = self.__combine(self._registers_x,self._phase_advance_x, self._xp_per_x)
#         signal_y = self.__combine(self._registers_y,self._phase_advance_y, self._yp_per_y)
#
#         if signal_x is not None:
#             for processor in self._processors_x:
#                 signal_x = processor.process(signal_x,slice_set,self._phase_advance_x)
#
#             correction_xp = self._gain_x * signal_x
#             bunch.xp[p_idx] -= correction_xp[s_idx]
#
#         if signal_y is not None:
#             for processor in self._processors_y:
#                 signal_y = processor.process(signal_y,slice_set,self._phase_advance_y)
#
#             correction_yp = self._gain_y*signal_y
#             bunch.yp[p_idx] -= correction_yp[s_idx]
#
#
#     def __combine(self,registers,reader_phase_advance,phase_conv_coeff):
#         # This function picks signals from different registers, turns them to correct phase advance and
#         # calculates an average of them after that. Actual phase shift in betatron phase is done in a combine method
#         # written to the registers. The combine method might or might not require multiple signals (from different turns
#         # or different registers) depending on the register. Thus, two signals are givens as a argument for combine
#         # method of the register object.
#
#         total_signal = None
#         n_signals = 0
#
#         if len(registers) == 1:
#             # If there is only one register, uses signals from different turns for combination
#
#             prev_signal = None
#             for signal in registers[0]:
#                 if total_signal is None:
#                     prev_signal = signal
#                     total_signal = np.zeros(len(signal[0]))
#                 total_signal += registers[0].combine(signal, prev_signal,reader_phase_advance, True)
#                 n_signals += 1
#                 prev_signal = signal
#
#         else:
#             # if len(registers) == 2 and registers[0].combination == 'combined':
#
#             if registers[0].combination == 'combined':
#                 # If there are only two register and the combination requires signals from two register, there is only
#                 # one pair of registers
#                 prev_register = registers[0]
#                 first_iterable = 1
#             else:
#                 # In other cases, loop can go through all successive register pairs
#                 prev_register = registers[-1]
#                 first_iterable = 0
#
#             for register in registers[first_iterable:]:
#                 # print prev_register
#                 # print register
#                 for signal_1, signal_2 in itertools.izip(prev_register,register):
#                     if total_signal is None:
#                         total_signal = np.zeros(len(signal_1[0]))
#                     total_signal += prev_register.combine(signal_1,signal_2,reader_phase_advance, True)
#                     n_signals += 1
#                 prev_register = register
#
#         if total_signal is not None:
#             total_signal /= float(n_signals)
#             total_signal *= phase_conv_coeff
#
#         return total_signal
