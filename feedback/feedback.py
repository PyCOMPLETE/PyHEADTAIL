import numpy as np
import collections
import itertools
import copy
from abc import ABCMeta, abstractmethod
import timeit
from PyHEADTAIL.mpi import mpi_data
from processors.signal import SignalParameters, BeamParameters
"""
    This file contains modules, which can be used as a feedback module/object in PyHEADTAIL. Actual signal processing is
    done by using signal processors written to files processors.py and digital_processors.py. A list of signal
    processors is given as a argument for feedback elements.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN
"""

"""
    Must be discussed:
        - turn by turn varying slice width -> will be forgot
        - varying slice width in the bunch -> is it necessary
        - future of matrix filters?
        -

"""

# TODO: add beta function

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
        if 'bunch' in processor.extensions:
            required_variables.extend(processor.required_variables)

    required_variables = list(set(required_variables))

    if 'z_bins' in required_variables:
        required_variables.remove('z_bins')

    return required_variables


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


class FeedbackMapObject(object):
    __metaclass__ = ABCMeta

    def __init__(self,slicer, processors_x, processors_y, required_variables, beam_parameters_x=None,
                 beam_parameters_y=None, mpi=False, gain_x=None, gain_y=None, extra_statistics=None):

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y

        self._gain_x = gain_x
        self._gain_y = gain_y

        self._mpi = mpi
        self._extra_statistics = extra_statistics
        self._required_variables = required_variables

        self._beam_parameters_x = beam_parameters_x
        self._beam_parameters_y = beam_parameters_y

        self._n_local_bunches = None
        self._n_total_bunches = None
        self._n_slices_per_bunch = None
        self._local_bunch_indexes = None

        self._input_signal_parameters_x = None
        self._input_signal_parameters_y = None

        # variables, which will be set each time, when the track method is called
        self._processor_slice_sets = None
        self._local_slice_sets = None
        self._bunch_list = None
        self._input_signal_x = None
        self._input_signal_y = None

    @abstractmethod
    def track(self, bunch):
        pass

    def _init_track(self,bunch):

        if self._n_total_bunches is None:
            self._init_variables(bunch)
        elif self._mpi:
            self._mpi_gatherer.gather(bunch)

        if self._mpi:
            self._processor_slice_sets = self._mpi_gatherer.bunch_by_bunch_data
            self._local_slice_sets = self._mpi_gatherer.slice_set_list
            self._bunch_list = self._mpi_gatherer.bunch_list

        else:
            self._processor_slice_sets = [bunch.get_slices(self._slicer, statistics=self._required_variables)]
            self._local_slice_sets = self._processor_slice_sets
            self._bunch_list = [bunch]

    def _init_variables(self,bunch):
        if self._extra_statistics is not None:
            self._required_variables += self._extra_statistics

        self._required_variables = get_processor_variables(self._processors_x, self._required_variables)
        self._required_variables = get_processor_variables(self._processors_y, self._required_variables)

        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer, self._required_variables)
            self._mpi_gatherer.gather(bunch)

            self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes
            processor_slice_sets = self._mpi_gatherer.bunch_by_bunch_data
            local_slice_sets = self._mpi_gatherer.slice_set_list
        else:

            if 'n_macroparticles_per_slice' in self._required_variables:
                self._required_variables.remove('n_macroparticles_per_slice')

            self._local_bunch_indexes = [0]
            processor_slice_sets = [bunch.get_slices(self._slicer, statistics=self._required_variables)]
            local_slice_sets = processor_slice_sets

        self._n_local_bunches = len(local_slice_sets)
        self._n_total_bunches = len(processor_slice_sets)
        self._n_slices_per_bunch = len(processor_slice_sets[0].z_bins) - 1

        self._input_signal_x = np.zeros(self._n_total_bunches * self._n_slices_per_bunch)
        self._input_signal_y = np.zeros(self._n_total_bunches * self._n_slices_per_bunch)

        input_bin_edges = None
        original_segment_mids = []
        for slice_set in processor_slice_sets:
            edges = np.transpose(np.array([slice_set.z_bins[:-1], slice_set.z_bins[1:]]))
            original_segment_mids.append(np.mean(slice_set.z_bins))
            if input_bin_edges is None:
                input_bin_edges = np.copy(edges)
            else:
                input_bin_edges = np.append(input_bin_edges,edges, axis=0)

        self._input_signal_parameters_x = SignalParameters(0,input_bin_edges,len(processor_slice_sets),
                                                           int(len(input_bin_edges)/len(processor_slice_sets)),
                                                           original_segment_mids,self._beam_parameters_x)
        self._input_signal_parameters_y = SignalParameters(0,input_bin_edges,len(processor_slice_sets),
                                                           int(len(input_bin_edges)/len(processor_slice_sets)),
                                                           original_segment_mids,
                                                           self._beam_parameters_y)

    def _read_signal(self, attr_x, attr_y):

        for idx, slice_set in enumerate(self._processor_slice_sets):
            idx_from = idx * self._n_slices_per_bunch
            idx_to = (idx + 1) * self._n_slices_per_bunch

            np.copyto(self._input_signal_x[idx_from:idx_to],getattr(slice_set, attr_x))
            np.copyto(self._input_signal_y[idx_from:idx_to],getattr(slice_set, attr_y))

    def _process_signal(self):
        # TODO: check signal classes

        signal_x = np.copy(self._input_signal_x)
        signal_y = np.copy(self._input_signal_y)

        if self._input_signal_x is not None:
            signal_parameters_x = copy.copy(self._input_signal_parameters_x)
            for processor in self._processors_x:
                signal_parameters_x, signal_x = processor.process(signal_parameters_x, signal_x,slice_sets = self._processor_slice_sets)
        else:
            print 'Warning: Correction signal in x-plane is None'

        if self._input_signal_y is not None:
            signal_parameters_y = copy.copy(self._input_signal_parameters_y)
            for processor in self._processors_y:
                signal_parameters_y, signal_y = processor.process(signal_parameters_y, signal_y,slice_sets = self._processor_slice_sets)
        else:
            print 'Warning: Correction signal in y-plane is None'

        return signal_x, signal_y


    def _do_kick(self,bunch,signal_x,signal_y,attr_x, attr_y):

        for local_idx, (bunch_idx, sub_bunch) in enumerate(zip(self._local_bunch_indexes,self._bunch_list)):

            # the slice set data from all bunches in all processors pass the signal processors. Here, the correction
            # signals for the bunches tracked in this processors are picked by using indexes found from
            # mpi_gatherer.total_data.local_data_locations
            idx_from = bunch_idx * self._n_slices_per_bunch
            idx_to = (bunch_idx + 1) * self._n_slices_per_bunch


            # mpi_gatherer has also slice set list, which can be used for applying the kicks
            p_idx = self._local_slice_sets[local_idx].particles_within_cuts
            s_idx = self._local_slice_sets[local_idx].slice_index_of_particle.take(p_idx)

            if self._input_signal_x is not None:
                correction_x = self._gain_x*signal_x[idx_from:idx_to]
                particle_coordinates_x = getattr(sub_bunch,attr_x)
                particle_coordinates_x[p_idx] -= correction_x[s_idx]
                setattr(sub_bunch,attr_x,particle_coordinates_x)

            if self._input_signal_y is not None:
                correction_y = self._gain_y*signal_y[idx_from:idx_to]
                particle_coordinates_y = getattr(sub_bunch,attr_y)
                particle_coordinates_y[p_idx] -= correction_y[s_idx]
                setattr(sub_bunch,attr_y,particle_coordinates_y)

        if self._mpi:
            # at the end the superbunch must be rebunched. Without that the kicks do not apply to the next turn
            self._mpi_gatherer.rebunch(bunch)

class OneboxFeedback(FeedbackMapObject):

    def __init__(self,gain, slicer, processors_x, processors_y, axis='divergence', **kwargs):

        beam_parameters_x = BeamParameters(0.,1.)
        beam_parameters_y = BeamParameters(0.,1.)

        if isinstance(gain, collections.Container):
            gain_x = gain[0]
            gain_y = gain[1]
        else:
            gain_x = gain
            gain_y = gain

        if axis == 'divergence':
            self._slice_attr_x = 'mean_xp'
            self._slice_attr_y = 'mean_yp'
            self._particle_attr_x = 'xp'
            self._particle_attr_y = 'yp'
        elif axis == 'displacement':
            self._slice_attr_x = 'mean_x'
            self._slice_attr_y = 'mean_y'
            self._particle_attr_x = 'x'
            self._particle_attr_y = 'y'
        else:
            raise ValueError('Unknown input value for axis in OneboxFeedback')

        required_variables = [self._slice_attr_x,self._slice_attr_y]

        super(self.__class__, self).__init__(slicer, processors_x, processors_y, required_variables,
                 gain_x=gain_x, gain_y=gain_y, beam_parameters_x=beam_parameters_x,
                 beam_parameters_y=beam_parameters_y, **kwargs)


    def track(self, bunch):
        self._init_track(bunch)
        self._read_signal(self._slice_attr_x, self._slice_attr_y)
        signal_x, signal_y = self._process_signal()
        self._do_kick(bunch,signal_x,signal_y,self._particle_attr_x, self._particle_attr_y)


class PickUp(FeedbackMapObject):
    def __init__(self, slicer, processors_x, processors_y, beam_parameters_x, beam_parameters_y, **kwargs):
        self._slice_attr_x = 'mean_x'
        self._slice_attr_y = 'mean_y'

        required_variables = [self._slice_attr_x,self._slice_attr_y]

        super(self.__class__, self).__init__(slicer, processors_x, processors_y, required_variables,
                                             beam_parameters_x = beam_parameters_x, beam_parameters_y = beam_parameters_y,
                                             **kwargs)

    def track(self, bunch):
        self._init_track(bunch)
        self._read_signal(self._slice_attr_x, self._slice_attr_y)
        self._process_signal()


class Kicker(FeedbackMapObject):
    def __init__(self,gain,slicer,processors_x,processors_y,
                 registers_x,registers_y, beam_parameters_x, beam_parameters_y, **kwargs):

        if isinstance(gain, collections.Container):
            gain_x = gain[0]
            gain_y = gain[1]
        else:
            gain_x = gain
            gain_y = gain

        self._particle_attr_x = 'xp'
        self._particle_attr_y = 'yp'

        required_variables = []

        self._registers_x = registers_x
        self._registers_y = registers_y

        super(self.__class__, self).__init__(slicer, processors_x, processors_y, required_variables, gain_x = gain_x,
                                             gain_y = gain_y, beam_parameters_x = beam_parameters_x,
                                             beam_parameters_y = beam_parameters_y, **kwargs)

        # FIXME: This is ugly way
        self._first_kick_x = True
        self._first_kick_y = True

    def track(self, bunch):
        self._init_track(bunch)

        self._input_signal_x = self.__combine(self._registers_x,self._beam_parameters_x)
        self._input_signal_y = self.__combine(self._registers_y,self._beam_parameters_y)

        if self._first_kick_x and (self._input_signal_x is not None):
            self._first_kick_x = False
            self._input_signal_parameters_x = copy.copy(self._registers_x[0].signal_parameters)

        if self._first_kick_y and (self._input_signal_y is not None):
            self._first_kick_y = False
            self._input_signal_parameters_y = copy.copy(self._registers_y[0].signal_parameters)

        signal_x, signal_y = self._process_signal()
        self._do_kick(bunch,signal_x,signal_y,self._particle_attr_x, self._particle_attr_y)

    def __combine(self,registers,beam_parameters):
        # This function picks signals from different registers, turns them to correct phase advance and
        # calculates an average of them after that. Actual phase shift in betatron phase is done in a combine method
        # written to the registers. The combine method might or might not require multiple signals (from different turns
        # or different registers) depending on the register. Thus, two signals are givens as a argument for combine
        # method of the register object.

        reader_phase_advance = beam_parameters.phase_advance

        total_signal = None
        n_signals = 0

        if len(registers) == 1:
            # If there is only one register, uses signals from different turns for combination

            prev_signal = None
            for signal in registers[0]:
                if total_signal is None:
                    prev_signal = signal
                    total_signal = np.zeros(len(signal[0]))
                phase_conv_coeff = 1. / np.sqrt(beam_parameters.beta_function * registers[0].beam_parameters.beta_function)
                total_signal += phase_conv_coeff*registers[0].combine(signal, prev_signal,reader_phase_advance, True)
                n_signals += 1
                prev_signal = signal

        else:
            # if len(registers) == 2 and registers[0].combination == 'combined':

            if registers[0].combination == 'combined':
                # If there are only two register and the combination requires signals from two register, there is only
                # one pair of registers
                prev_register = registers[0]
                first_iterable = 1
            else:
                # In other cases, loop can go through all successive register pairs
                prev_register = registers[-1]
                first_iterable = 0

            for register in registers[first_iterable:]:
                # print prev_register
                # print register
                phase_conv_coeff = 1. / np.sqrt(beam_parameters.beta_function * prev_register.beam_parameters.beta_function)
                for signal_1, signal_2 in itertools.izip(prev_register,register):
                    if total_signal is None:
                        total_signal = np.zeros(len(signal_1[0]))
                    total_signal += phase_conv_coeff*prev_register.combine(signal_1,signal_2,reader_phase_advance, True)
                    n_signals += 1
                prev_register = register

        if total_signal is not None:
            total_signal /= float(n_signals)

        return total_signal
