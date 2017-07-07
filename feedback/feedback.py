import numpy as np
import collections
from PyHEADTAIL.mpi import mpi_data
from core import get_processor_variables, process, Parameters
from core import z_bins_to_bin_edges, append_bin_edges
from processors.register import VectorSumCombiner, CosineSumCombiner
from processors.register import HilbertCombiner
"""
    This file contains feedback modules for PyHEADTAIL, which can be used as interfaces between
    PyHEADTAIL and the signal processors.

    @author Jani Komppula
    @date 24/03/2017
    @copyright CERN
"""


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


def get_local_slice_sets(bunch, slicer, required_variables):
    signal_slice_sets = [bunch.get_slices(slicer, statistics=required_variables)]
    bunch_slice_sets = signal_slice_sets
    bunch_list = [bunch]

    return signal_slice_sets, bunch_slice_sets, bunch_list


def get_mpi_slice_sets(superbunch, mpi_gatherer):
    mpi_gatherer.gather(superbunch)
    signal_slice_sets = mpi_gatherer.bunch_by_bunch_data
    bunch_slice_sets = mpi_gatherer.slice_set_list
    bunch_list = mpi_gatherer.bunch_list

    return signal_slice_sets, bunch_slice_sets, bunch_list


def generate_parameters(signal_slice_sets, location=0., beta=1.):

    bin_edges = None
    segment_ref_points = []

    for slice_set in signal_slice_sets:
            edges = z_bins_to_bin_edges(slice_set.z_bins)
            segment_ref_points.append(np.mean(slice_set.z_bins))
            if bin_edges is None:
                bin_edges = np.copy(edges)
            else:
                bin_edges = append_bin_edges(bin_edges, edges)

    n_bins_per_segment = len(bin_edges)/len(signal_slice_sets)
    segment_ref_points = np.array(segment_ref_points)

    parameters = Parameters()
    parameters['class'] = 0
    parameters['bin_edges'] = bin_edges
    parameters['n_segments'] = len(signal_slice_sets)
    parameters['n_bins_per_segment'] = n_bins_per_segment
    parameters['segment_ref_points'] = segment_ref_points
    parameters['location'] = location
    parameters['beta'] = beta

    return parameters


def read_signal(signal_x, signal_y, signal_slice_sets, axis, mpi):
    # TODO: change the mpi code to support n_slices
    if mpi:
        n_slices_per_bunch = signal_slice_sets[0]._n_slices
    else:
        n_slices_per_bunch = signal_slice_sets[0].n_slices

    total_length = len(signal_slice_sets) * n_slices_per_bunch

    if (signal_x is None) or (len(signal_x) != total_length):
        raise ValueError('Wrong signal length')

    if (signal_y is None) or (len(signal_x) != total_length):
        raise ValueError('Wrong signal length')

    for idx, slice_set in enumerate(signal_slice_sets):
        idx_from = idx * n_slices_per_bunch
        idx_to = (idx + 1) * n_slices_per_bunch

        if axis == 'divergence':
            np.copyto(signal_x[idx_from:idx_to], slice_set.mean_xp)
            np.copyto(signal_y[idx_from:idx_to], slice_set.mean_yp)
        elif axis == 'displacement':
            np.copyto(signal_x[idx_from:idx_to], slice_set.mean_x)
            np.copyto(signal_y[idx_from:idx_to], slice_set.mean_y)
        else:
            raise ValueError('Unknown axis')


def kick_bunches(local_slice_sets, bunch_list, local_bunch_indexes,
                 signal_x, signal_y, axis):

    n_slices_per_bunch = local_slice_sets[0].n_slices

    for slice_set, bunch_idx, bunch in zip(local_slice_sets,
                                           local_bunch_indexes, bunch_list):

        idx_from = bunch_idx * n_slices_per_bunch
        idx_to = (bunch_idx + 1) * n_slices_per_bunch

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        if axis == 'divergence':
            if signal_x is not None:
                correction_x = np.array(signal_x[idx_from:idx_to], copy=False)
                bunch.xp[p_idx] -= correction_x[s_idx]
            if signal_y is not None:
                correction_y = np.array(signal_y[idx_from:idx_to], copy=False)
                bunch.yp[p_idx] -= correction_y[s_idx]

        elif axis == 'displacement':
            if signal_x is not None:
                correction_x = np.array(signal_x[idx_from:idx_to], copy=False)
                bunch.x[p_idx] -= correction_x[s_idx]
            if signal_y is not None:
                correction_y = np.array(signal_y[idx_from:idx_to], copy=False)
                bunch.y[p_idx] -= correction_y[s_idx]
        else:
            raise ValueError('Unknown axis')


class OneboxFeedback(object):

    def __init__(self, gain, slicer, processors_x, processors_y,
                 axis='divergence', mpi=False):

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
        if axis == 'divergence':
            self._required_variables = ['mean_xp', 'mean_yp']
        elif axis == 'displacement':
            self._required_variables = ['mean_x', 'mean_y']

        self._required_variables = get_processor_variables(self._processors_x,
                                                     self._required_variables)
        self._required_variables = get_processor_variables(self._processors_y,
                                                     self._required_variables)
        # TODO: Normally n_macroparticles_per_slice is removed from
        #       the statistical variables. Check if it is not necessary.

        self._parameters_x = None
        self._parameters_y = None
        self._signal_x = None
        self._signal_y = None

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      self._required_variables)
        self._local_bunch_indexes = None

    def track(self, bunch):
        if self._mpi:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_mpi_slice_sets(bunch, self._mpi_gatherer)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes

        else:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_local_slice_sets(bunch, self._slicer, self._required_variables)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = [0]

        if (self._parameters_x is None) or (self._signal_x is None):
            self._parameters_x = generate_parameters(signal_slice_sets)
            n_segments = self._parameters_x['n_segments']
            n_bins_per_segment = self._parameters_x['n_bins_per_segment']
            self._signal_x = np.zeros(n_segments * n_bins_per_segment)

        if (self._parameters_y is None) or (self._signal_y is None):
            self._parameters_y = generate_parameters(signal_slice_sets)
            n_segments = self._parameters_y['n_segments']
            n_bins_per_segment = self._parameters_y['n_bins_per_segment']
            self._signal_y = np.zeros(n_segments * n_bins_per_segment)


        read_signal(self._signal_x, self._signal_y, signal_slice_sets,
                    self._axis,self._mpi)

        kick_parameters_x, kick_signal_x = process(self._parameters_x,
                                                   self._signal_x,
                                                   self._processors_x,
                                                   slice_sets=signal_slice_sets)

        if kick_signal_x is not None:
            kick_signal_x = kick_signal_x * self._gain_x

        kick_parameters_y, kick_signal_y = process(self._parameters_y,
                                                   self._signal_y,
                                                   self._processors_y,
                                                   slice_sets=signal_slice_sets)
        if kick_signal_x is not None:
            kick_signal_y = kick_signal_y * self._gain_y


#        print 'signal_x: ' + str(kick_signal_x)
#        print 'self._gain_x: ' + str(self._gain_x)

        kick_bunches(bunch_slice_sets, bunch_list, self._local_bunch_indexes,
                 kick_signal_x, kick_signal_y, self._axis)

        if self._mpi:
            self._mpi_gatherer.rebunch(bunch)

class PickUp(object):
    def __init__(self, slicer, processors_x, processors_y, location_x, beta_x,
                 location_y, beta_y, mpi=False):

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y

        self._required_variables = ['mean_x', 'mean_y']

        self._required_variables = get_processor_variables(self._processors_x,
                                                     self._required_variables)
        self._required_variables = get_processor_variables(self._processors_y,
                                                     self._required_variables)
        # TODO: Normally n_macroparticles_per_slice is removed from
        #       the statistical variables. Check if it is not necessary.

        self._location_x = location_x
        self._beta_x = beta_x
        self._location_y = location_y
        self._beta_y = beta_y

        self._parameters_x = None
        self._parameters_y = None
        self._signal_x = None
        self._signal_y = None

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      self._required_variables)

        self._local_bunch_indexes = None

    def track(self, bunch):
        if self._mpi:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_mpi_slice_sets(bunch, self._mpi_gatherer)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes

        else:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_local_slice_sets(bunch, self._slicer, self._required_variables)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = [0]


        if (self._parameters_x is None) or (self._signal_x is None):
            self._parameters_x = generate_parameters(signal_slice_sets, self._location_x, self._beta_x)
            n_segments = self._parameters_x['n_segments']
            n_bins_per_segment = self._parameters_x['n_bins_per_segment']
            self._signal_x = np.zeros(n_segments * n_bins_per_segment)

        if (self._parameters_y is None) or (self._signal_y is None):
            self._parameters_y = generate_parameters(signal_slice_sets, self._location_y, self._beta_y)
            n_segments = self._parameters_y['n_segments']
            n_bins_per_segment = self._parameters_y['n_bins_per_segment']
            self._signal_y = np.zeros(n_segments * n_bins_per_segment)

        read_signal(self._signal_x, self._signal_y, signal_slice_sets,
                    'displacement',self._mpi)

        if self._signal_x is not None:
            end_parameters_x, end_signal_x = process(self._parameters_x,
                                                       self._signal_x,
                                                       self._processors_x,
                                                       slice_sets=signal_slice_sets)

        if self._signal_y is not None:
            end_parameters_y, end_signal_y = process(self._parameters_y,
                                                       self._signal_y,
                                                       self._processors_y,
                                                       slice_sets=signal_slice_sets)


class Kicker(object):
    def __init__(self, gain, slicer, processors_x, processors_y,
                 registers_x, registers_y, location_x, beta_x,
                 location_y, beta_y, combiner='vector_sum', mpi=False):

        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y

        self._parameters_x = None
        self._parameters_y = None
        self._signal_x = None
        self._signal_y = None


        if isinstance(combiner, (str,unicode)):
            if combiner == 'vector_sum':
                self._combiner_x = VectorSumCombiner(registers_x,
                                                     location_x, beta_x,
                                                     beta_conversion = '90_deg')
                self._combiner_y = VectorSumCombiner(registers_y,
                                                     location_y, beta_y,
                                                     beta_conversion = '90_deg')
            elif combiner == 'cosine_sum':
                self._combiner_x = CosineSumCombiner(registers_x,
                                                     location_x, beta_x,
                                                     beta_conversion = '90_deg')
                self._combiner_y = CosineSumCombiner(registers_y,
                                                     location_y, beta_y,
                                                     beta_conversion = '90_deg')

            elif combiner == 'hilbert':
                self._combiner_x = HilbertCombiner(registers_x,
                                                     location_x, beta_x,
                                                     beta_conversion = '90_deg')
                self._combiner_y = HilbertCombiner(registers_y,
                                                     location_y, beta_y,
                                                     beta_conversion = '90_deg')
            else:
                raise ValueError('Unknown combiner type')
        else:
            self._combiner_x = combiner(registers_x, location_x, beta_x,
                                        beta_conversion = '90_deg')
            self._combiner_y = combiner(registers_y, location_y, beta_y,
                                        beta_conversion = '90_deg')

        self._required_variables = ['mean_x', 'mean_y', 'mean_xp', 'mean_yp']
        self._required_variables = get_processor_variables(self._processors_x,
                                                     self._required_variables)
        self._required_variables = get_processor_variables(self._processors_y,
                                                     self._required_variables)
        # TODO: Normally n_macroparticles_per_slice is removed from
        #       the statistical variables. Check if it is not necessary.

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      self._required_variables)
        self._local_bunch_indexes = None

    def track(self, bunch):
        if self._mpi:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_mpi_slice_sets(bunch, self._mpi_gatherer)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes

        else:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_local_slice_sets(bunch, self._slicer, self._required_variables)
            if self._local_bunch_indexes is None:
                self._local_bunch_indexes = [0]

        parameters_x, signal_x = self._combiner_x.process()
        parameters_y, signal_y = self._combiner_y.process()

        if signal_x is not None:
            parameters_x, signal_x = process(parameters_x, signal_x,
                                             self._processors_x,
                                             slice_sets=signal_slice_sets)
            signal_x = signal_x * self._gain_x

        if signal_y is not None:
            parameters_y, signal_y = process(parameters_y, signal_y,
                                             self._processors_y,
                                             slice_sets=signal_slice_sets)
            signal_y = signal_y * self._gain_y

        kick_bunches(bunch_slice_sets, bunch_list, self._local_bunch_indexes,
                     signal_x, signal_y, 'divergence')

        if self._mpi:
            self._mpi_gatherer.rebunch(bunch)
