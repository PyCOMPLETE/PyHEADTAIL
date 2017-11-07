import numpy as np
import collections
from PyHEADTAIL.mpi import mpi_data
from core import get_processor_variables, process, Parameters
from core import z_bins_to_bin_edges, append_bin_edges
from processors.register import VectorSumCombiner, CosineSumCombiner
from processors.register import HilbertCombiner, DummyCombiner
from scipy.constants import c
"""
    This file contains objecst, which can be used as transverse feedback
    systems in the one turn map in PyHEADTAIL. The signal processing in the
    feedback systems can be modelled by giving a list of the necessary signal
    processors describing the system to the objects.

    @author Jani Komppula
    @date 11/10/2017
"""


class IdealBunchFeedback(object):
    """ The simplest possible feedback. It corrects a gain fraction of a mean xp/yp value of the bunch.
    """
    def __init__(self,gain, multi_bunch=False):
        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain
            
        self.multi_bunch = multi_bunch

    def track(self,bunch):
        
        if self.multi_bunch:
            bunch_list = bunch.split_to_views()
            
            for b in bunch_list:
                b.xp -= self._gain_x *b.mean_xp()
                b.yp -= self._gain_y*b.mean_yp()
        else:
            bunch.xp -= self._gain_x *bunch.mean_xp()
            bunch.yp -= self._gain_y*bunch.mean_yp()


class IdealSliceFeedback(object):
    """Corrects a gain fraction of a mean xp/yp value of each slice in the bunch."""
    def __init__(self,gain,slicer, multi_bunch=False):
        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain
            
        self.multi_bunch = multi_bunch

        self._slicer = slicer

    def track(self,bunch):
        
        if self.multi_bunch:
            bunch_list = bunch.split_to_views()
            
            for b in bunch_list:
                slice_set = b.get_slices(self._slicer, statistics = ['mean_xp', 'mean_yp'])
                p_idx = slice_set.particles_within_cuts
                s_idx = slice_set.slice_index_of_particle.take(p_idx)
        
                b.xp[p_idx] -= self._gain_x * slice_set.mean_xp[s_idx]
                b.yp[p_idx] -= self._gain_y * slice_set.mean_yp[s_idx]
        
        else:
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


def generate_parameters(signal_slice_sets, location=0., beta=1.,
                        circumference=None, h_bunch=None):

    bin_edges = None
    segment_ref_points = []

    for slice_set in signal_slice_sets:
            z_bins = np.copy(slice_set.z_bins)
            if circumference is not None:
                z_bins -= slice_set.bucket_id*circumference/float(h_bunch)
        
            edges = -1.*z_bins_to_bin_edges(z_bins)/c
            segment_ref_points.append(-1.*np.mean(z_bins)/c)
            if bin_edges is None:
                bin_edges = np.copy(edges)
            else:
                bin_edges = append_bin_edges(bin_edges, edges)

    bin_edges = bin_edges[::-1]
    bin_edges = np.fliplr(bin_edges)
    segment_ref_points = segment_ref_points[::-1]

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

def read_signal(signal_x, signal_y, signal_slice_sets, axis, mpi,
                phase_x, phase_y, beta_x, beta_y):
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
            if phase_x is None:
                np.copyto(signal_x[idx_from:idx_to], slice_set.mean_xp)
            else:
                np.copyto(signal_x[idx_from:idx_to], (-np.sin(phase_x)*slice_set.mean_x/beta_x +
                                  np.cos(phase_x)*slice_set.mean_xp))
            if phase_y is None:
                np.copyto(signal_y[idx_from:idx_to], slice_set.mean_yp)
            else:
                np.copyto(signal_y[idx_from:idx_to], (-np.sin(phase_y)*slice_set.mean_y/beta_y +
                                  np.cos(phase_y)*slice_set.mean_yp))

        elif axis == 'displacement':
            if phase_x is None:
                np.copyto(signal_x[idx_from:idx_to], slice_set.mean_x)
            else:
                np.copyto(signal_x[idx_from:idx_to], (np.cos(phase_x)*slice_set.mean_x +
                                  beta_x*np.sin(phase_x)*slice_set.mean_xp))
            if phase_y is None:
                np.copyto(signal_y[idx_from:idx_to], slice_set.mean_y)
            else:
                np.copyto(signal_y[idx_from:idx_to], (np.cos(phase_y)*slice_set.mean_y +
                                  beta_y*np.sin(phase_y)*slice_set.mean_yp))
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
    """ An transverse feedback object for the one turn map in PyHEADTAIL.

    By using this object, the pickup and the kicker are in the same location
    of the accelerator. Bandwidth limitations, turn delays, noise, etc can be
    applied by using signal processors. The axises for the pickup signal and
    the correction are by default same, but they can be also specified to be
    different (e.g. displacement and divergence).
    """

    def __init__(self, gain, slicer, processors_x, processors_y,
                 pickup_axis='divergence', kicker_axis=None, mpi=False,
                 phase_x=None, phase_y=None, beta_x=1., beta_y=1.,
                 circumference=None, h_bunch=None):
        """
        Parameters
        ----------
        gain : float or tuple
            A fraction of the oscillations is corrected, when the perfectly
            betatron motion corrected pickup signal by passes the signal
            processors without modifications, i.e. 2/(damping time [turns]).
            Separate values can be set to x and y planes by giving two values
            in a tuple.
        slicer : PyHEADTAIL slicer object
        processors_x : list
            A list of signal processors for the x-plane
        processors_y : list
            A list of signal processors for the y-plane
        pickup_axis : str
            A axis, which values are used as a pickup signal
        kicker_axis : str
            A axis, to which the correction is applied. If None, the axis is
            same as the pickup axis
        mpi : bool
            If True, data from multiple bunches are gathered by using MPI
        phase_x : float
            Initial betatron phase rotation for the signal in x-plane in the
            units of radians
        phase_y : float
            Initial betatron phase rotation for the signal in y-plane in the
            units of radians
        beta_x : float
            A value of the x-plane beta function in the feedback location
        beta_y : float
            A value of the y-plane beta function in the feedback location
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

        self._phase_x = phase_x
        self._phase_y = phase_y

        self._beta_x = beta_x
        self._beta_y = beta_y
        
        if mpi:
            if (circumference is None) or (h_bunch is None):
                raise ValueError("""Both circumference and h_bunch must be
                                 given if the feedback module is used in the
                                 mpi mode""")
        self._circumference = circumference
        self._h_bunch = h_bunch


        self._pickup_axis = pickup_axis
        if kicker_axis is None:
            self._kicker_axis = pickup_axis
        else:
            self._kicker_axis = kicker_axis

        self._required_variables = []
        if (self._pickup_axis == 'divergence') or \
            (self._kicker_axis == 'divergence') or \
            (phase_x is not None) or \
            (phase_y is not None):
#            print 'I am adding divergence parameters!'

            self._required_variables.append('mean_xp')
            self._required_variables.append('mean_yp')

        if (self._pickup_axis == 'displacement') or \
            (self._kicker_axis == 'displacement') or \
            (phase_x is not None) or \
            (phase_y is not None):
#            print 'I am adding displacement parameters!'

            self._required_variables.append('mean_x')
            self._required_variables.append('mean_y')

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
            self._parameters_x = generate_parameters(signal_slice_sets,
                                                     circumference=self._circumference,
                                                     h_bunch=self._h_bunch)
            n_segments = self._parameters_x['n_segments']
            n_bins_per_segment = self._parameters_x['n_bins_per_segment']
            self._signal_x = np.zeros(n_segments * n_bins_per_segment)

        if (self._parameters_y is None) or (self._signal_y is None):
            self._parameters_y = generate_parameters(signal_slice_sets,
                                                     circumference=self._circumference,
                                                     h_bunch=self._h_bunch)
            n_segments = self._parameters_y['n_segments']
            n_bins_per_segment = self._parameters_y['n_bins_per_segment']
            self._signal_y = np.zeros(n_segments * n_bins_per_segment)


        read_signal(self._signal_x, self._signal_y, signal_slice_sets,
                    self._pickup_axis,self._mpi, self._phase_x, self._phase_y,
                    self._beta_x, self._beta_y)

        kick_parameters_x, kick_signal_x = process(self._parameters_x,
                                                   self._signal_x,
                                                   self._processors_x,
                                                   slice_sets=signal_slice_sets)

        if kick_signal_x is not None:
            kick_signal_x = kick_signal_x * self._gain_x

            if self._pickup_axis == 'displacement' and self._kicker_axis == 'divergence':
                kick_signal_x = kick_signal_x / self._beta_x
            elif self._pickup_axis == 'divergence' and self._kicker_axis == 'displacement':
                kick_signal_x = kick_signal_x * self._beta_x

        kick_parameters_y, kick_signal_y = process(self._parameters_y,
                                                   self._signal_y,
                                                   self._processors_y,
                                                   slice_sets=signal_slice_sets)
        if kick_signal_x is not None:
            kick_signal_y = kick_signal_y * self._gain_y

            if self._pickup_axis == 'displacement' and self._kicker_axis == 'divergence':
                kick_signal_y = kick_signal_y / self._beta_y
            elif self._pickup_axis == 'divergence' and self._kicker_axis == 'displacement':
                kick_signal_y = kick_signal_y * self._beta_y


#        print 'signal_x: ' + str(kick_signal_x)
#        print 'self._gain_x: ' + str(self._gain_x)

        kick_bunches(bunch_slice_sets, bunch_list, self._local_bunch_indexes,
                 kick_signal_x, kick_signal_y, self._kicker_axis)

        if self._mpi:
            self._mpi_gatherer.rebunch(bunch)

class PickUp(object):
    """ A pickup object for the one turn map in PyHEADTAIL.

    This object can be used as a pickup in the trasverse feedback systems
    consisting of separate pickup(s) and kicker(s). A model for signal
    processing (including, for example, bandwidth limitations and noise) can be
    implemented by using signal processors. The signal can be transferred to
    kicker(s) by putting registers to the signal processor chains.
    """

    def __init__(self, slicer, processors_x, processors_y, location_x, beta_x,
                 location_y, beta_y, mpi=False, phase_x=None, phase_y=None,
                 circumference=None, h_bunch=None):
        """
        Parameters
        ----------
        slicer : PyHEADTAIL slicer object
        processors_x : list
            A list of signal processors for the x-plane
        processors_y : list
            A list of signal processors for the y-plane
            used as a signal source in the y-plane
        location_x : float
            A location of the pickup in x-plane in the units of betatron phase
            advance from a chosen reference point
        beta_x : float
            A value of the x-plane beta function in the pickup location
        location_y : float
            A location of the pickup in y-plane in the units of betatron phase
            advance from a chosen reference point
        beta_y : float
            A value of the y-plane beta function in the pickup location
        mpi : bool
            If True, data from multiple bunches are gathered by using MPI
        phase_x : float
            Initial betatron phase rotation of the signal in x-plane in the
            units of radians
        phase_y : float
            Initial betatron phase rotation of the signal in y-plane in the
            units of radians
        """

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y

        self._phase_x = phase_x
        self._phase_y = phase_y

        self._required_variables = ['mean_x', 'mean_y']
        if (phase_x is not None) or (phase_x is not None):
            self._required_variables.append('mean_xp')
            self._required_variables.append('mean_yp')

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
        
        if mpi:
            if (circumference is None) or (h_bunch is None):
                raise ValueError("""Both circumference and h_bunch must be
                                 given if the feedback module is used in the
                                 mpi mode""")
        self._circumference = circumference
        self._h_bunch = h_bunch

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
            self._parameters_x = generate_parameters(signal_slice_sets,
                                                     self._location_x,
                                                     self._beta_x,
                                                     self._circumference,
                                                     self._h_bunch)
            n_segments = self._parameters_x['n_segments']
            n_bins_per_segment = self._parameters_x['n_bins_per_segment']
            self._signal_x = np.zeros(n_segments * n_bins_per_segment)

        if (self._parameters_y is None) or (self._signal_y is None):
            self._parameters_y = generate_parameters(signal_slice_sets,
                                                     self._location_y,
                                                     self._beta_y,
                                                     self._circumference,
                                                     self._h_bunch)
            n_segments = self._parameters_y['n_segments']
            n_bins_per_segment = self._parameters_y['n_bins_per_segment']
            self._signal_y = np.zeros(n_segments * n_bins_per_segment)

        read_signal(self._signal_x, self._signal_y, signal_slice_sets,
                    'displacement', self._mpi, self._phase_x, self._phase_y,
                    self._beta_x, self._beta_y)

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
    """ A Kicker object for the one turn map in PyHEADTAIL.

    This object can be used as a kicker in the trasverse feedback systems
    consisting of separate pickup(s) and kicker(s). A model for signal
    processing (including, for example, bandwidth limitations and noise) can be
    implemented by using signal processors. The input signals for the kicker
    are the lists of register objects given as a input paramter.
    """
    def __init__(self, gain, slicer, processors_x, processors_y,
                 registers_x, registers_y, location_x, beta_x,
                 location_y, beta_y, combiner='vector_sum', mpi=False):
        """
        Parameters
        ----------
        gain : float or tuple
            A fraction of the oscillations is corrected, when the perfectly
            betatron motion corrected pickup signal by passes the signal
            processors without modifications, i.e. 2/(damping time [turns]).
            Separate values can be set to x and y planes by giving two values
            in a tuple.
        slicer : PyHEADTAIL slicer object
        processors_x : list
            A list of signal processors for the x-plane
        processors_y : list
            A list of signal processors for the y-plane
        registers_x : list
            A list of register object(s) (from pickup(s) processor chain(s)
            used as a signal source in the x-plane
        registers_y : list
            A list of register object(s) (from pickup(s) processor chain(s)
            used as a signal source in the y-plane
        location_x : float
            A location of the kicker in x-plane in the units of betatron phase
            advance from a chosen reference point
        beta_x : float
            A value of the x-plane beta function in the kicker location
        location_y : float
            A location of the kicker in y-plane in the units of betatron phase
            advance from a chosen reference point
        beta_y : float
            A value of the y-plane beta function in the kicker location
        combiner : string or object
            A combiner, which is used for combining signals from
            the registers.
        mpi : bool
            If True, data from multiple bunches are gathered by using MPI
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

            elif combiner == 'dummy':
                self._combiner_x = DummyCombiner(registers_x,
                                                     location_x, beta_x,
                                                     beta_conversion = '90_deg')
                self._combiner_y = DummyCombiner(registers_y,
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
