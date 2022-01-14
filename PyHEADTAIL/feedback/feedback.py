import collections
import numpy as np
from scipy.constants import c

from PyHEADTAIL.mpi import mpi_data
from .core import get_processor_variables, process, Parameters
from .core import z_bins_to_bin_edges, append_bin_edges
from .processors.register import VectorSumCombiner, CosineSumCombiner
from .processors.register import HilbertCombiner, DummyCombiner
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



class GenericOneTurnMapObject(object):
    def __init__(self, gain, slicer, processors_x, processors_y=None,
                 pickup_axis='divergence', kicker_axis=None, mpi=False,
                 phase_x=None, phase_y=None, location_x=0., location_y=0.,
                 beta_x=1., beta_y=1., **kwargs):
        
        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain
        
        self._slicer = slicer
        
        self._processors_x = processors_x
        self._processors_y = processors_y
        
        # beam parameters
        self._pickup_axis = pickup_axis
        self._kicker_axis = kicker_axis
        self._phase_x = phase_x
        self._phase_y = phase_y
        self._location_x = location_x
        self._location_y = location_y
        self._beta_x = beta_x
        self._beta_y = beta_y
        
        self._local_sets = None
        self._signal_sets_x = None
        self._signal_sets_y = None
        self._loc_signal_sets_x = None
        self._loc_signal_sets_y = None
        self._required_variables = []
        
        if (self._pickup_axis == 'divergence') or (phase_x is not None):
            self._required_variables.append('mean_xp')
        if (self._pickup_axis == 'displacement') or (phase_x is not None):
            self._required_variables.append('mean_x')

        self._required_variables = get_processor_variables(self._processors_x,
                                                     self._required_variables)
        if self._processors_y is not None:
            if (self._pickup_axis == 'divergence') or (phase_y is not None):
                self._required_variables.append('mean_yp')
            if (self._pickup_axis == 'displacement') or (phase_y is not None):
                self._required_variables.append('mean_y')
                
            self._required_variables = get_processor_variables(self._processors_y,
                                                         self._required_variables)
#        # TODO: Normally n_macroparticles_per_slice is removed from
#        #       the statistical variables. Check if it is not necessary.

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      self._required_variables)
        self._parameters_x = None
        self._signal_x = None
        
        self._parameters_y = None
        self._signal_y = None
        

    def _init_signals(self, bunch_list, signal_slice_sets_x, signal_slice_sets_y):
        
        self._parameters_x = self._generate_parameters(signal_slice_sets_x,
                                                        self._location_x,
                                                        self._beta_x)
        
        n_segments = self._parameters_x['n_segments']
        n_bins_per_segment = self._parameters_x['n_bins_per_segment']
        self._signal_x = np.zeros(n_segments * n_bins_per_segment)
        
        
        if self._processors_y is not None:
            self._parameters_y = self._generate_parameters(signal_slice_sets_y,
                                                            self._location_y,
                                                            self._beta_y)
        
            n_segments = self._parameters_y['n_segments']
            n_bins_per_segment = self._parameters_y['n_bins_per_segment']
            self._signal_y = np.zeros(n_segments * n_bins_per_segment)

    def _get_slice_sets(self, superbunch):
        if self._mpi:
            self._mpi_gatherer.gather(superbunch)
            all_slice_sets = self._mpi_gatherer.bunch_by_bunch_data
            local_slice_sets = self._mpi_gatherer.slice_set_list
            bunch_list = self._mpi_gatherer.bunch_list
            self._local_sets = self._mpi_gatherer.local_bunch_indexes
        else:
            all_slice_sets = [superbunch.get_slices(self._slicer,
                                                    statistics=self._required_variables)]
            local_slice_sets = all_slice_sets
            bunch_list = [superbunch]
            self._local_sets = [0]
            
        if self._signal_sets_x is None:
            indexes = self._parse_relevant_bunches(local_slice_sets,
                                                   all_slice_sets,
                                                   self._processors_x)
            self._signal_sets_x = indexes[0]
            self._loc_signal_sets_x = indexes[1]
            
            if self._processors_y is not None:
                indexes = self._parse_relevant_bunches(local_slice_sets,
                                                       all_slice_sets,
                                                       self._processors_y)
                self._signal_sets_y = indexes[0]
                self._loc_signal_sets_y = indexes[1]
            
        signal_slice_sets_x = []
        for idx in self._signal_sets_x:
            signal_slice_sets_x.append(all_slice_sets[idx])
        
        if self._processors_y is not None:
            signal_slice_sets_y = []
            for idx in self._signal_sets_y:
                signal_slice_sets_y.append(all_slice_sets[idx])
        else:
            signal_slice_sets_y = None
        
        return bunch_list, local_slice_sets, signal_slice_sets_x, signal_slice_sets_y
            
    def _generate_parameters(self, signal_slice_sets, location=0., beta=1.):
    
        bin_edges = None
        segment_ref_points = []
    
        circumference = signal_slice_sets[0].circumference
        h_bunch = signal_slice_sets[0].h_bunch
            
        for slice_set in signal_slice_sets:
                z_bins = np.copy(slice_set.z_bins)
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
    
    def _parse_relevant_bunches(self, local_slice_sets, all_slice_sets, processors):

        circumference = all_slice_sets[0].circumference
        h_bunch = all_slice_sets[0].h_bunch
        time_scale = 0.
        
        for processor in processors:
            if processor.time_scale > time_scale:
                time_scale = processor.time_scale
        local_set_edges = np.zeros((len(local_slice_sets), 2))
        
        included_sets = []
        set_is_included = np.zeros(len(all_slice_sets), dtype=int)
        set_counter = np.zeros(len(all_slice_sets), dtype=int)
        
        for i, slice_set in enumerate(local_slice_sets):
            local_set_edges[i,0] = np.min(slice_set.z_bins-slice_set.bucket_id*circumference/float(h_bunch))/c
            local_set_edges[i,1] = np.max(slice_set.z_bins-slice_set.bucket_id*circumference/float(h_bunch))/c
            
        
        local_min = np.min(local_set_edges)
        local_max = np.max(local_set_edges)
        
        counter = 0
        for i, slice_set in enumerate(all_slice_sets):
            set_min = np.min(slice_set.z_bins-slice_set.bucket_id*circumference/float(h_bunch))/c
            set_max = np.max(slice_set.z_bins-slice_set.bucket_id*circumference/float(h_bunch))/c
    #        print 'set_min ' + str(set_min) + ' and (local_min - time_scale)' + str((local_min - time_scale))
            if (set_max > (local_min - time_scale)) and (set_min < (local_max + time_scale)):
                included_sets.append(i)
                set_is_included[i] = 1
                set_counter[i] = counter
                counter += 1
            else:
                pass
    #            print('skip!!!')
                
        local_sets = []
        for idx in self._local_sets:
            if set_is_included[idx] != 1:
                raise ValueError('All local bunches are not included!')
            else:
                local_sets.append(set_counter[idx])
                    
        
        return included_sets, local_sets
    
    
    def _read_signal(self, signal, signal_slice_sets, plane, betatron_phase,
                    beta_value):     
        if self._mpi:
            n_slices_per_bunch = signal_slice_sets[0]._n_slices
        else:
            n_slices_per_bunch = signal_slice_sets[0].n_slices
    
        total_length = len(signal_slice_sets) * n_slices_per_bunch
    
        if (signal is None) or (len(signal) != total_length):
            raise ValueError('Wrong signal length')
    
        for idx, slice_set in enumerate(signal_slice_sets):
            idx_from = idx * n_slices_per_bunch
            idx_to = (idx + 1) * n_slices_per_bunch
    
            
            if plane == 'x':
                if self._pickup_axis == 'displacement' or (betatron_phase is not None):
                    x_values = np.copy(slice_set.mean_x)
                if (self._pickup_axis == 'divergence') or (betatron_phase is not None):
                    xp_values = np.copy(slice_set.mean_xp)
            elif plane == 'y':
                if self._pickup_axis == 'displacement' or (betatron_phase is not None):
                    x_values = np.copy(slice_set.mean_y)
                if (self._pickup_axis == 'divergence') or (betatron_phase is not None):
                    xp_values = np.copy(slice_set.mean_yp)
            else:
                raise ValueError('Unknown plane')
    
            if self._pickup_axis == 'divergence': 
                if betatron_phase is None:
                    np.copyto(signal[idx_from:idx_to], xp_values)
                else:
                    np.copyto(signal[idx_from:idx_to], (-np.sin(betatron_phase)*x_values/beta_value +
                                      np.cos(betatron_phase)*xp_values))
            elif self._pickup_axis == 'displacement':
                if betatron_phase is None:
                    np.copyto(signal[idx_from:idx_to], x_values)
                else:
                    np.copyto(signal[idx_from:idx_to], (np.cos(betatron_phase)*x_values +
                                      beta_value*np.sin(betatron_phase)*xp_values))
            else:
                raise ValueError('Unknown axis')
        
        if signal is not None:
            np.copyto(signal, signal[::-1])

    
    def _kick_bunches(self, signal, plane, local_slice_sets, bunch_list, local_sets):
    
        if signal is not None:
            np.copyto(signal, signal[::-1])
        
            n_slices_per_bunch = local_slice_sets[0].n_slices
        
            for slice_set, bunch_idx, bunch in zip(local_slice_sets,
                                                   local_sets, bunch_list):
                idx_from = bunch_idx * n_slices_per_bunch
                idx_to = (bunch_idx + 1) * n_slices_per_bunch
        
                p_idx = slice_set.particles_within_cuts
                s_idx = slice_set.slice_index_of_particle.take(p_idx)
        
                if self._kicker_axis == 'divergence':
                    if plane == 'x':
                        correction_x = np.array(signal[idx_from:idx_to], copy=False)
                        bunch.xp[p_idx] -= correction_x[s_idx]
                    elif plane == 'y':
                        correction_y = np.array(signal[idx_from:idx_to], copy=False)
                        bunch.yp[p_idx] -= correction_y[s_idx]
                    else:
                        raise ValueError('Unknown plane')
        
                elif self._kicker_axis == 'displacement':
                    if plane == 'x':
                        correction_x = np.array(signal[idx_from:idx_to], copy=False)
                        bunch.x[p_idx] -= correction_x[s_idx]
                    elif plane == 'y':
                        correction_y = np.array(signal[idx_from:idx_to], copy=False)
                        bunch.y[p_idx] -= correction_y[s_idx]
                    else:
                        raise ValueError('Unknown plane')
                else:
                    raise ValueError('Unknown axis')
    
class OneboxFeedback(GenericOneTurnMapObject):
    """ An transverse feedback object for the one turn map in PyHEADTAIL.

    By using this object, the pickup and the kicker are in the same location
    of the accelerator. Bandwidth limitations, turn delays, noise, etc can be
    applied by using signal processors. The axises for the pickup signal and
    the correction are by default same, but they can be also specified to be
    different (e.g. displacement and divergence).
    """

    def __init__(self, gain, slicer, processors_x, processors_y,
                 pickup_axis='divergence', kicker_axis=None, mpi=False,
                 phase_x=None, phase_y=None, beta_x=1., beta_y=1., **kwargs):
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
        
        if kicker_axis is None:
            kicker_axis = pickup_axis
        
        super(self.__class__, self).__init__(gain, slicer, processors_x,
             processors_y=processors_y, pickup_axis=pickup_axis,
             kicker_axis=kicker_axis, mpi=mpi, phase_x=phase_x,
             phase_y=phase_y, beta_x=beta_x, beta_y=beta_y, **kwargs)

    def track(self, bunch):
        
        bunch_list, local_slice_sets, signal_slice_sets_x, signal_slice_sets_y = self._get_slice_sets(bunch)
        
        if self._signal_x is None:
            self._init_signals(bunch_list, signal_slice_sets_x, signal_slice_sets_y)

        self._read_signal(self._signal_x, signal_slice_sets_x, 'x',
                           self._phase_x, self._beta_x)   
        
        kick_parameters_x, kick_signal_x = process(self._parameters_x,
                                                   self._signal_x,
                                                   self._processors_x,
                                                   slice_sets=signal_slice_sets_x)
        
        if kick_signal_x is not None:
            kick_signal_x = kick_signal_x * self._gain_x

            if self._pickup_axis == 'displacement' and self._kicker_axis == 'divergence':
                kick_signal_x = kick_signal_x / self._beta_x
            elif self._pickup_axis == 'divergence' and self._kicker_axis == 'displacement':
                kick_signal_x = kick_signal_x * self._beta_x
                
        self._kick_bunches(kick_signal_x, 'x', local_slice_sets, bunch_list,
                            self._loc_signal_sets_x)
        
        if self._processors_y is not None:

            self._read_signal(self._signal_y, signal_slice_sets_y, 'y',
                               self._phase_y, self._beta_y)   
            
            kick_parameters_y, kick_signal_y = process(self._parameters_y,
                                                       self._signal_y,
                                                       self._processors_y,
                                                       slice_sets=signal_slice_sets_y)
            
            if kick_signal_y is not None:
                kick_signal_y = kick_signal_y * self._gain_y
    
                if self._pickup_axis == 'displacement' and self._kicker_axis == 'divergence':
                    kick_signal_y = kick_signal_y / self._beta_y
                elif self._pickup_axis == 'divergence' and self._kicker_axis == 'displacement':
                    kick_signal_y = kick_signal_y * self._beta_y
                    
            self._kick_bunches(kick_signal_y, 'y', local_slice_sets, bunch_list,
                                self._loc_signal_sets_y)
    

class PickUp(GenericOneTurnMapObject):
    """ A pickup object for the one turn map in PyHEADTAIL.

    This object can be used as a pickup in the trasverse feedback systems
    consisting of separate pickup(s) and kicker(s). A model for signal
    processing (including, for example, bandwidth limitations and noise) can be
    implemented by using signal processors. The signal can be transferred to
    kicker(s) by putting registers to the signal processor chains.
    """

    def __init__(self, slicer, processors_x, processors_y, location_x, beta_x,
                 location_y, beta_y, mpi=False, phase_x=None, phase_y=None,
                 **kwargs):
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
        
        super(self.__class__, self).__init__(0, slicer, processors_x,
             processors_y=processors_y, pickup_axis='displacement',
             kicker_axis=None, mpi=mpi, phase_x=phase_x, location_x=location_x,
             location_y=location_y, phase_y=phase_y, beta_x=beta_x,
             beta_y=beta_y, **kwargs)

    def track(self, bunch):
        
        bunch_list, local_slice_sets, signal_slice_sets_x, signal_slice_sets_y = self._get_slice_sets(bunch)
        
        if self._signal_x is None:
            self._init_signals(bunch_list, signal_slice_sets_x, signal_slice_sets_y)

        self._read_signal(self._signal_x, signal_slice_sets_x, 'x',
                           self._phase_x, self._beta_x)   
        
        end_parameters_x, end_signal_x = process(self._parameters_x,
                                                   self._signal_x,
                                                   self._processors_x,
                                                   slice_sets=signal_slice_sets_x)
        
        if self._processors_y is not None:

            self._read_signal(self._signal_y, signal_slice_sets_y, 'y',
                               self._phase_y, self._beta_y)   
            
            end_parameters_y, end_signal_y = process(self._parameters_y,
                                                       self._signal_y,
                                                       self._processors_y,
                                                       slice_sets=signal_slice_sets_y)


class Kicker(GenericOneTurnMapObject):
    """ A Kicker object for the one turn map in PyHEADTAIL.

    This object can be used as a kicker in the trasverse feedback systems
    consisting of separate pickup(s) and kicker(s). A model for signal
    processing (including, for example, bandwidth limitations and noise) can be
    implemented by using signal processors. The input signals for the kicker
    are the lists of register objects given as a input paramter.
    """
    def __init__(self, gain, slicer, processors_x, processors_y,
                 registers_x, registers_y, location_x, beta_x,
                 location_y, beta_y, combiner='vector_sum', mpi=False,
                 **kwargs):
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

        if isinstance(combiner, str):
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
        
        super(self.__class__, self).__init__(gain, slicer, processors_x,
             processors_y=processors_y, pickup_axis='divergence',
             kicker_axis='divergence', mpi=mpi, location_x=location_x,
             location_y=location_y,beta_x=beta_x, beta_y=beta_y, **kwargs)

    def track(self, bunch):
        
        bunch_list, local_slice_sets, signal_slice_sets_x, signal_slice_sets_y = self._get_slice_sets(bunch)
        
        if self._signal_x is None:
            self._init_signals(bunch_list, signal_slice_sets_x, signal_slice_sets_y)

        parameters_x, signal_x = self._combiner_x.process()   
        parameters_x, signal_x = process(parameters_x,
                                                   signal_x,
                                                   self._processors_x,
                                                   slice_sets=signal_slice_sets_x)
        if signal_x is not None:

            signal_x = signal_x * self._gain_x
            self._kick_bunches(signal_x, 'x', local_slice_sets,
                                bunch_list, self._loc_signal_sets_x)
        
        if self._processors_y is not None:
            self._parameters_y, self._signal_y = self._combiner_y.process() 
            kick_parameters_y, kick_signal_y = process(self._parameters_y,
                                                       self._signal_y,
                                                       self._processors_y,
                                                       slice_sets=signal_slice_sets_y)
            if kick_signal_y is not None:
                kick_signal_y = kick_signal_y * self._gain_y
                self._kick_bunches(kick_signal_y, 'y', local_slice_sets,
                                    bunch_list, self._loc_signal_sets_y)
