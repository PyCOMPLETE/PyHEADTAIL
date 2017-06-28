import numpy as np
from scipy.constants import c, pi
import copy, collections
#from cython_hacks import cython_matrix_product
# from scipy.interpolate import interp1d
from scipy import interpolate
from ..core import Parameters, bin_edges_to_z_bins, z_bins_to_bin_edges, append_bin_edges, bin_mids
from ..core import debug_extension
from scipy.sparse import csr_matrix

"""
    This file contains signal processors which can be used for emulating digital signal processing in the feedback
    module. All the processors can be used separately, but digital filters assumes uniform slice spacing (bin width).
    If UniformCharge mode is used in the slicer, uniform bin width can be formed with ADC and DAC processors.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""

class Resampler(object):

    def __init__(self, method, n_samples=None, offset=0., data_conversion='sum',
                 label='Resampler', n_extra_samples = 0, **kwargs):
        self._method = method
        self._n_samples = n_samples
        self._offset = offset

        self._n_extra_samples = n_extra_samples

        self._data_conversion = data_conversion

        self._output_parameters = None
        self._output_signal = None

        self._convert_signal = None

        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, label, **kwargs)]

    def _init_harmonic_bins(self, parameters, signal):
        circumference = self._method[1][0]
        h_RF = self._method[1][1]
        if parameters['n_segments'] > 1:
            min_ref_point = np.min(parameters['segment_ref_points'])
            max_ref_point = np.max(parameters['segment_ref_points'])
            start_mid = parameters['segment_ref_points'][0]
        else:
            mids = bin_mids(parameters['bin_edges'])
            min_ref_point = np.min(mids)
            max_ref_point = np.max(mids)
            start_mid = mids[0]

        if self._n_samples is not None:
            n_bins_per_segment = self._n_samples
        else:
            n_bins_per_segment = 1

        segment_length = circumference/float(h_RF)
        bin_width = segment_length/float(n_bins_per_segment)

        n_sampled_sequencies = (max_ref_point-min_ref_point) / segment_length + 1
        n_sampled_sequencies = int(np.round(n_sampled_sequencies))

        total_n_samples = int(n_sampled_sequencies * n_bins_per_segment)

        segment_z_bins = np.linspace(0, segment_length, n_bins_per_segment+1)
        segment_z_bins = segment_z_bins + (self._offset - np.floor(n_bins_per_segment/2.)-0.5)*bin_width
        segment_bin_edges = z_bins_to_bin_edges(segment_z_bins)

        bin_edges = None

        for i in xrange(n_sampled_sequencies):
            offset = i*segment_length + start_mid
            if bin_edges is None:
                bin_edges = np.copy(segment_bin_edges+offset)
            else:
                bin_edges = append_bin_edges(bin_edges, segment_bin_edges+offset)

        signal_class = 2
        n_segments = 1
#        print 'n_bins_per_segment: '
#        print n_bins_per_segment
#        print ''
#        print ''
        n_bins_per_segment = total_n_samples
#        print 'segment_bin_edges: '
#        print segment_bin_edges
#        print ''
#        print ''
#        print 'max_ref_point: '
#        print max_ref_point
#        print ''
#        print ''
#        print 'min_ref_point: '
#        print min_ref_point
#        print ''
#        print ''
#        print 'segment_length: '
#        print segment_length
#        print ''
#        print ''
#        print 'n_sampled_sequencies: '
#        print n_sampled_sequencies
#        print ''
#        print ''
#        print 'bin_edges: '
#        print bin_edges
#        print ''
#        print ''
#        print 'parameters:'
#        print parameters
#        print ''
#        print ''
#        print 'self.label: '
#        print self.label
#        print ''
#        print ''
        segment_ref_points = [np.mean(bin_edges_to_z_bins(bin_edges))]
        previous_parameters = []
        location = parameters['location']
        beta = parameters['beta']

        self._output_parameters = Parameters(signal_class, bin_edges, n_segments,
                                             n_bins_per_segment, segment_ref_points,
                                             previous_parameters, location, beta)
        temp_parameters = copy.deepcopy(parameters)
        temp_parameters['previous_parameters'] = []
        self._output_parameters['previous_parameters'] = copy.deepcopy(parameters['previous_parameters'])
        self._output_parameters['previous_parameters'].append(temp_parameters)
        self._output_signal = np.zeros(total_n_samples)


    def _init_sequenced_bins(self, parameters, signal):
        bin_width = 1./self._method[1]*c
        if self._n_samples is not None:
            n_bins_per_segment = self._n_samples
        else:
            segment_from = parameters['bin_edges'][0,0]
            segment_to = parameters['bin_edges'][parameters['n_bins_per_segment']-1,1]
            raw_segment_length = segment_to - segment_from
            n_bins_per_segment = int(np.ceil(raw_segment_length/bin_width))

        segment_z_bins = np.linspace(0, n_bins_per_segment/self._method[1]*c, n_bins_per_segment+1)
        segment_z_bins = segment_z_bins - np.mean(segment_z_bins) + self._offset*bin_width
        segment_bin_edges = z_bins_to_bin_edges(segment_z_bins)

        bin_edges = None
        for offset in parameters['segment_ref_points']:
            if bin_edges is None:
                temp = (segment_bin_edges+offset)
                bin_edges = temp
            else:
                bin_edges = append_bin_edges(bin_edges, segment_bin_edges+offset)
        signal_class = 1
        n_segments = parameters['n_segments']
        segment_ref_points = parameters['segment_ref_points']
        previous_parameters =  []
        location = parameters['location']
        beta = parameters['beta']
        self._output_parameters = Parameters(signal_class, bin_edges, n_segments,
                                             n_bins_per_segment, segment_ref_points,
                                             previous_parameters, location, beta)
        temp_parameters = copy.deepcopy(parameters)
        temp_parameters['previous_parameters'] = []
        self._output_parameters['previous_parameters'] = copy.deepcopy(parameters['previous_parameters'])
        self._output_parameters['previous_parameters'].append(temp_parameters)
        self._output_signal = np.zeros(self._output_parameters['n_segments'] * self._output_parameters['n_bins_per_segment'])

    def _init_previous_bins(self, parameters, signal):
        self._output_parameters = copy.deepcopy(parameters['previous_parameters'][self._method[1]])
        self._output_parameters['previous_parameters'] = copy.deepcopy(parameters['previous_parameters'][0:self._method[1]])

        self._output_signal = np.zeros(self._output_parameters['n_segments'] * self._output_parameters['n_bins_per_segment'])


    def _init_interp_conversion(self, parameters, signal):
        conversion_map = np.zeros(len(self._output_signal), dtype=bool)

        input_bin_mids = bin_mids(parameters['bin_edges'])
        output_bin_mids = bin_mids(self._output_parameters['bin_edges'])

        for i in xrange(parameters['n_segments']):
            i_min = i * parameters['n_bins_per_segment']
            i_max = (i + 1) * parameters['n_bins_per_segment'] - 1
            segment_min_z = input_bin_mids[i_min]
            segment_max_z = input_bin_mids[i_max]

            map_below_max = (output_bin_mids < segment_max_z)
            map_above_min = (output_bin_mids > segment_min_z)

            conversion_map = conversion_map + map_below_max*map_above_min

        def convert_signal(input_signal):
            output_signal = np.zeros(len(output_bin_mids))
            tck = interpolate.splrep(input_bin_mids, input_signal, s=0)
            output_signal[conversion_map] = interpolate.splev(output_bin_mids[conversion_map], tck, der=0)
            return output_signal

        return convert_signal

    def _init_sum_conversion(self, parameters, signal):
        def CDF(x, ref_edges):
            if x <= ref_edges[0]:
                return 0.
            elif x < ref_edges[1]:
                return (x-ref_edges[0])/float(ref_edges[1]-ref_edges[0])
            else:
                return 1.

        big_matrix = np.zeros((len(self._output_signal), len(signal)))
        for i, output_edges in enumerate(self._output_parameters['bin_edges']):
            for j, input_edges in enumerate(parameters['bin_edges']):
                big_matrix[i, j] = CDF(output_edges[1], input_edges) - CDF(output_edges[0], input_edges)

        sparse_matrix = csr_matrix(big_matrix)

        def convert_signal(input_signal):
            return sparse_matrix.dot(input_signal)

        return convert_signal

    def _init_integral_conversion(self, parameters, signal):
        def CDF(x, ref_edges):
            if x <= ref_edges[0]:
                return 0.
            elif x < ref_edges[1]:
                return (x-ref_edges[0])/float(ref_edges[1]-ref_edges[0])
            else:
                return 1.

        big_matrix = np.zeros((len(self._output_signal), len(signal)))

        for i, output_edges in enumerate(self._output_parameters['bin_edges']):
            for j, input_edges in enumerate(parameters['bin_edges']):
                bin_width = input_edges[1] - input_edges[0]
                big_matrix[i, j] = (CDF(output_edges[1], input_edges) - CDF(output_edges[0], input_edges))*bin_width

        sparse_matrix = csr_matrix(big_matrix)

        def convert_signal(input_signal):
            return sparse_matrix.dot(input_signal)

        return convert_signal

    def _init_avg_conversion(self, parameters, signal):
        def CDF(x, ref_edges):
            if x <= ref_edges[0]:
                return 0.
            elif x < ref_edges[1]:
                return (x-ref_edges[0])/float(ref_edges[1]-ref_edges[0])
            else:
                return 1.

        big_matrix = np.zeros((len(self._output_signal), len(signal)))


        for i, output_edges in enumerate(self._output_parameters['bin_edges']):
            for j, input_edges in enumerate(parameters['bin_edges']):
                width_coeff =(input_edges[1]-input_edges[0])/(output_edges[1]-output_edges[0])
                big_matrix[i, j] = (CDF(output_edges[1], input_edges) - CDF(output_edges[0], input_edges))*width_coeff

        sparse_matrix = csr_matrix(big_matrix)

        def convert_signal(input_signal):
            return sparse_matrix.dot(input_signal)

        return convert_signal

    def _init_avg_bin_conversion(self, parameters, signal):
        def CDF(x, ref_edges):
            if x <= ref_edges[0]:
                return 0.
            elif x < ref_edges[1]:
                return (x-ref_edges[0])/float(ref_edges[1]-ref_edges[0])
            else:
                return 1.

        big_matrix = np.zeros((len(self._output_signal), len(signal)))


        for i, output_edges in enumerate(self._output_parameters['bin_edges']):
            for j, input_edges in enumerate(parameters['bin_edges']):
                big_matrix[i, j] = (CDF(output_edges[1], input_edges) - CDF(output_edges[0], input_edges))
            if np.sum(big_matrix[i, :]) != 0.:
                big_matrix[i, :] = big_matrix[i, :]/np.sum(big_matrix[i, :])

        sparse_matrix = csr_matrix(big_matrix)

        def convert_signal(input_signal):
            return sparse_matrix.dot(input_signal)

        return convert_signal

    def _init_extremum_conversion(self, parameters, signal):
        # use np.split etc
        pass

    def _init_variables(self, parameters, signal):
        if isinstance(self._method, tuple):
            if self._method[0] == 'harmonic':
                self._init_harmonic_bins(parameters, signal)
            elif self._method[0] == 'sequenced':
                self._init_sequenced_bins(parameters, signal)
            elif self._method[0] == 'previous':
                self._init_previous_bins(parameters, signal)
            else:
                raise ValueError('Unknown sampling method')

        else:
            raise ValueError('Unknown sampling method')

        if self._data_conversion == 'interpolation':
            self._convert_signal = self._init_interp_conversion(parameters, signal)
        elif self._data_conversion == 'sum':
            self._convert_signal = self._init_sum_conversion(parameters, signal)
        elif self._data_conversion == 'integral':
            self._convert_signal = self._init_integral_conversion(parameters, signal)
        elif self._data_conversion == 'average':
            self._convert_signal = self._init_avg_conversion(parameters, signal)
        elif self._data_conversion == 'average_bin_value':
            self._convert_signal = self._init_avg_bin_conversion(parameters, signal)
        else:
            raise ValueError('Unknown data conversion method')

    def process(self, parameters, signal, *args, **kwargs):
        if self._convert_signal is None:
            self._init_variables(parameters,signal)

        output_signal = self._convert_signal(signal)

        for extension in self._extension_objects:
            extension(self, parameters, signal, self._output_parameters, output_signal,
                      *args, **kwargs)

        return self._output_parameters, output_signal

class Quantizer(object):
    def __init__(self,n_bits,input_range, label = 'Quantizer', **kwargs):

        """ Quantizates signal to discrete levels determined by the number of bits and input range.
        :param n_bits: the signal is quantized (rounded) to 2^n_bits levels
        :param input_range: the maximum and minimum values for the levels in the units of input signal
        """

        self._n_bits = n_bits
        self._n_steps = np.power(2,self._n_bits)-1.
        self._input_range = input_range
        self._step_size = (self._input_range[1]-self._input_range[0])/float(self._n_steps)

        self.signal_classes = (0, 0)

        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, label, **kwargs)]

    def process(self, parameters, signal, *args, **kwargs):
        output_signal = self._step_size*np.floor(signal/self._step_size+0.5)

        output_signal[output_signal < self._input_range[0]] = self._input_range[0]
        output_signal[output_signal > self._input_range[1]] = self._input_range[1]

        for extension in self._extension_objects:
            extension(self, parameters, signal, parameters, output_signal,
                      *args, **kwargs)

        return parameters, output_signal


class ADC(object):
    def __init__(self,sampling_rate,  n_bits=None, input_range=None, n_samples=None,
                 data_conversion='sum', **kwargs):
        """ A model for an analog to digital converter, which changes a length of the input signal to correspond to
            the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
            the input signal to discrete levels.
        :param sampling_rate: sampling rate of the ADC [Hz]
        :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
                is not quantizated. The default value is None.
        :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
                The default value is None.
        :param sync_method: The time range of the input signal might not correspond to an integer number of
            samples determined by sampling rate.
                'rounded': The time range of the input signal is divided to number of samples, which correspons to
                    the closest integer of samples determined by the sampling rate (defaul)
                'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
                    of the signal
                'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
                    of the signal
                'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
                    in the beginning and end of the signal
        """
        self.label = 'ADC'
        self.signal_classes = (0, 1)
        self._resampler = Resampler(('sequenced', sampling_rate) , n_samples,
                                    data_conversion=data_conversion, **kwargs)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range, **kwargs)
        elif (n_bits is not None) or (input_range is not None):
            raise ValueError('Both n_bits and input_range are required for the Quantizer.')



        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, 'ADC', **kwargs)]

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal
                                                                              , *args, **kwargs)

        for extension in self._extension_objects:
            extension(self, parameters, signal, output_parameters, output_signal,
                      *args, **kwargs)

        return output_parameters, output_signal

class HarmonicADC(object):
    def __init__(self,circumference, h_RF, multiplier, n_bits=None, input_range=None,
                 data_conversion='average_bin_value', **kwargs):
        """ A model for an analog to digital converter, which changes a length of the input signal to correspond to
            the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
            the input signal to discrete levels.
        :param sampling_rate: sampling rate of the ADC [Hz]
        :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
                is not quantizated. The default value is None.
        :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
                The default value is None.
        :param sync_method: The time range of the input signal might not correspond to an integer number of
            samples determined by sampling rate.
                'rounded': The time range of the input signal is divided to number of samples, which correspons to
                    the closest integer of samples determined by the sampling rate (defaul)
                'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
                    of the signal
                'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
                    of the signal
                'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
                    in the beginning and end of the signal
        """
        self.label = 'ADC'
        self.signal_classes = (0, 1)
        self._resampler = Resampler(('harmonic', (circumference, h_RF)) , multiplier,
                                    data_conversion=data_conversion, **kwargs)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range, **kwargs)
        elif (n_bits is not None) or (input_range is not None):
            raise ValueError('Both n_bits and input_range are required for the Quantizer.')

        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, 'HarmonicADC', **kwargs)]

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal
                                                                              , *args, **kwargs)

        for extension in self._extension_objects:
            extension(self, parameters, signal, output_parameters, output_signal,
                      *args, **kwargs)

        return output_parameters, output_signal


class DAC(object):
    def __init__(self,  n_bits = None, output_range = None, target_binset = 0,
                 data_conversion='interpolation', **kwargs):
        """ A model for an digital to analog converter, which changes a length of the input signal to correspond to
            the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
            the input signal to discrete levels.
        :param sampling_rate: sampling rate of the ADC [Hz]
        :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
                is not quantizated. The default value is None.
        :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
                The default value is None.
        :param sync_method: The time range of the input signal might not correspond to an integer number of
            samples determined by sampling rate.
                'rounded': The time range of the input signal is divided to number of samples, which correspons to
                    the closest integer of samples determined by the sampling rate (defaul)
                'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
                    of the signal
                'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
                    of the signal
                'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
                    in the beginning and end of the signal
        """

        if isinstance(target_binset, (int, long)):
            self.signal_classes = (1, 0)
            self._resampler = Resampler(('previous',target_binset),
                                    data_conversion=data_conversion, **kwargs)
        else:
            self._resampler = Resampler(target_binset,
                                    data_conversion=data_conversion, **kwargs)
            self.signal_classes = self._resampler.signal_classes

        self._digitizer = None
        if (n_bits is not None) and (output_range is not None):
            self._digitizer = Quantizer(n_bits,output_range, **kwargs)
        elif (n_bits is not None) or (output_range is not None):
            raise ValueError('Both n_bits and input_range are required for the Quantizer.')

        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, 'DAC', **kwargs)]

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal,
                                                                              *args, **kwargs)

        for extension in self._extension_objects:
            extension(self, parameters, signal, output_parameters, output_signal,
                      *args, **kwargs)

        return output_parameters, output_signal

class HarmonicDAC(object):
    def __init__(self,circumference, h_RF, multiplier, n_bits=None, input_range=None,
                 data_conversion='interpolation', **kwargs):
        """ A model for an analog to digital converter, which changes a length of the input signal to correspond to
            the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
            the input signal to discrete levels.
        :param sampling_rate: sampling rate of the ADC [Hz]
        :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
                is not quantizated. The default value is None.
        :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
                The default value is None.
        :param sync_method: The time range of the input signal might not correspond to an integer number of
            samples determined by sampling rate.
                'rounded': The time range of the input signal is divided to number of samples, which correspons to
                    the closest integer of samples determined by the sampling rate (defaul)
                'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
                    of the signal
                'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
                    of the signal
                'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
                    in the beginning and end of the signal
        """
        self.label = 'HarmonicDAC'
        self.signal_classes = (0, 1)
        self._resampler = Resampler(('harmonic', (circumference, h_RF)) , multiplier,
                                    data_conversion=data_conversion, **kwargs)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range, **kwargs)
        elif (n_bits is not None) or (input_range is not None):
            raise ValueError('Both n_bits and input_range are required for the Quantizer.')

        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, 'HarmonicADC', **kwargs)]

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal
                                                                              , *args, **kwargs)

        for extension in self._extension_objects:
            extension(self, parameters, signal, output_parameters, output_signal,
                      *args, **kwargs)

        return output_parameters, output_signal

class BackToOriginalBins(Resampler):
    def __init__(self, data_conversion='interpolation', target_binset = 0, **kwargs):
            super(self.__class__, self).__init__(('previous',target_binset),
                  data_conversion=data_conversion,
                  label='BackToOriginalBins',
                  **kwargs)



#class UpSampler(Resampler):
#    def __init__(self, multiplier, kernel = None, **kwargs):
#
#        if kernel is None:
#            kernel = [0.]*multiplier
#            kernel[0] = 1.
#
#        data_conversion = ('kernel',kernel)
#
#        sampling_rate = ('multiplied', multiplier)
#
#        if 'data_conversion' in kwargs:
#            super(self.__class__, self).__init__('reconstructed',sampling_rate, sync_method='rising_edge', **kwargs)
#        else:
#            super(self.__class__, self).__init__('reconstructed',sampling_rate, sync_method='rising_edge',
#                                                 data_conversion = data_conversion, **kwargs)
#        self.label = 'UpSampler'
#
#
#class DownSampler(Resampler):
#    def __init__(self,multiplier,**kwargs):
#
#        sampling_rate = ('multiplied', multiplier)
#
#        if 'data_conversion' in kwargs:
#            super(self.__class__, self).__init__('reconstructed',sampling_rate, **kwargs)
#        else:
#            super(self.__class__, self).__init__('reconstructed',sampling_rate,data_conversion = 'bin_average',
#                                                 sync_method='rising_edge', **kwargs)
#        self.label = 'DownSampler'