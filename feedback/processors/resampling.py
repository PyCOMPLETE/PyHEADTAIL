import numpy as np
from scipy.constants import c, pi
import copy, collections
from cython_hacks import cython_matrix_product
# from scipy.interpolate import interp1d
from scipy import interpolate
from ..core import Parameters

"""
    This file contains signal processors which can be used for emulating digital signal processing in the feedback
    module. All the processors can be used separately, but digital filters assumes uniform slice spacing (bin width).
    If UniformCharge mode is used in the slicer, uniform bin width can be formed with ADC and DAC processors.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""


class Resampler(object):
    def __init__(self,sampling_type, sampling_rate = None, signal_length = None, sync_method = 'bin_mid',
                 length_rounding = 'round', data_conversion = 'bin_average', store_signal  = False):
        """

        :param sampling_type:
        :param sampling_rate:
        :param signal_length:
        :param sync_method:
        :param length_rounding:
        :param data_conversion:
        :param store_signal:
        """

        self._sampling_type = sampling_type
        if self._sampling_type == 'original':
            self.signal_classes = (1, 0)
        else:
            self.signal_classes = (0, 1)
        self._sampling_rate = sampling_rate
        self._sync_method = sync_method

        if signal_length is not None:
            self._signal_length = signal_length*c
        else:
            self._signal_length = None

        self._length_rounding = length_rounding
        self._data_conversion = data_conversion

        self._n_segments = None

        self._input_parameters = None
        self._input_bin_spacing = None
        self._input_n_bins_per_segment = None
        self._input_z_bins = None
        self._input_bin_edges = None
        self._total_input_bin_edges = None
        self._total_input_bin_mids = None

        self._output_parameters = None
        self._output_z_bins = None
        self._output_n_bins_per_segment = None
        self._output_bin_spacing = None
        self._output_bin_edges = None
        self._total_output_bin_edges = None
        self._total_output_bin_mids = None

        # cache for output signal
        self._output_signal = None

        self._conversion_type = None
        self._conversion_matrix = None

        # for storing the signal
        self.signal_classes = (1, 1)

        self.extensions = ['store']
        if self._sampling_type == 'original':
            self.extensions.append('bunch')
            self.required_variables = []
        self._store_signal = store_signal

        self.input_signal = None
        self.input_parameters = None

        self.output_signal = None
        self.output_parameters = None
        self.label = 'Resampler'

    def process(self, parameters, signal,slice_sets = None, *args, **kwargs):
        if self._conversion_type is None:
            self.__init_variables(parameters,slice_sets)

        if self._conversion_type == 'matrix':
            self._output_signal.fill(0.)

            for i in xrange(parameters['n_segments']):
                input_from = i * self._input_n_bins_per_segment
                input_to = (i + 1) * self._input_n_bins_per_segment
                output_from = i * self._output_n_bins_per_segment
                output_to = (i + 1) * self._output_n_bins_per_segment

                np.copyto(self._output_signal[output_from:output_to],
                          np.array(cython_matrix_product(self._conversion_matrix, np.array(signal[input_from:input_to]))))
        elif self._conversion_type == 'interpolation':
            tck = interpolate.splrep(self._total_input_bin_mids, signal, s=0)
            self._output_signal = interpolate.splev(self._total_output_bin_mids, tck, der=0)
        else:
            raise ValueError('Unknown conversion type')

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(self._output_signal)
            self.output_parameters = copy.copy(self._output_parameters)

        return self._output_parameters,self._output_signal

    def __init_variables(self,parameters,slice_sets):
        self._input_parameters = copy.copy(parameters)
        self._n_segments = parameters['n_segments']
        self._input_n_bins_per_segment = parameters['n_bins_per_segment']

        self._input_bin_edges = np.copy(parameters['bin_edges'])
        self._input_z_bins = parameters['bin_edges'][0:self._input_n_bins_per_segment,0]
        self._input_z_bins = np.append(self._input_z_bins,parameters['bin_edges'][(self._input_n_bins_per_segment-1),1])
        self._input_z_bins = self._input_z_bins - parameters['segment_midpoints'][0] # A re
        self._input_bin_spacing = np.mean(parameters['bin_edges'][0:self._input_n_bins_per_segment,1]-parameters['bin_edges'][0:self._input_n_bins_per_segment,0])
        self._total_input_bin_edges = np.copy(parameters['bin_edges'])
        self._total_input_bin_mids = (self._total_input_bin_edges[:,0]+self._total_input_bin_edges[:,1])/2.

        if isinstance(self._sampling_rate, float):
            pass
        elif isinstance(self._sampling_rate, int):
            self._sampling_rate = float(self._sampling_rate)
        elif self._sampling_rate is None:
            self._sampling_rate = c/self._input_bin_spacing
        elif isinstance(self._sampling_rate, tuple):
            if self._sampling_rate[0] == 'multiplied':
                self._sampling_rate = self._sampling_rate[1] * c/self._input_bin_spacing
        else:
            raise ValueError('Unknown value type in Resampler._sampling_rate')


        if self._signal_length is None:
            self._signal_length = self._input_z_bins[-1] - self._input_z_bins[0]


        if self._sampling_type == 'reconstructed':
            z_bins, n_bins_per_segment, bin_spacing, sampling_rate, signal_length = \
                self.__reconstruct_z_bins(self._signal_length, self._sampling_rate, self._input_z_bins)
        elif self._sampling_type == 'original':
            z_bins = np.copy(slice_sets[0].z_bins) - np.mean(slice_sets[0].z_bins)
            n_bins_per_segment = len(z_bins) -1
            bin_spacing = (z_bins[-1] - z_bins[0]) / float(n_bins_per_segment)
            sampling_rate = bin_spacing/c
            signal_length = bin_spacing * n_bins_per_segment
        else:
            raise ValueError('Unknown value in Resampler._sampling_type')

        self._output_z_bins = z_bins
        self._output_n_bins_per_segment = n_bins_per_segment
        self._output_bin_spacing = bin_spacing
        self._sampling_rate = sampling_rate
        self._signal_length = signal_length
        self._output_bin_edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))

        self._total_output_bin_edges = None
        for z_mid in parameters['segment_midpoints']:
            edges = self._output_bin_edges + z_mid

            if self._total_output_bin_edges is None:
                self._total_output_bin_edges = np.copy(edges)
            else:
                self._total_output_bin_edges = np.append(self._total_output_bin_edges, edges, axis=0)

        self._total_output_bin_mids = (self._total_output_bin_edges[:,0]+self._total_output_bin_edges[:,1])/2.
        self._output_signal = np.zeros(len(self._total_output_bin_edges))
        self._output_parameters = Parameters()
        self._output_parameters['class'] = self._input_parameters['class']
        self._output_parameters['bin_edges'] = self._total_output_bin_edges
        self._output_parameters['n_segments'] = self._n_segments
        self._output_parameters['n_bins_per_segment'] = self._output_n_bins_per_segment
        self._output_parameters['segment_midpoints'] =self._input_parameters['segment_midpoints']
        self._output_parameters['location'] = self._input_parameters['location']
        self._output_parameters['beta'] = self._input_parameters['beta']

        if self._data_conversion == 'interpolation':
            self._conversion_type = 'interpolation'
        elif self._data_conversion == 'bin_sum':
            norm_coeff = 1.
            self.__contruct_value_conversion_matrix(norm_coeff)
            self._conversion_type = 'matrix'
        elif self._data_conversion == 'bin_integral':
            # weights the signal sum from difference slices by bin spacinf,
            # i.e. the time integral of the signals stays constant
            norm_coeff = self._input_bin_spacing / self._output_bin_spacing
            self.__contruct_value_conversion_matrix(norm_coeff)
            self._conversion_type = 'matrix'
        elif self._data_conversion == 'bin_average':
            # sets output bin value to an average value of input bins contributing to the output bin
            norm_coeff = 1. / min(self._output_bin_spacing / self._input_bin_spacing,
                                           float(self._input_n_bins_per_segment))
            self.__contruct_value_conversion_matrix(norm_coeff)
            self._conversion_type = 'matrix'
        elif isinstance(self._data_conversion,tuple):
            if self._data_conversion[0] == 'kernel':
                self.__contruct_value_conversion_matrix(self._data_conversion[1])
                self._conversion_type = 'matrix'
            else:
                raise ValueError('Unknown value for Resampler._data_normalization')
        else:
            raise ValueError('Unknown value for Resampler._data_normalization')

    def __reconstruct_z_bins(self,signal_length, sampling_rate, input_z_bins):

        if self._length_rounding == 'round':
            n_bins_per_segment = np.round(signal_length * sampling_rate / c)
            signal_length = float(n_bins_per_segment) * c / sampling_rate
        elif self._length_rounding == 'floor':
            n_bins_per_segment = np.floor(signal_length*sampling_rate / c)
            signal_length = float(n_bins_per_segment) * c / sampling_rate
        elif self._length_rounding == 'ceil':
            n_bins_per_segment = np.ceil(signal_length*sampling_rate / c)
            signal_length = float(n_bins_per_segment) * c / sampling_rate
        elif self._length_rounding == 'exact':
            n_bins_per_segment = np.round(signal_length*sampling_rate / c)
            sampling_rate = signal_length / (c * float(n_bins_per_segment))
        else:
            raise ValueError('Unknown value in Resampler._length_rounding')

        bin_spacing = signal_length / float(n_bins_per_segment)

        if self._sync_method == 'rising_edge':
            z_from = input_z_bins[0]
            z_to = z_from + signal_length

        elif self._sync_method == 'falling_edge':
            z_from = input_z_bins[-1] - signal_length
            z_to = input_z_bins[-1]
        elif self._sync_method == 'middle':
            z_from = np.mean(input_z_bins) - 0.5 * signal_length
            z_to = np.mean(input_z_bins) - 0.5 * signal_length

        elif self._sync_method == 'bin_mid':
            bins_adv = np.round((n_bins_per_segment - 1)/ 2)
            z_from = -1. * (0.5 + float(bins_adv)) * bin_spacing
            z_to = (0.5 + float(n_bins_per_segment - bins_adv - 1)) * bin_spacing
        elif self._sync_method == 'bin_mid_advance':
            if input_z_bins[-1] > 0.5*bin_spacing:
                bins_after = np.ceil((input_z_bins[-1] - 0.5 * bin_spacing)/bin_spacing)
            else:
                bins_after = 0.
            z_from = -1. * (0.5 + float(float(n_bins_per_segment) - bins_after - 1.)) * bin_spacing
            z_to = (bins_after+0.5) * bin_spacing
        elif self._sync_method == 'bin_mid_delay':
            if input_z_bins[0] < -0.5*bin_spacing:
                bins_adv = np.ceil((-1.*input_z_bins[0] - 0.5 * bin_spacing)/bin_spacing)
            else:
                bins_adv = 0.
            z_from = -1. * (bins_adv+0.5) * bin_spacing
            z_to = (0.5 + float(float(n_bins_per_segment) - bins_adv - 1.)) * bin_spacing

        else:
            raise ValueError('Unknown value for Resampler._sync')

        z_bins = np.linspace(z_from, z_to, n_bins_per_segment + 1)

        return z_bins, n_bins_per_segment, bin_spacing, sampling_rate, signal_length

    def __resample_z_bins(self,signal_length, sampling_rate, input_z_bins):

        signal_length = input_z_bins[-1] - input_z_bins[0]

        if self._length_rounding == 'round':
            n_bins_per_segment = np.round(signal_length * sampling_rate / c)
            signal_length = sampling_rate * float(n_bins_per_segment) * c
        elif self._length_rounding == 'floor':
            n_bins_per_segment = np.floor(signal_length*sampling_rate / c)
            signal_length = sampling_rate * float(n_bins_per_segment) * c
        elif self._length_rounding == 'ceil':
            n_bins_per_segment = np.ceil(signal_length*sampling_rate / c)
            signal_length = sampling_rate * float(n_bins_per_segment) * c
        elif self._length_rounding == 'exact':
            n_bins_per_segment = np.round(signal_length*sampling_rate / c)
            sampling_rate = signal_length / (c * float(n_bins_per_segment))
        else:
            raise ValueError('Unknown value in Resampler._length_rounding')

        bin_spacing = signal_length / float(n_bins_per_segment)

        length_difference = signal_length - (input_z_bins[-1] - input_z_bins[0])

        z_from = input_z_bins[0] - length_difference / 2.
        z_to = input_z_bins[-1] + length_difference / 2.
        z_bins = np.linspace(z_from, z_to, n_bins_per_segment + 1)

        return z_bins, n_bins_per_segment, bin_spacing, sampling_rate, signal_length

    def __contruct_value_conversion_matrix(self,conversion_value):
        self._conversion_matrix = np.zeros((len(self._output_z_bins) - 1, len(self._input_z_bins) - 1))

        for i, (i_min, i_max) in enumerate(zip(self._output_z_bins, self._output_z_bins[1:])):
            for j, (j_min, j_max) in enumerate(zip(self._input_z_bins, self._input_z_bins[1:])):
                if isinstance(conversion_value, float):
                    self._conversion_matrix[i, j] = (self.__CDF(i_max, j_min, j_max) -
                                                     self.__CDF(i_min, j_min, j_max)) * conversion_value
                elif isinstance(conversion_value, list):
                    if (i >= j * len(conversion_value)) and (i < (j+1) * len(conversion_value)):
                        kernel_idx = i % len(conversion_value)
                        self._conversion_matrix[i, j] = conversion_value[kernel_idx]
                else:
                    raise ValueError('Unknown data type in conversion_value')

    def __CDF(self,x,ref_bin_from, ref_bin_to):
            if x <= ref_bin_from:
                return 0.
            elif x < ref_bin_to:
                return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
            else:
                return 1.

class Quantizer(object):
    def __init__(self,n_bits,input_range, store_signal = False):

        """ Quantizates signal to discrete levels determined by the number of bits and input range.
        :param n_bits: the signal is quantized (rounded) to 2^n_bits levels
        :param input_range: the maximum and minimum values for the levels in the units of input signal
        """

        self._n_bits = n_bits
        self._n_steps = np.power(2,self._n_bits)-1.
        self._input_range = input_range
        self._store_signal = store_signal
        self._step_size = (self._input_range[1]-self._input_range[0])/float(self._n_steps)

        self.signal_classes = (0, 0)
        self.extensions = ['store']
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

        self.label = 'Quantizer'

    def process(self, parameters, signal, *args, **kwargs):
        output_signal = self._step_size*np.floor(signal/self._step_size+0.5)

        output_signal[output_signal < self._input_range[0]] = self._input_range[0]
        output_signal[output_signal > self._input_range[1]] = self._input_range[1]

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(parameters)

        return parameters, output_signal


class ADC(object):
    def __init__(self,sampling_rate, n_bits = None, input_range = None, store_signal = False, **kwargs):
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
        self.extensions = ['store']
        self._resampler = Resampler('reconstructed' , sampling_rate, **kwargs)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range, *kwargs)
        elif (n_bits is not None) or (input_range is not None):
            raise ValueError('Either both n_bits and input_range must have values or they must be None')

        # for storing the signal
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal
                                                                              , *args, **kwargs)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_signal_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(output_parameters)

        return output_parameters, output_signal


class DAC(object):
    def __init__(self,sampling_rate = None, n_bits = None, output_range = None, store_signal = False, **kwargs):
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
        self.label = 'DAC'
        self.extensions = ['store']

        if sampling_rate is None:
            self.signal_classes = (1, 0)
            self._resampler = Resampler('original',data_conversion = 'interpolation', **kwargs)
            self.extensions.append('bunch')
            self.required_variables = copy.copy(self._resampler.required_variables)
        else:
            self.signal_classes = (1, 1)
            self._resampler = Resampler('reconstructed', sampling_rate,data_conversion = 'interpolation')

        self._digitizer = None
        if (n_bits is not None) and (output_range is not None):
            self._digitizer = Quantizer(n_bits,output_range, **kwargs)
        elif (n_bits is not None) or (output_range is not None):
            raise ValueError('Either both n_bits and input_range must have values or they must be None')

        # for storing the signal
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal,
                                                                              *args, **kwargs)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(output_parameters)

        return output_parameters, output_signal


class BackToOriginalBins(Resampler):
    def __init__(self, **kwargs):
        if 'data_conversion' in kwargs:
            super(self.__class__, self).__init__('original', **kwargs)
        else:
            super(self.__class__, self).__init__('original',data_conversion = 'interpolation', **kwargs)


class UpSampler(Resampler):
    def __init__(self, multiplier, kernel = None, **kwargs):

        if kernel is None:
            kernel = [0.]*multiplier
            kernel[0] = 1.

        data_conversion = ('kernel',kernel)

        sampling_rate = ('multiplied', multiplier)

        if 'data_conversion' in kwargs:
            super(self.__class__, self).__init__('reconstructed',sampling_rate, sync_method='rising_edge', **kwargs)
        else:
            super(self.__class__, self).__init__('reconstructed',sampling_rate, sync_method='rising_edge',
                                                 data_conversion = data_conversion, **kwargs)
        self.label = 'UpSampler'


class DownSampler(Resampler):
    def __init__(self,multiplier,**kwargs):

        sampling_rate = ('multiplied', multiplier)

        if 'data_conversion' in kwargs:
            super(self.__class__, self).__init__('reconstructed',sampling_rate, **kwargs)
        else:
            super(self.__class__, self).__init__('reconstructed',sampling_rate,data_conversion = 'bin_average',
                                                 sync_method='rising_edge', **kwargs)
        self.label = 'DownSampler'