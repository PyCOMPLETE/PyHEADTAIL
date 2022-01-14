import numpy as np
import copy
from scipy import interpolate
from scipy.sparse import csr_matrix

from ..core import Parameters, bin_edges_to_z_bins, z_bins_to_bin_edges
from ..core import append_bin_edges, bin_mids, default_macros

"""Signal processors for resampling a signal.

@author Jani Komppula
@date: 11/10/2017
"""

class Resampler(object):

    def __init__(self, method, n_samples=None, offset=0., data_conversion='sum',
                 n_extras = 0, **kwargs):
        """
        Resamples the input signal into a new bin set

        Parameters
        ----------
        method : tuple
            Resampling method. Possible options are:
                ('harmonic', double)
                    The input signal is converted into one continously sampled
                    segment. The given number corresponds to the segment spacing
                    frequency of the input signal (e.g. the harmonic or bunch
                    frequency of the accelerator).
                ('sequenced', double)
                    Each segment of the signal is resampled by using a given
                    sampling frequency.
                ('previous', int)
                    The signal is resampled into the previous bin set. The given
                    number corresponds to an index of the previous parameters in
                    the input signal paramters.
                ('upsampling', int)
                    Multiplies the original sampling rate by the given number
                ('downsampling', int)
                    Reduces the original sampling rate by the given number. If
                    the given number is not an harmonic of the number of bins
                    per segment, the last bins of the segments are skipped.
        n_samples : int
            A number of samples per input segment when options 'harmonic' or
            'sequenced' have been used. If the given value is None, the number
            of samples corresponds to the ceil(segment_length*f_sampling)
        offset : double
            By default the mid points of the new bin set for the segments have
            been set to the found segment reference points from the input signal
            parameters. The give value correspods the mid point offsets to the
            reference points in the units of bins.
        data_conversion : string
            A method how the data of the input signal are converted to the output
            binset. The output signal can be converted by using:
                'interpolation'
                    interpolates from the input data.
                'sum'
                    calculates a bin value sum over the over lapping bins
                'integral'
                    integrates the input signal over an output bin
                'average'
                    calculates a bin width weighted average of the overlaping bins
                'average_bin_value'
                    calculates an average value of the overlaping bins
                'value'
                    returns a value of the overlapping bin
                ('upsampler_kernel', list)
                    uses a kernel to map an old value to a corresponding
                    section of upsampled bins
        n_extras : int
            A number of extra samples added before the first segment and after
            the last segment
        """

        self._method = method
        self._n_samples = n_samples
        self._offset = offset

        self._n_extras = n_extras

        self._data_conversion = data_conversion

        self._output_parameters = None
        self._output_signal = None

        self._convert_signal = None

        self.extensions = []
        self._macros = [] + default_macros(self, 'Resampler', **kwargs)
        self.signal_classes = None

    def _init_harmonic_bins(self, parameters, signal):
        self.signal_classes = (1,2)
        base_frequency = self._method[1]

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

        segment_length = 1./base_frequency
        bin_width = segment_length/float(n_bins_per_segment)

        n_sampled_sequencies = (max_ref_point-min_ref_point) / segment_length + 1
        n_sampled_sequencies = int(np.round(n_sampled_sequencies))

        total_n_samples = int((n_sampled_sequencies + 2*self._n_extras) * n_bins_per_segment)

        segment_z_bins = np.linspace(0, segment_length, n_bins_per_segment+1)
        segment_z_bins = segment_z_bins + (self._offset - np.floor(n_bins_per_segment/2.)-0.5)*bin_width
        segment_bin_edges = z_bins_to_bin_edges(segment_z_bins)

        bin_edges = None

        for i in range(self._n_extras):
            offset = start_mid - (self._n_extras-i)*segment_length
            if bin_edges is None:
                bin_edges = np.copy(segment_bin_edges+offset)
            else:
                bin_edges = append_bin_edges(bin_edges, segment_bin_edges+offset)

        for i in range(n_sampled_sequencies):
            offset = i*segment_length + start_mid
            if bin_edges is None:
                bin_edges = np.copy(segment_bin_edges+offset)
            else:
                bin_edges = append_bin_edges(bin_edges, segment_bin_edges+offset)

        for i in range(self._n_extras):
            offset = start_mid + (i+n_sampled_sequencies)*segment_length
            if bin_edges is None:
                bin_edges = np.copy(segment_bin_edges+offset)
            else:
                bin_edges = append_bin_edges(bin_edges, segment_bin_edges+offset)

        signal_class = 2
        n_segments = 1
        n_bins_per_segment = total_n_samples
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
        self.signal_classes = (0,1)
        bin_width = 1./self._method[1]
        if self._n_samples is not None:
            n_bins_per_segment = self._n_samples
        else:
            segment_from = parameters['bin_edges'][0,0]
            segment_to = parameters['bin_edges'][parameters['n_bins_per_segment']-1,1]
            raw_segment_length = segment_to - segment_from
            n_bins_per_segment = int(np.ceil(raw_segment_length/bin_width))

        segment_z_bins = np.linspace(0, n_bins_per_segment/self._method[1], n_bins_per_segment+1)
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
        self.signal_classes = (0,0)
        self._output_parameters = copy.deepcopy(parameters['previous_parameters'][self._method[1]])
        self._output_parameters['previous_parameters'] = copy.deepcopy(parameters['previous_parameters'][0:self._method[1]])

        self._output_signal = np.zeros(self._output_parameters['n_segments'] * self._output_parameters['n_bins_per_segment'])

    def _init_upsampling(self, parameters, signal):
        self.signal_classes = (0,0)
        multiplier = self._method[1]

        original_edges = parameters['bin_edges']
        new_edges = None

        for edges in original_edges:
            new_bin_width = (edges[1]-edges[0])/float(multiplier)

            temp_edges = np.zeros((multiplier, 2))

            for i in range(multiplier):
                temp_edges[i,0] = edges[0] + i * new_bin_width
                temp_edges[i,1] = edges[0] + (i + 1) * new_bin_width

            if new_edges is None:
                new_edges = temp_edges
            else:
                new_edges = append_bin_edges(new_edges,temp_edges)


        signal_class = parameters['class']
        n_segments = parameters['n_segments']
        n_bins_per_segment = parameters['n_bins_per_segment']*multiplier
        segment_ref_points = parameters['segment_ref_points']
        previous_parameters =  []
        location = parameters['location']
        beta = parameters['beta']
        self._output_parameters = Parameters(signal_class, new_edges, n_segments,
                                             n_bins_per_segment, segment_ref_points,
                                             previous_parameters, location, beta)
        temp_parameters = copy.deepcopy(parameters)
        temp_parameters['previous_parameters'] = []
        self._output_parameters['previous_parameters'] = copy.deepcopy(parameters['previous_parameters'])
        self._output_parameters['previous_parameters'].append(temp_parameters)
        self._output_signal = np.zeros(len(signal)*multiplier)

    def _init_downsampling(self, parameters, signal):
        self.signal_classes = (0,0)
        multiplier = self._method[1]

        original_edges = parameters['bin_edges']
        original_n_bins_per_segment = parameters['n_bins_per_segment']

        n_bins_per_segment = int(np.floor(original_n_bins_per_segment/multiplier))
        new_edges = None

        for j in range(parameters['n_segments']):
            for i in range(n_bins_per_segment):
                first_edge = j * original_n_bins_per_segment + i * multiplier
                last_edge = j * original_n_bins_per_segment + (i + 1) * multiplier -1

                temp_edges = np.zeros((1, 2))
                temp_edges[0,0] = original_edges[first_edge,0]
                temp_edges[0,1] = original_edges[last_edge,1]

                if new_edges is None:
                    new_edges = temp_edges
                else:
                    new_edges = append_bin_edges(new_edges,temp_edges)


        signal_class = parameters['class']
        n_segments = parameters['n_segments']
        segment_ref_points = parameters['segment_ref_points']
        previous_parameters =  []
        location = parameters['location']
        beta = parameters['beta']
        self._output_parameters = Parameters(signal_class, new_edges, n_segments,
                                             n_bins_per_segment, segment_ref_points,
                                             previous_parameters, location, beta)
        temp_parameters = copy.deepcopy(parameters)
        temp_parameters['previous_parameters'] = []
        self._output_parameters['previous_parameters'] = copy.deepcopy(parameters['previous_parameters'])
        self._output_parameters['previous_parameters'].append(temp_parameters)
        self._output_signal = np.zeros(n_bins_per_segment*n_segments)


    def _init_interp_conversion(self, parameters, signal):
        conversion_map = np.zeros(len(self._output_signal), dtype=bool)

        input_bin_mids = bin_mids(parameters['bin_edges'])
        output_bin_mids = bin_mids(self._output_parameters['bin_edges'])

        for i in range(parameters['n_segments']):
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

    def _init_upsampler_kernel_conversion(self, parameters, signal):
        kernel = self._data_conversion[1]
        big_matrix = np.zeros((len(self._output_signal), len(signal)))
        for j, input_edges in enumerate(parameters['bin_edges']):
            for k in range(len(kernel)):
                i = j*len(kernel) + k
                big_matrix[i, j] = kernel[k]
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


    def _init_value_conversion(self, parameters, signal):
        def CDF(x, ref_edges):
            if x <= ref_edges[0]:
                return 0.
            elif x < ref_edges[1]:
                return (x-ref_edges[0])/float(ref_edges[1]-ref_edges[0])
            else:
                return 1.

        big_matrix = np.zeros((len(self._output_signal), len(signal)))
        output_bin_mids = bin_mids(self._output_parameters['bin_edges'])

        for i, mid in enumerate(output_bin_mids):
            for j, edges in enumerate(parameters['bin_edges']):
                if (mid >= edges[0]) and (mid < edges[1]) :
                    big_matrix[i, j] = 1

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
            elif self._method[0] == 'upsampling':
                self._init_upsampling(parameters, signal)
            elif self._method[0] == 'downsampling':
                self._init_downsampling(parameters, signal)
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
        elif self._data_conversion == 'value':
            self._convert_signal = self._init_value_conversion(parameters, signal)
        elif isinstance(self._method, tuple):
            if self._data_conversion[0] == 'upsampler_kernel':
                self._convert_signal = self._init_upsampler_kernel_conversion(parameters, signal)
            else:
                raise ValueError('Unknown data conversion method')
        else:
            raise ValueError('Unknown data conversion method')

    def process(self, parameters, signal, *args, **kwargs):
        if self._convert_signal is None:
            self._init_variables(parameters,signal)

        output_signal = self._convert_signal(signal)

        return self._output_parameters, output_signal

class Quantizer(object):
    def __init__(self, n_bits, input_range, **kwargs):
        """
        Quantizates the input signal into discrete levels

        Parameters
        ----------
        n_bits : int
            A number of bits in the output signal. In the other
            worlds the singal is rounded into 2^n_bits levels.
        input_range : tuple
            A range which is divided into the n bits. The signal values exceed
            the range are limited into the range values
        """

        self._n_bits = n_bits
        self._n_steps = np.power(2,self._n_bits)-1.
        self._input_range = input_range
        self._step_size = (self._input_range[1]-self._input_range[0])/float(self._n_steps)

        self.signal_classes = (0, 0)

        self.extensions = []
        self._macros = [] + default_macros(self, 'Quantizer', **kwargs)

    def process(self, parameters, signal, *args, **kwargs):
        output_signal = self._step_size*np.floor(signal/self._step_size+0.5)

        output_signal[output_signal < self._input_range[0]] = self._input_range[0]
        output_signal[output_signal > self._input_range[1]] = self._input_range[1]

        return parameters, output_signal


class ADC(object):
    def __init__(self, sampling_rate,  n_bits=None, input_range=None, n_samples=None,
                 data_conversion='sum', **kwargs):
        """
        A model for an analog to digital converter. The input signal is
        resamapled segment by segment by using a given sampling rate.
        If both n_bits and input_range have been given, the output signal is also
        quantitized.


        Parameters
        ----------
        sampling rate : float
            A number of samples per second.
        n_bits : int
            A number of bits for the quantizer
        input_range : tuple
            A range for the quantizer
        n_samples : int
            A number of bins per segment is set. If None, the number
            of samples corresponds to the ceil(segment_length*f_sampling)
        """

        self.signal_classes = (0, 1)
        self._resampler = Resampler(('sequenced', sampling_rate) , n_samples,
                                    data_conversion=data_conversion, **kwargs)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range, **kwargs)
        elif (n_bits is not None) or (input_range is not None):
            raise ValueError('Both n_bits and input_range are required for the Quantizer.')

        self.extensions = []
        self._macros = [] + default_macros(self, 'ADC', **kwargs)

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal
                                                                              , *args, **kwargs)

        return output_parameters, output_signal

class HarmonicADC(object):
    def __init__(self, base_frequency, n_bits=None, input_range=None,
                 multiplier = 1, data_conversion='average_bin_value', **kwargs):
        """
        A model for an analog to digital converter, which is simular to the
        regular ADC object expect that the input signal is continously resampled
        ovet the segments. If both n_bits and input_range have been given,
        the output signal is also quantitized.

        Parameters
        ----------
        base_frequency : float
            A base frequency, which corresponds to segment spacing (e.g.
            a harmonic frequency of the accelerator)
        n_bits : int
            A number of bits for the quantizer
        input_range : tuple
            A range for the quantizer
        multiplier : int
            A multiplier for the base frequnecy, which together define
            the sampling rate, i.e. f_sampling = f_base * multiplier
        """
        self.signal_classes = (0, 2)
        self._resampler = Resampler(('harmonic', (base_frequency)) , multiplier,
                                    data_conversion=data_conversion, **kwargs)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range, **kwargs)
        elif (n_bits is not None) or (input_range is not None):
            raise ValueError('Both n_bits and input_range are required for the Quantizer.')

        self.extensions = []
        self._macros = [] + default_macros(self, 'HarmonicADC', **kwargs)

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal
                                                                              , *args, **kwargs)

        return output_parameters, output_signal


class DAC(object):
    def __init__(self,  n_bits = None, output_range = None, method = ('upsampling', 4),
                 data_conversion='value', **kwargs):
        """
        An model for a digital to analog converter, which quantitizes and
        and upsamples the signal by default. The bin set is upsampled by default,
        because the sampling rate is often minimized in the real life applications,
        but after the DAC the signal is reprocessed by using analog electronics. An
        analog signal is continous, which modelling requres higher smapling rate.

        Parameters
        ----------
        n_bits : int
            A number of bits for the quantizer
        output_range : tuple
            A range for the quantizer
        method : tuple
            Resampling method. Possible options are:
                ('upsampling', int)
                    Multiplies the original sampling rate by the given number
                ('previous', int)
                    Returns the previous bin set, which index is given
                ('downsampling', int)
                    Reduces the sampling rate by the given factor

        """
        self._resampler = Resampler(method,
                                data_conversion=data_conversion, **kwargs)
        self.signal_classes = self._resampler.signal_classes

        self._digitizer = None
        if (n_bits is not None) and (output_range is not None):
            self._digitizer = Quantizer(n_bits,output_range, **kwargs)
        elif (n_bits is not None) or (output_range is not None):
            raise ValueError('Both n_bits and input_range are required for the Quantizer.')

        self.extensions = []
        self._macros = [] + default_macros(self, 'DAC', **kwargs)

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal,
                                                                              *args, **kwargs)


        return output_parameters, output_signal

class Upsampler(Resampler):
    def __init__(self, multiplier, kernel=None, **kwargs):
        """
        Multiplies sampling rate by a given number
        
        Parameters
        ----------
        multiplier : int
            A number of new samples per old sample
        kernel : list
            A list of number, which is used as a kernel (map) to determine
            values to the upsampled bins
        """
        if kernel is None:
            data_conversion = 'value'
        else:
            if multiplier != len(kernel):
                raise ValueError('Kernel length must match the multiplier ')
            
            data_conversion = ('upsampler_kernel',kernel)
            
        
        super(self.__class__, self).__init__(('upsampling', multiplier),
                  data_conversion=data_conversion, **kwargs)
        self.label='Upsampler'

class BackToOriginalBins(Resampler):
    def __init__(self, data_conversion='interpolation', target_binset = 0, **kwargs):
        """
        Returns signal to the original bin set.
        Parameters
        ----------
        data_conversion : str
            The method how the input signal values are converted
        taget_binset : int
            Index of the target bin set. Index 0 correspons to the first bin
            set used.
        """
        super(self.__class__, self).__init__(('previous',target_binset),
                  data_conversion=data_conversion, **kwargs)
        self.label='BackToOriginalBins'


class BunchByBunchSampler(Resampler):
    def __init__(self,f_harmonic, multiplier=1, data_conversion='average_bin_value', **kwargs):
        super(self.__class__, self).__init__(('harmonic', (f_harmonic)) , multiplier,
                                    data_conversion=data_conversion, **kwargs)
        self.label = 'Bunch by bunch sampler'


