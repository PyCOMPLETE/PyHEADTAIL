import copy, math
import numpy as np
from abc import ABCMeta, abstractmethod

from ..core import Parameters, Signal
from ..core import bin_widths, bin_mids, bin_edges_to_z_bins, z_bins_to_bin_edges
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import UnivariateSpline
from ..core import debug_extension
# TODO: - 2nd order cutoff by using gaussian filter
# TODO: - FIR filter

class Convolution(object):
    __metaclass__ = ABCMeta

    def __init__(self, label='Convolution',**kwargs):

        self._dashed_impulse_responses = None
        self._impulses_from_segments = None
        self._impulses_to_segments = None

        self._n_seg = None
        self._n_bins = None

        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, label, **kwargs)]

    def _init_convolution(self, parameters):

        self._n_seg = parameters['n_segments']
        self._n_bins = parameters['n_bins_per_segment']
#        print 'self._n_bins'
#        print self._n_bins
        bin_edges = parameters['bin_edges']
        n_seg = parameters['n_segments']
        n_bins = parameters['n_bins_per_segment']
        ref_points = parameters['segment_ref_points']

        extra_bins = np.ceil(n_bins/2.)

        impulse_ref_edges = None

        self._dashed_impulse_responses = []
        self._impulses_from_segments = []
        self._impulses_to_segments = []

        for i in xrange(n_seg):
            self._impulses_to_segments.append([])

        ref_points = []

        for i in xrange(n_seg):
            i_from = i*n_bins
            i_to = (i+1)*n_bins
            org_edges = bin_edges[i_from:i_to, :]
            prefix_offset = org_edges[(extra_bins-1), 1]-org_edges[0, 0]
            postfix_offset = org_edges[-extra_bins, 0]-org_edges[-1, 1]

            edges = np.concatenate(((org_edges[:extra_bins]-prefix_offset), org_edges), axis=0)
            edges = np.concatenate((edges, org_edges[extra_bins:]-postfix_offset), axis=0)
            ref_points.append(np.mean(bin_edges_to_z_bins(org_edges)))
            if impulse_ref_edges is None:
                impulse_ref_edges = edges
            else:
                impulse_ref_edges = np.concatenate((impulse_ref_edges, edges), axis=0)

        for i, ref_point in enumerate(ref_points):

            impulse_edges = impulse_ref_edges-ref_point

            target_segments, dashed_impulse_response = self.response_function(impulse_edges, n_seg,
                                                                              n_bins + 2 * extra_bins)

            self._dashed_impulse_responses.append(dashed_impulse_response)

            self._impulses_from_segments.append(np.zeros(len(dashed_impulse_response)))
            for idx, target_idx in enumerate(target_segments):
                i_from = idx*(n_bins + 2 * extra_bins) + extra_bins
                i_to = idx*(n_bins + 2 * extra_bins) + extra_bins + n_bins

                self._impulses_to_segments[target_idx].append(np.array(self._impulses_from_segments[i][i_from:i_to], copy=False))

    @abstractmethod
    def response_function(self, impulse_ref_edges, n_seg, n_bins_per_segment):
        # Impulse response function of the processor
        pass

    def _apply_convolution(self, parameters, signal):

        if self._dashed_impulse_responses is None:
            self._init_convolution(parameters)

        for i in xrange(self._n_seg):

            i_from = i*self._n_bins
            i_to = (i+1)*self._n_bins

            np.copyto(self._impulses_from_segments[i],
                      np.convolve(self._dashed_impulse_responses[i],
                                  signal[i_from:i_to], mode='same'))

        output_signal = np.zeros(len(signal))
        for i in xrange(self._n_seg):

            i_from = i*self._n_bins
            i_to = (i+1)*self._n_bins
            np.copyto(output_signal[i_from:i_to], np.sum(self._impulses_to_segments[i], axis=0))

        return output_signal

    def process(self, parameters, signal, *args, **kwargs):

        output_signal = self._apply_convolution(parameters, signal)

        for extension in self._extension_objects:
            extension(self, parameters, signal, parameters, output_signal,
                      *args, **kwargs)

        return parameters, output_signal

class Delay(Convolution):
    def __init__(self,delay, **kwargs):

        self._z_delay = delay*c

        if self._z_delay < 0.:
            impulse_range = (self._z_delay, 0.)
        else:
            impulse_range = (0., self._z_delay)

        super(self.__class__, self).__init__(impulse_range, **kwargs)
        self.label = 'Delay'

    def response_function(self, impulse_ref_edges, n_segments, n_bins_per_segment):
        impulse_values = np.zeros(len(n_segments*n_bins_per_segment))
        bin_spacing =  np.mean(impulse_ref_edges[:,1]-impulse_ref_edges[:,0])

        ref_bin_from = -0.5*bin_spacing+self._z_delay
        ref_bin_to = 0.5*bin_spacing+self._z_delay

        for i, edges in enumerate(impulse_ref_edges):
            impulse_values[i] = self._CDF(edges[1],ref_bin_from,ref_bin_to) - self._CDF(edges[0],ref_bin_from,ref_bin_to)

        return impulse_values

    def _CDF(self,x,ref_bin_from, ref_bin_to):
        # FIXME: this is not gonna work for nagative delays?

        if x <= ref_bin_from:
            return 0.
        elif x < ref_bin_to:
            return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.


class MovingAverage(Convolution):
    """ Returns a signal, which consists an average value of the input signal. A sums of the rows in the matrix
        are normalized to be one (i.e. a sum of the input signal doesn't change).
    """

    def __init__(self,window_length, quantity = 'time', **kwargs):

        if quantity == 'time':
            self._window = (-0.5 * window_length * c, 0.5 * window_length * c)
        elif quantity == 'distance':
            self._window = (-0.5 * window_length, 0.5 * window_length)
        else:
            raise ValueError('Unknown value in Average.quantity')

        super(self.__class__, self).__init__(self._window, **kwargs)
        self.label = 'Average'

    def response_function(self, impulse_ref_edges, n_segments, n_bins_per_segment):
        impulse_values = np.zeros(len(n_segments*n_bins_per_segment))

        for i, edges in enumerate(impulse_ref_edges):
            impulse_values[i] = self._CDF(edges[1], self._window[0], self._window[1]) \
                                   - self._CDF(edges[0], self._window[0], self._window[1])

        return impulse_values

    def _CDF(self, x, ref_bin_from, ref_bin_to):
        if x <= ref_bin_from:
            return 0.
        elif x < ref_bin_to:
            return (x - ref_bin_from) / float(ref_bin_to - ref_bin_from)
        else:
            return 1.
#
#
#class WaveletGenerator(Convolution):
#
#    def __init__(self,spacing,n_copies, **kwargs):
#        self._spacing = spacing
#        self._n_copies = n_copies
#
#        if isinstance(self._n_copies,tuple):
#            self._i_from = self._n_copies[0]
#            self._i_to = self._n_copies[1]
#
#        else:
#            self._i_from = min(self._n_copies,0)
#            self._i_to = max(self._n_copies,0)
#
#        self._window = (self._i_from*self._spacing*c,self._i_to*self._spacing*c)
#
#        super(self.__class__, self).__init__(self._window, **kwargs)
#        self.label = 'Wavelet generator'
#
#
#    def calculate_response(self, impulse_bin_mids, impulse_bin_edges):
#
#        bin_spacing = np.mean(impulse_bin_edges[:,1]-impulse_bin_edges[:,0])
#        impulse_values = np.zeros(len(impulse_bin_mids))
#
#        for i in xrange(self._i_from,(self._i_to+1)):
#            copy_mid = i*self._spacing*c
#            copy_from = copy_mid - 0.5 * bin_spacing
#            copy_to = copy_mid + 0.5 * bin_spacing
#
#            for j, edges in enumerate(impulse_bin_edges):
#                impulse_values[j] += (self._CDF(edges[1],copy_from,copy_to)-self._CDF(edges[0],copy_from,copy_to))
#
#        return impulse_values
#
#    def _CDF(self, x, ref_bin_from, ref_bin_to):
#        if x <= ref_bin_from:
#            return 0.
#        elif x < ref_bin_to:
#            return (x - ref_bin_from) / float(ref_bin_to - ref_bin_from)
#        else:
#            return 1.
#
#class ConvolutionFromFile(Convolution):
#    """ Interpolates matrix columns by using inpulse response data from a file. """
#
#    def __init__(self,filename, x_axis = 'time', calc_type = 'mean',  **kwargs):
#        self._filename = filename
#        self._x_axis = x_axis
#        self._calc_type = calc_type
#
#        self._data = np.loadtxt(self._filename)
#        if self._x_axis == 'time':
#            self._data[:, 0]=self._data[:, 0]*c
#
#        impulse_range = (self._data[0,0],self._data[-1,0])
#
#        super(self.__class__, self).__init__(impulse_range, **kwargs)
#        self.label = 'Convolution from external data'
#
#    def calculate_response(self, impulse_response_bin_mid, impulse_response_bin_edges):
#
#        if self._calc_type == 'mean':
#            return np.interp(impulse_response_bin_mid, self._data[:, 0], self._data[:, 1])
#        elif self._calc_type == 'integral':
#            s = UnivariateSpline(self._data[:, 0], self._data[:, 1])
#            response_values = np.zeros(len(impulse_response_bin_mid))
#
#            for i, edges in enumerate(impulse_response_bin_edges):
#                response_values[i], _ = s.integral(edges[0],edges[1])
#            return response_values
#
#        else:
#            raise ValueError('Unknown value in ConvolutionFromFile._calc_type')

class ConvolutionFilter(Convolution):
    __metaclass__ = ABCMeta

    def __init__(self,scaling,impulse_range,zero_bin_value = None, tip_cut_width=None,
                 normalization=None, label='ConvolutionFilter', **kwargs):

        self._scaling = scaling
        self._normalization = normalization
#        self._norm_range = norm_range
        self._zero_bin_value = zero_bin_value
        super(ConvolutionFilter, self).__init__(label=label,**kwargs)

        # NOTE: is the tip cut needed? How to work with the sharp tips of the ideal filters?
        if (self._normalization is None) and (tip_cut_width is not None):
            self._normalization = 'integral'
        self._impulse_response = self._impulse_response_generator(tip_cut_width)

    def response_function(self, impulse_ref_edges, n_segments, n_bins_per_segment):
#        print 'impulse_ref_edges'
#        print impulse_ref_edges
        impulse = np.zeros(len(impulse_ref_edges))

        for i, edges in enumerate(impulse_ref_edges):
#            print 'edges'
#            print edges
            integral_from = edges[0] * self._scaling
            integral_to = edges[1] * self._scaling

            impulse[i], _ = integrate.quad(self._impulse_response, integral_from, integral_to)

        impulse = self._normalize(impulse_ref_edges, impulse)

        if self._zero_bin_value is not None:
            for i, edges in enumerate(impulse_ref_edges):
                if (edges[0] <= 0.) and (0. < edges[1]):
                    impulse[i] = impulse_ref_edges[i] + self._zero_bin_value

        cleaned_impulse = np.array([])
        target_segments = []

        for i in xrange(n_segments):
            i_from = i * n_bins_per_segment
            i_to = (i+1) * n_bins_per_segment

            if np.sum(np.abs(impulse[i_from:i_to])) > 0.:
                target_segments.append(i)
                cleaned_impulse = np.append(cleaned_impulse, impulse[i_from:i_to])


#        return [0], impulse
        return target_segments, cleaned_impulse

    def _normalize(self, impulse_ref_edges, impulse):

        if self._normalization is None:
            pass
        elif isinstance(self._normalization, tuple):
            if self._normalization[0] == 'integral':
                norm_coeff, _ = integrate.quad(self._impulse_response, self._normalization[1][0], self._normalization[1][1])
        elif self._normalization == 'sum':
            impulse = impulse/np.sum(impulse)

        else:
            raise ValueError('Unknown normalization method')

        return impulse
#
#        if self._normalization is None:
#            pass
#        elif isinstance(self._normalization, float):
#            impulse_values = impulse_values/self._normalization
#        elif isinstance(self._normalization, tuple):
#            if self._normalization[0] == 'bunch_by_bunch':
#                bunch_spacing = self._normalization[1] * c
#
#                bunch_locations = np.array([])
#                if (impulse_bin_edges[0,0] < 0):
#                    bunch_locations = np.append(bunch_locations, -1.*np.arange(0.,-1.*impulse_bin_edges[0,0],bunch_spacing))
#                if (impulse_bin_edges[-1,1] > 0):
#                    bunch_locations = np.append(bunch_locations, np.arange(0.,impulse_bin_edges[-1,1],bunch_spacing))
#
#                bunch_locations = np.unique(bunch_locations)
#
#                min_mask = (bunch_locations >= impulse_bin_edges[0,0])
#                max_mask = (bunch_locations <= impulse_bin_edges[-1,1])
#
#                bunch_locations = bunch_locations[min_mask*max_mask]
#
#                total_sum = 0.
#
#                # TODO: check, which is the best way to calculate the normalization coefficient
#                total_sum = np.sum(np.interp([bunch_locations], impulse_bin_mids, impulse_values))
##                for location in bunch_locations:
##                    min_mask = (impulse_bin_mids > (location - bunch_length/2.))
##                    max_mask = (impulse_bin_mids < (location + bunch_length/2.))
##
##                    total_sum += np.mean(impulse_values[min_mask*max_mask])
#
#                impulse_values = impulse_values/total_sum
#
#            else:
#                raise ValueError('Unknown normalization method')
#
#        elif self._normalization == 'max':
#            impulse_values = impulse_values/np.max(impulse_values)
#        elif self._normalization == 'min':
#            impulse_values = impulse_values/np.min(impulse_values)
#        elif self._normalization == 'average':
#            impulse_values = impulse_values/np.abs(np.mean(impulse_values))
#        elif self._normalization == 'sum':
#            # TODO: check naming, this is not a sum, but an integral?
#            impulse_values = impulse_values/np.abs(np.sum(impulse_values))
#        elif self._normalization == 'integral':
#            bin_widths = impulse_bin_edges[:,1]-impulse_bin_edges[:,0]
#            impulse_values = impulse_values / np.abs(np.sum(impulse_values*bin_widths))
#        else:
#            raise ValueError('Unknown normalization method')
#
#        if self._zero_bin_value is not None:
#            for i, edges in enumerate(impulse_bin_edges):
#                if (edges[0] <= 0.) and (0. < edges[1]):
#                    impulse_values[i] = impulse_values[i] + self._zero_bin_value
#
#        return impulse_values

    @abstractmethod
    def _raw_impulse_response(self, x):
        """ Impulse response of the filter.
        :param x: normalized time (t*2.*pi*f_c)
        :return: response at the given time
        """
        pass

    def _impulse_response_generator(self,tip_cut_width):
        """ A function which generates the response function from the raw impulse response. If 2nd cut-off frequency
            is given, the value of the raw impulse response is set to constant at the time scale below that.
            The integral over the response function is normalized to value 1.
        """

        if tip_cut_width is not None:
            def transfer_function(x):
                if np.abs(x) < tip_cut_width:
                    return self._raw_impulse_response(np.sign(x)*tip_cut_width)
                else:
                    return self._raw_impulse_response(x)
        else:
            def transfer_function(x):
                    return self._raw_impulse_response(x)

        return transfer_function


class Lowpass(ConvolutionFilter):
    def __init__(self,f_cutoff, impulse_length = 5., f_cutoff_2nd = None, normalization=('integral',(-5.,5.)), **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (0, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__(scaling, impulse_range, tip_cut_width = tip_cut_width, normalization=normalization,**kwargs)
        self.label = 'Lowpass filter'

    def _raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return math.exp(-1. * x)

class Highpass(ConvolutionFilter):
    def __init__(self,f_cutoff, impulse_length = 5., f_cutoff_2nd = None, normalization=('integral',(-5.,5.)), **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (0, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, zero_bin_value= 1., tip_cut_width = tip_cut_width, normalization=normalization, **kwargs)
        self.label = 'Highpass filter'

    def _raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return -1.* math.exp(-1. * x)

class PhaseLinearizedLowpass(ConvolutionFilter):
    def __init__(self, f_cutoff, impulse_length = 5., f_cutoff_2nd = None, normalization=('integral',(-5.,5.)), **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (-1.*impulse_length/scaling, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, normalization=normalization, **kwargs)
        self.label = 'Phaselinearized lowpass filter'

    def _raw_impulse_response(self, x):
        if x == 0.:
            return 0.
        else:
            return special.k0(abs(x))


class Gaussian(ConvolutionFilter):
    def __init__(self, f_cutoff, impulse_length = 5., normalization=('integral',(-5.,5.)), **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (-1.*impulse_length/scaling, impulse_length/scaling)


        tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, normalization=normalization, **kwargs)
        self.label = 'Gaussian lowpass filter'

    def _raw_impulse_response(self, x):
        return np.exp(-x ** 2. / 2.) / np.sqrt(2. * pi)


class Sinc(ConvolutionFilter):
    """ A nearly ideal lowpass filter, i.e. a windowed Sinc filter. The impulse response of the ideal lowpass filter
        is Sinc function, but because it is infinite length in both positive and negative time directions, it can not be
        used directly. Thus, the length of the impulse response is limited by using windowing. Properties of the filter
        depend on the width of the window and the type of the windows and must be written down. Too long window causes
        ripple to the signal in the time domain and too short window decreases the slope of the filter in the frequency
        domain. The default values are a good compromise. More details about windowing can be found from
        http://www.dspguide.com/ch16.htm and different options for the window can be visualized, for example, by using
        code in example/test 004_analog_signal_processors.ipynb
    """

    def __init__(self, f_cutoff, window_width = 3, window_type = 'blackman', normalization=('integral',(-10.,10.)), **kwargs):
        """
        :param f_cutoff: a cutoff frequency of the filter
        :param delay: a delay of the filter [s]
        :param window_width: a (half) width of the window in the units of zeros of Sinc(x) [2*pi*f_c]
        :param window_type: a shape of the window, blackman or hamming
        :param norm_type: see class LinearTransform
        :param norm_range: see class LinearTransform
        """

        scaling = 2. * pi * f_cutoff / c

        self.window_width = float(window_width)
        self.window_type = window_type
        impulse_range = (-1.*pi *window_width/scaling, pi*window_width/scaling)
        super(self.__class__, self).__init__(scaling, impulse_range,normalization=normalization, **kwargs)
        self.label = 'Sinc filter'

    def _raw_impulse_response(self, x):
        if np.abs(x/pi) > self.window_width:
            return 0.
        else:
            if self.window_type == 'blackman':
                return np.sinc(x/pi)*self.blackman_window(x)
            elif self.window_type == 'hamming':
                return np.sinc(x/pi)*self.hamming_window(x)

    def blackman_window(self,x):
        return 0.42-0.5*np.cos(2.*pi*(x/pi+self.window_width)/(2.*self.window_width))\
               +0.08*np.cos(4.*pi*(x/pi+self.window_width)/(2.*self.window_width))

    def hamming_window(self, x):
        return 0.54-0.46*np.cos(2.*pi*(x/pi+self.window_width)/(2.*self.window_width))



class FIRFilter(Convolution):

    def __init__(self, coefficients, zero_tap = 0, **kwargs):

        self._zero_tap = zero_tap

        self._input_coefficients = coefficients




        super(FIRFilter, self).__init__(None, **kwargs)
        self.label = 'FIR filter'


    def response_function(self, impulse_ref_edges, n_segments, n_bins_per_segment):
#        print 'impulse_ref_edges'
#        print impulse_ref_edges
        impulse = np.zeros(len(impulse_ref_edges))
        impulse_bin_widths = bin_widths(impulse_ref_edges)
        impulse_bin_width = np.mean(impulse_bin_widths)
        impulse_bin_mids = bin_mids(impulse_ref_edges)

        n_coefficients = len(self._input_coefficients)
        min_filter_idx = -1*self._zero_tap
        max_filter_idx = min_filter_idx + n_coefficients -1
#        print 'min_filter_idx: ' + str(min_filter_idx)
#        print 'max_filter_idx: ' + str(max_filter_idx)

        for i, mid in enumerate(impulse_bin_mids):
            filter_idx = mid/impulse_bin_width
#            print 'filter_idx: ' + str(filter_idx)
            filter_idx = int(np.round(filter_idx))
#            print 'filter_idx: ' + str(filter_idx)

            if (filter_idx >= min_filter_idx) and (filter_idx <= max_filter_idx):
                impulse[i] = self._input_coefficients[filter_idx+self._zero_tap]


        cleaned_impulse = np.array([])
        target_segments = []

        for i in xrange(n_segments):
            i_from = i * n_bins_per_segment
            i_to = (i+1) * n_bins_per_segment

            if np.sum(np.abs(impulse[i_from:i_to])) > 0.:
                target_segments.append(i)
                cleaned_impulse = np.append(cleaned_impulse, impulse[i_from:i_to])

        return target_segments, cleaned_impulse