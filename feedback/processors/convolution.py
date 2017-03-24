import math, copy
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import signal
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import UnivariateSpline

# TODO: Delay
# TODO: Jitter


class Impulse(object):
    """
        An objects, which generates an impulse for a bunche and receives impulses from other bunches affecting
        corresponding bunch.
    """

    def __init__(self,bunch_idx, output_signal, output_signal_limits, impulse_response, impulse_response_limits):
        """
        :param bunch_idx: an unique list index for the Impulse
        :param output_signal: np.array where the output signal is stored
        :param output_signal_limits: limits of the output signal in the units of z [m]
        :param impulse_response: a numpy array, which contains impulse response values
        :param impulse_response_limits: limits of the impulse response in the units of z [m]. The zero position is
                    on the zero time of the impulse ("normalized distance")
        """

        self._bunch_idx = bunch_idx

        self._output_signal = output_signal
        self._output_signal_length = len(self._output_signal)
        self._output_signal_limits = output_signal_limits

        self._impulse_response = impulse_response

        self._total_impulse = np.zeros(len(self._impulse_response) + len(self._output_signal) - 1)
        self._total_impulse_length = len(self._total_impulse)
        self._total_impulse_limits = (self._output_signal_limits[0] + impulse_response_limits[0],
                                     self._output_signal_limits[1] + impulse_response_limits[1])

        self._bin_spacing = (self._output_signal_limits[1] -
                             self._output_signal_limits[0]) / float(len(self._output_signal))

        self.signal_views = []
        self.impulse_views = []
        self.target_bunches = []

    def build_impulse(self,input_signal):
        """
        :param input_signal: a part of the total input signal which corresponds to the signal of this bunch
        """
        np.copyto(self._total_impulse,np.convolve(self._impulse_response,input_signal))

    def check_if_target(self,target_object_idx,target_impulse_object):
        """
        This function checks if the impulse response of this bunch overlaps with the signal of the target bunch. In that
        case the necessary memory views for the impulse and the target are created.

        :param target_idx: a list index of the target bunch
        :param bunch_impulse_target: a bunch_impulse-object for the target bunch
        """

        signal_edges = target_impulse_object.signal_limits
        impulse_edges = self.impulse_limits
        max_signal_length = target_impulse_object.max_signal_length
        max_impulse_length = self.max_impulse_length

        if not ((impulse_edges[1] <= signal_edges[0]) or (impulse_edges[0] >= signal_edges[1])):

            # TODO: check rounding errors here
            if impulse_edges[0] <= signal_edges[0]:
                idx_target_from = 0
                idx_impulse_from = int(round((signal_edges[0]-impulse_edges[0])/self._bin_spacing))
            else:
                idx_target_from = int(round((impulse_edges[0]-signal_edges[0])/self._bin_spacing))
                idx_impulse_from = 0

            if impulse_edges[1] <= signal_edges[1]:
                idx_impulse_to = max_impulse_length
                idx_target_to = idx_target_from + idx_impulse_to - idx_impulse_from
            else:
                idx_target_to = max_signal_length
                idx_impulse_to = idx_impulse_from + idx_target_to - idx_target_from

            self.impulse_views.append(np.array(self._total_impulse[idx_impulse_from:idx_impulse_to], copy=False))
            self.target_bunches.append(target_object_idx)

            target_impulse_object.add_signal_view(self._bunch_idx,idx_target_from,idx_target_to)

    def add_signal_view(self,bunch_idx,idx_from,idx_to):
        """
        :param bunch_idx: a list index of the bunch giving the impulse
        :param idx_from: an index from where the impulse starts
        :param idx_to: an index where to the impulse ends
        """

        while len(self.signal_views) < (bunch_idx + 1):
            self.signal_views.append(None)

        self.signal_views[bunch_idx] = np.array(self._output_signal[idx_from:idx_to], copy=False)

    @property
    def max_impulse_length(self):
        return self._total_impulse_length

    @property
    def max_signal_length(self):
        return self._output_signal_length

    @property
    def impulse_limits(self):
        return self._total_impulse_limits

    @property
    def signal_limits(self):
        return self._output_signal_limits


class Convolution(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on convolution.
    """

    # TODO: store impulses

    def __init__(self, impulse_range, store_signal = False):

        self._impulse_range = impulse_range

        self.signal_classes = (1, 1)

        self.extensions = ['store']
        self._store_signal = store_signal

        self._n_bins_per_segment = None
        self._n_segments = None

        self._impulse_bin_edges = None
        self._impulse_bin_mids = None
        self._impulse_values = None

        self._bin_spacing = None

        self._impulse_objects = None

        self._store_signal  = store_signal

        self.input_signal = None
        self.input_parameters = None

        self.output_signal = None
        self.output_parameters = None


    def process(self, parameters, signal, *args, **kwargs):

        if self.output_signal is None:
            self.__init_variables(parameters['bin_edges'],signal,parameters['n_segments'],
                                  parameters['n_bins_per_segment'])
        else:
            self.output_signal.fill(0.)

        for i,impulse_object in enumerate(self._impulse_objects):

            signal_from = i*self._n_bins_per_segment
            signal_to = (i+1)*self._n_bins_per_segment
            impulse_object.build_impulse(signal[signal_from:signal_to])

            for target_bunch, impulse_view in zip(impulse_object.target_bunches, impulse_object.impulse_views):
                if self._impulse_objects[target_bunch].signal_views[i] is not None:
                    self._impulse_objects[target_bunch].signal_views[i] += impulse_view
                    # self._impulse_objects[target_bunch].signal_views[i] = np.add(self._impulse_objects[target_bunch].signal_views[i],impulse_view)
                else:
                    raise ValueError('Memviews are not synchronized!')

        if self._store_signal:
            self.input_signal = copy.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_parameters = copy.copy(parameters)

        return parameters, self.output_signal

    def __init_variables(self,bin_edges,signal,n_segments,n_bins_per_segment):

        # generates variables
        self._n_segments = n_segments
        self._n_bins_per_segment = n_bins_per_segment
        self.output_signal = np.zeros(len(signal))
        self._bin_spacing = np.mean(bin_edges[0:self._n_bins_per_segment,1]-bin_edges[0:self._n_bins_per_segment,0])

        # generates an impulse response
        if self._impulse_range[0] < -0.5*self._bin_spacing:
            temp = np.arange(0.5*self._bin_spacing,-1.*self._impulse_range[0] + self._bin_spacing,self._bin_spacing)
            z_bins_minus = -1.*temp[::-1]
        else:
            z_bins_minus = np.array([-0.5 * self._bin_spacing])

        if self._impulse_range[1] > 0.5*self._bin_spacing:
            z_bins_plus = np.arange(0.5*self._bin_spacing,self._impulse_range[1] + self._bin_spacing,self._bin_spacing)
        else:
            z_bins_plus = np.array([0.5*self._bin_spacing])

        # TODO: change naming and also z bins to standard bin edges
        impulse_z_bins = np.append(z_bins_minus,z_bins_plus)
        impulse_z_bins = impulse_z_bins[impulse_z_bins >= (self._impulse_range[0] - 0.5*self._bin_spacing)]
        impulse_z_bins = impulse_z_bins[impulse_z_bins <= (self._impulse_range[1] + 0.5*self._bin_spacing)]
        self._impulse_bin_edges = np.transpose(np.array([impulse_z_bins[:-1],impulse_z_bins[1:]]))
        self._impulse_bin_mids = (impulse_z_bins[1:]+impulse_z_bins[:-1])/2.
        self._impulse_values = self.calculate_response(self._impulse_bin_mids,self._impulse_bin_edges)

        # generates bunch impulses
        self._impulse_objects = []

        for i in xrange(self._n_segments):
            idx_from = i * self._n_bins_per_segment
            idx_to = (i + 1) * self._n_bins_per_segment

            impulse_limits = (self._impulse_bin_edges[0,0]+ 0.5*self._bin_spacing, self._impulse_bin_edges[-1,1]+ 0.5*self._bin_spacing)
            signal_from = bin_edges[self._n_bins_per_segment * i,0]
            signal_to = bin_edges[self._n_bins_per_segment * (i + 1)-1, 1]

            signal_limits = (signal_from, signal_to)

            self._impulse_objects.append(Impulse(i,np.array(self.output_signal[idx_from:idx_to], copy=False),
                                                signal_limits, self._impulse_values, impulse_limits))

        # checks if the impulses overlap the signals of the bunches
        for i, bunch_impulse in enumerate(self._impulse_objects):
            for j, bunch_impulse_target in enumerate(self._impulse_objects):
                bunch_impulse.check_if_target(j,bunch_impulse_target)

    @abstractmethod
    def calculate_response(self, impulse_bin_mids, impulse_bin_edges):
        # Impulse response function of the processor
        pass


class Delay(Convolution):
    def __init__(self,delay, **kwargs):

        self._z_delay = delay*c

        if self._z_delay < 0.:
            impulse_range = (self._z_delay, 0.)
        else:
            impulse_range = (0., self._z_delay)

        super(self.__class__, self).__init__(impulse_range, **kwargs)
        self.label = 'Delay'

    def calculate_response(self, impulse_bin_mids, impulse_bin_edges):
        impulse_values = np.zeros(len(impulse_bin_mids))
        bin_spacing =  np.mean(impulse_bin_edges[:,1]-impulse_bin_edges[:,0])

        ref_bin_from = -0.5*bin_spacing+self._z_delay
        ref_bin_to = 0.5*bin_spacing+self._z_delay

        for i, edges in enumerate(impulse_bin_edges):
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

    def calculate_response(self, impulse_bin_mids, impulse_bin_edges):
        impulse_values = np.zeros(len(impulse_bin_mids))

        for i, edges in enumerate(impulse_bin_edges):
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


class WaveletGenerator(Convolution):

    def __init__(self,spacing,n_copies, **kwargs):
        self._spacing = spacing
        self._n_copies = n_copies

        if isinstance(self._n_copies,tuple):
            self._i_from = self._n_copies[0]
            self._i_to = self._n_copies[1]

        else:
            self._i_from = min(self._n_copies,0)
            self._i_to = max(self._n_copies,0)

        self._window = (self._i_from*self._spacing*c,self._i_to*self._spacing*c)

        super(self.__class__, self).__init__(self._window, **kwargs)
        self.label = 'Wavelet generator'


    def calculate_response(self, impulse_bin_mids, impulse_bin_edges):

        bin_spacing = np.mean(impulse_bin_edges[:,1]-impulse_bin_edges[:,0])
        impulse_values = np.zeros(len(impulse_bin_mids))

        for i in xrange(self._i_from,(self._i_to+1)):
            copy_mid = i*self._spacing*c
            copy_from = copy_mid - 0.5 * bin_spacing
            copy_to = copy_mid + 0.5 * bin_spacing

            for j, edges in enumerate(impulse_bin_edges):
                impulse_values[j] += (self._CDF(edges[1],copy_from,copy_to)-self._CDF(edges[0],copy_from,copy_to))

        return impulse_values

    def _CDF(self, x, ref_bin_from, ref_bin_to):
        if x <= ref_bin_from:
            return 0.
        elif x < ref_bin_to:
            return (x - ref_bin_from) / float(ref_bin_to - ref_bin_from)
        else:
            return 1.

class ConvolutionFromFile(Convolution):
    """ Interpolates matrix columns by using inpulse response data from a file. """

    def __init__(self,filename, x_axis = 'time', calc_type = 'mean',  **kwargs):
        self._filename = filename
        self._x_axis = x_axis
        self._calc_type = calc_type

        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0]=self._data[:, 0]*c

        impulse_range = (self._data[0,0],self._data[-1,0])

        super(self.__class__, self).__init__(impulse_range, **kwargs)
        self.label = 'Convolution from external data'

    def calculate_response(self, impulse_response_bin_mid, impulse_response_bin_edges):

        if self._calc_type == 'mean':
            return np.interp(impulse_response_bin_mid, self._data[:, 0], self._data[:, 1])
        elif self._calc_type == 'integral':
            s = UnivariateSpline(self._data[:, 0], self._data[:, 1])
            response_values = np.zeros(len(impulse_response_bin_mid))

            for i, edges in enumerate(impulse_response_bin_edges):
                response_values[i], _ = s.integral(edges[0],edges[1])
            return response_values

        else:
            raise ValueError('Unknown value in ConvolutionFromFile._calc_type')

class ConvolutionFilter(Convolution):
    __metaclass__ = ABCMeta

    def __init__(self,scaling,impulse_range,zero_bin_value = None, tip_cut_width=None, normalization=None, norm_range=None, **kwargs):

        self._scaling = scaling
        self._normalization = normalization
        self._norm_range = norm_range
        self._zero_bin_value = zero_bin_value
        super(ConvolutionFilter, self).__init__(impulse_range, **kwargs)

        # NOTE: is the tip cut needed? How to work with the sharp tips of the ideal filters?
        if (self._normalization is None) and (tip_cut_width is not None):
            self._normalization = 'integral'
        self._impulse_response = self._impulse_response_generator(tip_cut_width)

    def calculate_response(self, impulse_bin_mids, impulse_bin_edges):

        # FIXME: take into account the symmetry of the impulse response

        impulse_values = np.zeros(len(impulse_bin_mids))

        for i, edges in enumerate(impulse_bin_edges):
            integral_from = edges[0] * self._scaling
            integral_to = edges[1] * self._scaling

            impulse_values[i], _ = integrate.quad(self._impulse_response, integral_from, integral_to)

        if self._normalization is None:
            pass
        elif isinstance(self._normalization, float):
            impulse_values = impulse_values/self._normalization
        elif isinstance(self._normalization, tuple):
            if self._normalization[0] == 'bunch_by_bunch':
                bunch_length = self._normalization[1] * c
                bunch_spacing = self._normalization[2] * c

                bunch_locations = np.array([])
                if (impulse_bin_edges[0,0] < 0):
                    bunch_locations = np.append(bunch_locations, -1.*np.arange(0.,-1.*impulse_bin_edges[0,0],bunch_spacing))
                if (impulse_bin_edges[-1,1] > 0):
                    bunch_locations = np.append(bunch_locations, np.arange(0.,impulse_bin_edges[-1,1],bunch_spacing))

                bunch_locations = np.unique(bunch_locations)

                min_mask = (bunch_locations >= impulse_bin_edges[0,0])
                max_mask = (bunch_locations <= impulse_bin_edges[-1,1])

                bunch_locations = bunch_locations[min_mask*max_mask]

                total_sum = 0.

                # TODO: check, which is the best way to calculate the normalization coefficient
                total_sum = np.sum(np.interp([bunch_locations], impulse_bin_mids, impulse_values))
#                for location in bunch_locations:
#                    min_mask = (impulse_bin_mids > (location - bunch_length/2.))
#                    max_mask = (impulse_bin_mids < (location + bunch_length/2.))
#
#                    total_sum += np.mean(impulse_values[min_mask*max_mask])

                impulse_values = impulse_values/total_sum

            else:
                raise ValueError('Unknown normalization method')

        elif self._normalization == 'max':
            impulse_values = impulse_values/np.max(impulse_values)
        elif self._normalization == 'min':
            impulse_values = impulse_values/np.min(impulse_values)
        elif self._normalization == 'average':
            impulse_values = impulse_values/np.abs(np.mean(impulse_values))
        elif self._normalization == 'sum':
            impulse_values = impulse_values/np.abs(np.sum(impulse_values))
        elif self._normalization == 'integral':
            bin_widths = impulse_bin_edges[:,1]-impulse_bin_edges[:,0]
            impulse_values = impulse_values / np.abs(np.sum(impulse_values*bin_widths))
        else:
            raise ValueError('Unknown normalization method')

        if self._zero_bin_value is not None:
            for i, edges in enumerate(impulse_bin_edges):
                if (edges[0] <= 0.) and (0. < edges[1]):
                    impulse_values[i] = impulse_values[i] + self._zero_bin_value

        return impulse_values

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
    def __init__(self,f_cutoff, impulse_length = 5., f_cutoff_2nd = None, **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (0, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, **kwargs)
        self.label = 'Lowpass filter'

    def _raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return math.exp(-1. * x)

class Highpass(ConvolutionFilter):
    def __init__(self,f_cutoff, impulse_length = 5., f_cutoff_2nd = None, **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (0, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, zero_bin_value= 1., tip_cut_width = tip_cut_width, **kwargs)
        self.label = 'Highpass filter'

    def _raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return -1.* math.exp(-1. * x)

class PhaseLinearizedLowpass(ConvolutionFilter):
    def __init__(self, f_cutoff, impulse_length = 5., f_cutoff_2nd = None, **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (-1.*impulse_length/scaling, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, **kwargs)
        self.label = 'Phaselinearized lowpass filter'

    def _raw_impulse_response(self, x):
        if x == 0.:
            return 0.
        else:
            return special.k0(abs(x))


class GaussianLowpass(ConvolutionFilter):
    def __init__(self, f_cutoff, impulse_length = 5., **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (-1.*impulse_length/scaling, impulse_length/scaling)


        tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, **kwargs)
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

    def __init__(self, f_cutoff, window_width = 3, window_type = 'blackman', **kwargs):
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
        super(self.__class__, self).__init__(scaling, impulse_range, **kwargs)
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



########################################################################################################################
####### PLAIN CONVOLUTION ##############################################################################################
########################################################################################################################

class FIRfilter(object):
    def __init__(self,coefficients, mode = 'bunch_by_bunch',store_signal=False):
        """ Filters the signal by convolving the signal and the input array of filter (FIR) coefficients
        :param coefficients: A numpy array of filter (convolution) coefficients
        """
        self._coefficients = coefficients
        self.required_variables = []
        self._mode = mode
        self._n_bins_per_segment = None
        self._n_segments = None
        self._bin_check = None

        if self._mode == 'bunch_by_bunch':
            self.signal_classes = (1, 1)
        elif self._mode == 'continuous':
            self.signal_classes = (2, 2)
        else:
            raise ValueError('Unknown mode for FIRfilter')

        self.extensions = ['store']
        self._store_signal = store_signal

        self.input_signal = None
        self.input_parameters = None

        self.output_signal = None
        self.output_parameters = None

        self.label = 'Digital filter'

    def process(self,parameters, signal, *args, **kwargs):

        if self._mode == 'bunch_by_bunch':

            if self._n_segments is None:
                self._n_segments = parameters['n_segments']
                self._n_bins_per_segment = parameters['n_bins_per_segment']
                self.output_signal = np.zeros(self._n_segments*self._n_bins_per_segment)


            for i in xrange(self._n_segments):
                i_from = i * self._n_bins_per_segment
                i_to = (i+1) * self._n_bins_per_segment
                temp_signal = signal[i_from:i_to]
                out_temp_signal =  np.convolve(temp_signal, self._coefficients, mode='same')
                np.copyto(self.output_signal[i_from:i_to],out_temp_signal)

        elif self._mode == 'continuous':
            self.output_signal =  np.convolve(np.array(signal), np.array(self._coefficients), mode='same')

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_bin_edges = copy.copy(parameters)

        return parameters, self.output_signal


class NumpyFIRfilter(FIRfilter):
    def __init__(self,n_taps, f_cutoffs, sampling_rate, **kwargs):

        """ A digital FIR (finite impulse response) filter, which uses firwin function from SciPy library to determine
            filter coefficients. Note that the set value of the cut-off frequency corresponds to the real cut-off
            frequency of the filter only when length of the signal is on the same order of longer than an period of
            the cut off frequency and sampling rate is ("significantly") higher than the cut-off frequency. In other
            words, do not trust too much to set value of the cut-off frequency,

        :param n_taps: length of the filter (number of coefficients, i.e. the filter order + 1).
            Odd number is recommended, when
        :param f_cutoffs: cut-off frequencies of the filter. Multiple values are allowed as explained in
            the documentation of firwin-function in SciPy
        :param sampling_rate: sampling rate of the ADC (or a number of slices per seconds)
        """

        self._n_taps = n_taps
        self._f_cutoffs = f_cutoffs
        self._nyq = sampling_rate / 2.

        coefficients = signal.firwin(self._n_taps, self._f_cutoffs, nyq=self._nyq)

        super(self.__class__, self).__init__(coefficients, **kwargs)

        self.label = 'FIR filter'


