import itertools
import math
import copy
from collections import deque
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
from scipy import linalg
from cython_hacks import cython_matrix_product
from ..core import debug_extension

# TODO: clean code here!

class LinearTransform(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on linear transformation. The signal is processed by
        calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced with an abstract
        method, namely response_function(*args), which returns an elements of the matrix (an effect of
        the ref_bin to the bin)
    """

    def __init__(self, mode = 'bunch_by_bunch', normalization=None,
                 bin_middle = 'bin', label = 'LinearTransform', **kwargs):
        """

        :param norm_type: Describes normalization method for the transfer matrix
            'bunch_average':    an average value over the bunch is equal to 1
            'fixed_average':    an average value over a range given in a parameter norm_range is equal to 1
            'bunch_integral':   an integral over the bunch is equal to 1
            'fixed_integral':   an integral over a fixed range given in a parameter norm_range is equal to 1
            'matrix_sum':       a sum over elements in the middle column of the matrix is equal to 1
            None:               no normalization
        :param norm_range: Normalization length in cases of self.norm_type == 'fi
        xed_length_average' or
            self.norm_type == 'fixed_length_integral'
        :param bin_check: if True, a change of the bin_set is checked every time process() is called and matrix is
            recalculated if any change is found
        :param bin_middle: defines if middle points of the bins are determined by a middle point of the bin
            (bin_middle = 'bin') or an average place of macro particles (bin_middle = 'particles')
        """

        self._mode = mode

        self._normalization = normalization
        self._bin_middle = bin_middle

        self._z_bin_set = None
        self._matrix = None

        self._recalculate_matrix = True

        self.signal_classes = (0,0)

        self._n_segments = None
        self._n_bins_per_segment = None
        self._mid_bunch = None

        self.extensions = ['debug']
        if bin_middle == 'particles':
            self.extensions.append('bunch')
            self.required_variables = ['mean_z']

        self._extension_objects = [debug_extension(self, label, **kwargs)]



    @abstractmethod
    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Impulse response function of the processor
        pass

    def process(self,parameters, signal, slice_sets = None, *args, **kwargs):

        if self._matrix is None:

            if self._bin_middle == 'particles':
                bin_midpoints = np.array([])
                for slice_set in slice_sets:
                    bin_midpoints = np.append(bin_midpoints, slice_set.mean_z)
            elif self._bin_middle == 'bin':
                bin_midpoints = (parameters['bin_edges'][:, 1] + parameters['bin_edges'][:, 0]) / 2.
            else:
                raise ValueError('Unknown value for LinearTransform._bin_middle ')

            self._n_segments = parameters['n_segments']
            self._n_bins_per_segment = parameters['n_bins_per_segment']

            self.__generate_matrix(parameters['bin_edges'],bin_midpoints)

        if self._mode == 'total':
            output_signal = np.array(cython_matrix_product(self._matrix, signal))
        elif self._mode == 'bunch_by_bunch':
            output_signal = np.zeros(len(signal))

            for i in xrange(self._n_segments):
                idx_from = i * self._n_bins_per_segment
                idx_to = (i+1) * self._n_bins_per_segment
                np.copyto(output_signal[idx_from:idx_to],cython_matrix_product(self._matrix, signal[idx_from:idx_to]))
        else:
            raise ValueError('Unknown value for LinearTransform._mode ')

        for extension in self._extension_objects:
            extension(self, parameters, signal, parameters, output_signal,
                      *args, **kwargs)

        return parameters, output_signal

        # np.dot can't be used, because it slows down the calculations in LSF by a factor of two or more
        # return np.dot(self._matrix,signal)

    def clear(self):
        self._matrix = np.array([])
        self._recalculate_matrix = True

    def print_matrix(self):
        for row in self._matrix:
            print "[",
            for element in row:
                print "{:6.3f}".format(element),
            print "]"

    def __generate_matrix(self,bin_edges, bin_midpoints):

        self._mid_bunch = int(self._n_segments/2)

        bunch_mid = (bin_edges[0,0]+bin_edges[(self._n_bins_per_segment - 1),1]) / 2.

        total_mid = bin_midpoints[int(len(bin_midpoints)/2)]

        norm_bunch_midpoints = bin_midpoints[:self._n_bins_per_segment]
        norm_bunch_midpoints = norm_bunch_midpoints - bunch_mid
        norm_bin_edges = bin_edges[:self._n_bins_per_segment]
        norm_bin_edges = norm_bin_edges - bunch_mid

        bin_spacing = np.mean(norm_bin_edges[:, 1] - norm_bin_edges[:, 0])

        if self._mode == 'bunch_by_bunch':

            self._matrix = np.identity(len(norm_bunch_midpoints))

            for i, midpoint_i in enumerate(norm_bunch_midpoints):
                for j, midpoint_j in enumerate(norm_bunch_midpoints):
                    self._matrix[j][i] = self.response_function(midpoint_i,norm_bin_edges[i,0],norm_bin_edges[i,1],
                                                                midpoint_j,norm_bin_edges[j,0],norm_bin_edges[j,1])
        elif self._mode == 'total':
            self._matrix = np.identity(len(bin_midpoints))
            for i, midpoint_i in enumerate(bin_midpoints):
                for j, midpoint_j in enumerate(bin_midpoints):
                    self._matrix[j][i] = self.response_function(midpoint_i, bin_edges[i, 0], bin_edges[i, 1],
                                                                midpoint_j, bin_edges[j, 0], bin_edges[j, 1])

        else:
            raise ValueError('Unrecognized value in LinearTransform._mode')

        matrix_size = self._matrix.shape

        total_impulse = np.append(self._matrix[:,-1],self._matrix[1:,0])
        bin_widths = bin_edges[:, 1]-bin_edges[:, 0]
        total_bin_widths = np.append(bin_widths,bin_widths[1:])

        if self._normalization is None:
            pass
        elif self._normalization == 'max':
            self._matrix = self._matrix/np.max(total_impulse)
        elif self._normalization == 'min':
            self._matrix = self._matrix/np.min(total_impulse)
        elif self._normalization == 'average':
            self._matrix = self._matrix/np.abs(np.mean(total_impulse))
        elif self._normalization == 'sum':
            self._matrix = self._matrix/np.abs(np.sum(total_impulse))
        elif self._normalization == 'column_sum':
            self._matrix = self._matrix/np.abs(np.sum(self._matrix[:,0]))
        elif self._normalization == 'integral':
            self._matrix = self._matrix / np.abs(np.sum(total_impulse* total_bin_widths))
        else:
            raise ValueError('Unrecognized value in LinearTransform._normalization')

class Averager(LinearTransform):
    """ Returns a signal, which consists an average value of the input signal. A sums of the rows in the matrix
        are normalized to be one (i.e. a sum of the input signal doesn't change).
    """

    def __init__(self, mode = 'bunch_by_bunch', normalization = 'column_sum', **kwargs):
        super(self.__class__, self).__init__(mode, normalization, **kwargs)
        self.label = 'Averager'

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        return 1

class Delay(LinearTransform):
    """ Delays signal in the units of [second].
    """
    def __init__(self,delay, **kwargs):
        self._delay = delay
        super(self.__class__, self).__init__( **kwargs)
        self.label = 'Delay'

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):

        return self.__CDF(bin_to, ref_bin_from, ref_bin_to) - self.__CDF(bin_from, ref_bin_from, ref_bin_to)

    def __CDF(self,x,ref_bin_from, ref_bin_to):
        if (x-self._delay*c) <= ref_bin_from:
            return 0.
        elif (x-self._delay*c) < ref_bin_to:
            return ((x-self._delay*c)-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.

class LinearTransformFromFile(LinearTransform):
    """ Interpolates matrix columns by using inpulse response data from a file. """

    def __init__(self,filename, x_axis = 'time', **kwargs):
        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0]=self._data[:, 0]*c

        super(self.__class__, self).__init__( **kwargs)
        self.label = 'LT from file'

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
            return np.interp(bin_mid - ref_bin_mid, self._data[:, 0], self._data[:, 1])


class LtFilter(LinearTransform):
    __metaclass__ = ABCMeta
    """ A general class for (analog) filters. Impulse response of the filter must be determined by overwriting
        the function raw_impulse_response.

        This processor includes two additional properties.

    """

    def __init__(self, filter_type, filter_symmetry,f_cutoff, delay = 0., f_cutoff_2nd = None, bunch_spacing = None
                 , **kwargs):
        """
        :param filter_type: Options are:
                'lowpass'
                'highpass'
        :param f_cutoff: a cut-off frequency of the filter [Hz]
        :param delay: a delay in the units of seconds
        :param f_cutoff_2nd: a second cutoff frequency [Hz], which is implemented by cutting the tip of the impulse
                    response function
        :param norm_type: see class LinearTransform
        :param norm_range: see class LinearTransform
        """


        self._bunch_spacing = bunch_spacing
        self._f_cutoff = f_cutoff
        self._delay_z = delay * c
        self._filter_type = filter_type
        self._filter_symmetry = filter_symmetry

        self._impulse_response = self.__impulse_response_generator(f_cutoff_2nd)
        super(LtFilter, self).__init__(**kwargs)


        self._CDF_time = None
        self._CDF_value = None
        self._PDF = None


    @abstractmethod
    def raw_impulse_response(self, x):
        """ Impulse response of the filter.
        :param x: normalized time (t*2.*pi*f_c)
        :return: response at the given time
        """
        pass

    def __impulse_response_generator(self,f_cutoff_2nd):
        """ A function which generates the response function from the raw impulse response. If 2nd cut-off frequency
            is given, the value of the raw impulse response is set to constant at the time scale below that.
            The integral over the response function is normalized to value 1.
        """

        if f_cutoff_2nd is not None:
            threshold_tau = (2.*pi * self._f_cutoff) / (2.*pi * f_cutoff_2nd)
            threshold_val_neg = self.raw_impulse_response(-1.*threshold_tau)
            threshold_val_pos = self.raw_impulse_response(threshold_tau)
            integral_neg, _ = integrate.quad(self.raw_impulse_response, -100., -1.*threshold_tau)
            integral_pos, _ = integrate.quad(self.raw_impulse_response, threshold_tau, 100.)

            norm_coeff = np.abs(integral_neg + integral_pos + (threshold_val_neg + threshold_val_pos) * threshold_tau)

            def transfer_function(x):
                if np.abs(x) < threshold_tau:
                    return self.raw_impulse_response(np.sign(x)*threshold_tau) / norm_coeff
                else:
                    return self.raw_impulse_response(x) / norm_coeff
        else:
            norm_coeff, _ = integrate.quad(self.raw_impulse_response, -100., 100.)
            norm_coeff = np.abs(norm_coeff)
            def transfer_function(x):
                    return self.raw_impulse_response(x) / norm_coeff

        return transfer_function


    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Frequency scaling must be done by scaling integral limits, because integration by substitution doesn't work
        # with np.quad (see quad_problem.ipynbl). An ugly way, which could be fixed.

        scaling = 2. * pi * self._f_cutoff / c
        temp, _ = integrate.quad(self._impulse_response, scaling * (bin_from - (ref_bin_mid + self._delay_z)),
                                 scaling * (bin_to - (ref_bin_mid + self._delay_z)))

        if ref_bin_mid == bin_mid:
            if self._filter_type == 'highpass':
                temp += 1.

        return temp

class Sinc(LtFilter):
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
        self.window_width = float(window_width)
        self.window_type = window_type
        super(self.__class__, self).__init__('lowpass', 'symmetric', f_cutoff, **kwargs)
        self.label = 'Sinc filter'

    def raw_impulse_response(self, x):
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


class Lowpass(LtFilter):
    """ Classical first order lowpass filter (e.g. a RC filter), which impulse response can be described as exponential
        decay.
        """
    def __init__(self, f_cutoff, **kwargs):
        super(self.__class__, self).__init__('lowpass','delay', f_cutoff, **kwargs)
        self.label = 'Lowpass filter'

    def raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return math.exp(-1. * x)

class Highpass(LtFilter):
    """The classical version of a highpass filter, which """
    def __init__(self, f_cutoff, **kwargs):
        super(self.__class__, self).__init__('highpass','advance', f_cutoff, **kwargs)
        self.label = 'Highpass filter'

    def raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return -1.*math.exp(-1. * x)

class PhaseLinearizedLowpass(LtFilter):
    def __init__(self, f_cutoff, **kwargs):
        super(self.__class__, self).__init__('lowpass','symmetric', f_cutoff, **kwargs)
        self.label = 'Phaselinearized lowpass filter'

    def raw_impulse_response(self, x):
        if x == 0.:
            return 0.
        else:
            return special.k0(abs(x))