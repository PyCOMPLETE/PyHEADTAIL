from collections import deque
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
from scipy import linalg
from .cython_hacks import cython_matrix_product
from ..core import default_macros
from . import abstract_filter_responses

"""Signal processors based on linear transformation.

@author Jani Komppula
@date: 11/10/2017
"""

class LinearTransform(object, metaclass=ABCMeta):
    """ An abstract class for signal processors which are based on linear transformation. The signal is processed by
        calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced with an abstract
        method, namely response_function(*args), which returns an elements of the matrix (an effect of
        the ref_bin to the bin)
    """

    def __init__(self, mode = 'bunch_by_bunch', normalization=None, bin_middle = 'bin', **kwargs):
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

        self.extensions = []
        self._macros = [] + default_macros(self, 'LinearTransform', **kwargs)

        if bin_middle == 'particles':
            self.extensions.append('bunch')
            self.required_variables = ['mean_z']


    @abstractmethod
    def response_function(self, parameters, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
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

            self.__generate_matrix(parameters, parameters['bin_edges'],bin_midpoints)

        if self._mode == 'total':
            output_signal = np.array(cython_matrix_product(self._matrix, signal))
        elif self._mode == 'bunch_by_bunch':
            output_signal = np.zeros(len(signal))

            for i in range(self._n_segments):
                idx_from = i * self._n_bins_per_segment
                idx_to = (i+1) * self._n_bins_per_segment
                np.copyto(output_signal[idx_from:idx_to],cython_matrix_product(self._matrix, signal[idx_from:idx_to]))
        else:
            raise ValueError('Unknown value for LinearTransform._mode ')

        return parameters, output_signal

        # np.dot can't be used, because it slows down the calculations in LSF by a factor of two or more
        # return np.dot(self._matrix,signal)

    def clear(self):
        self._matrix = np.array([])
        self._recalculate_matrix = True

    def print_matrix(self):
        for row in self._matrix:
            print("[", end=' ')
            for element in row:
                print("{:6.3f}".format(element), end=' ')
            print("]")

    def __generate_matrix(self,parameters, bin_edges, bin_midpoints):

        self._mid_bunch = int(self._n_segments/2)

        bunch_mid = (bin_edges[0,0]+bin_edges[(self._n_bins_per_segment - 1),1]) / 2.

        norm_bunch_midpoints = bin_midpoints[:self._n_bins_per_segment]
        norm_bunch_midpoints = norm_bunch_midpoints - bunch_mid
        norm_bin_edges = bin_edges[:self._n_bins_per_segment]
        norm_bin_edges = norm_bin_edges - bunch_mid

        if self._mode == 'bunch_by_bunch':

            self._matrix = np.identity(len(norm_bunch_midpoints))

            for i, midpoint_i in enumerate(norm_bunch_midpoints):
                for j, midpoint_j in enumerate(norm_bunch_midpoints):
                    self._matrix[j][i] = self.response_function(parameters,
                                                                midpoint_i,norm_bin_edges[i,0],norm_bin_edges[i,1],
                                                                midpoint_j,norm_bin_edges[j,0],norm_bin_edges[j,1])
        elif self._mode == 'total':
            self._matrix = np.identity(len(bin_midpoints))
            for i, midpoint_i in enumerate(bin_midpoints):
                for j, midpoint_j in enumerate(bin_midpoints):
                    self._matrix[j][i] = self.response_function(parameters,
                                                                midpoint_i, bin_edges[i, 0], bin_edges[i, 1],
                                                                midpoint_j, bin_edges[j, 0], bin_edges[j, 1])

        else:
            raise ValueError('Unrecognized value in LinearTransform._mode')

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

    def response_function(self, parameters, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        return 1

class Delay(LinearTransform):
    """ Delays signal in the units of [second].
    """
    def __init__(self,delay, **kwargs):
        self._delay = delay
        super(self.__class__, self).__init__( **kwargs)
        self.label = 'Delay'

    def response_function(self, parameters, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):

        return self.__CDF(bin_to, ref_bin_from, ref_bin_to) - self.__CDF(bin_from, ref_bin_from, ref_bin_to)

    def __CDF(self,x,ref_bin_from, ref_bin_to):
        if (x-self._delay) <= ref_bin_from:
            return 0.
        elif (x-self._delay) < ref_bin_to:
            return ((x-self._delay)-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.

class LinearTransformFromFile(LinearTransform):
    """ Interpolates matrix columns by using inpulse response data from a file. """

    def __init__(self,filename, x_axis = 'time', **kwargs):
        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0]=self._data[:, 0]

        super(self.__class__, self).__init__( **kwargs)
        self.label = 'LT from file'

    def response_function(self, parameters, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
            return np.interp(bin_mid - ref_bin_mid, self._data[:, 0], self._data[:, 1])


class LinearTransformFilter(LinearTransform, metaclass=ABCMeta):
    """ A general class for (analog) filters. Impulse response of the filter must be determined by overwriting
        the function raw_impulse_response.

        This processor includes two additional properties.

    """

    def __init__(self, scaling, zero_bin_value=None, normalization=None, **kwargs):

        self._scaling = scaling

        if normalization == 'sum':
            self._filter_normalization = None
            matrix_normalization = normalization
        else:
            self._filter_normalization = normalization
            matrix_normalization = None

        self._zero_bin_value = zero_bin_value
        super(LinearTransformFilter, self).__init__(normalization = matrix_normalization, **kwargs)
        self.label='LinearTransformFilter'

        self._norm_coeff = None

    def response_function(self, parameters, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Frequency scaling must be done by scaling integral limits, because integration by substitution doesn't work
        # with np.quad (see quad_problem.ipynbl). An ugly way, which could be fixed.

        temp, _ = integrate.quad(self._impulse_response, self._scaling * (bin_from - (ref_bin_mid)),
                                 self._scaling * (bin_to - (ref_bin_mid)))

#        temp, _ = integrate.quad(self._impulse_response, self._scaling * (bin_from - (ref_bin_mid + self._delay_z)),
#                                 self._scaling * (bin_to - (ref_bin_mid + self._delay_z)))

        if ref_bin_mid == bin_mid:
            if self._zero_bin_value is not None:
                temp += self._zero_bin_value

        if self._norm_coeff is None:
            self._norm_coeff = self._normalization_coefficient(parameters)

        return temp/self._norm_coeff

    def _normalization_coefficient(self, parameters):

        if self._filter_normalization is None:
            norm_coeff = 1.
        elif isinstance(self._filter_normalization, tuple):
            if self._filter_normalization[0] == 'integral':
                norm_coeff, _ = integrate.quad(self._impulse_response, self._filter_normalization[1][0], self._filter_normalization[1][1])
            elif self._filter_normalization[0] == 'bunch_by_bunch':
                f_h = self._filter_normalization[1]

                norm_coeff = 0.
                for i in range(-1000,1000):
                    x = float(i)* (1./f_h) * self._scaling
                    norm_coeff += self._impulse_response(x)

                bin_edges = parameters['bin_edges']
                n_bins_per_segment = parameters['n_bins_per_segment']
                segment_length = bin_edges[n_bins_per_segment-1,1] - bin_edges[0,0]

                norm_coeff = norm_coeff*(segment_length * self._scaling)
            else:
                raise ValueError('Unknown normalization method!')
#        elif self._normalization == 'sum':
#            norm_coeff = np.sum(impulse)

        else:
            raise ValueError('Unknown normalization method!')

        return norm_coeff

class Lowpass(LinearTransformFilter):
    """ A classical lowpass filter, which is also known as a RC-filter or one
        poll roll off.
    """
    def __init__(self,f_cutoff, normalization=None, max_impulse_length = 5., **kwargs):
        scaling = 2. * pi * f_cutoff

        if normalization is None:
            normalization=('integral',(-max_impulse_length,max_impulse_length))

        self._impulse_response = abstract_filter_responses.normalized_lowpass(max_impulse_length)

        super(self.__class__, self).__init__(scaling, normalization=normalization,**kwargs)
        self.label = 'Lowpass filter'


class Highpass(LinearTransformFilter):
    """ A high pass version of the lowpass filter, which is constructed by
        multiplying the lowpass filter by a factor of -1 and adding to the first
        bin 1
    """
    def __init__(self,f_cutoff, normalization=None, max_impulse_length = 5., **kwargs):
        scaling = 2. * pi * f_cutoff

        if normalization is None:
            normalization=('integral',(-max_impulse_length,max_impulse_length))

        self._impulse_response = abstract_filter_responses.normalized_highpass(max_impulse_length)

        super(self.__class__, self).__init__( scaling, zero_bin_value= 1., normalization=normalization, **kwargs)
        self.label = 'Highpass filter'

class PhaseLinearizedLowpass(LinearTransformFilter):
    """ A phase linearized 1st order lowpass filter. Note that the narrow and
        sharp peak of the impulse response makes the filter to be sensitive
        to the bin width and may yield an unrealistically good response for the
        short signals. Thus, it is recommended to use a higher bandwidth Gaussian
        filter together with this filter.
    """

    def __init__(self,f_cutoff, normalization=None, max_impulse_length = 5., **kwargs):
        scaling = 2. * pi * f_cutoff

        if normalization is None:
            normalization=('integral',(-max_impulse_length,max_impulse_length))

        self._impulse_response = abstract_filter_responses.normalized_phase_linearized_lowpass(max_impulse_length)

        super(self.__class__, self).__init__( scaling, normalization=normalization, **kwargs)
        self.label = 'Phaselinearized lowpass filter'


class Gaussian(LinearTransformFilter):
    """ A Gaussian low pass filter, which impulse response is a Gaussian function.
    """
    def __init__(self,f_cutoff, normalization=None, max_impulse_length = 5., **kwargs):
        scaling = 2. * pi * f_cutoff

        if normalization is None:
            normalization=('integral',(-max_impulse_length,max_impulse_length))

        self._impulse_response = abstract_filter_responses.normalized_Gaussian(max_impulse_length)

        super(self.__class__, self).__init__( scaling, normalization=normalization, **kwargs)
        self.label = 'Gaussian lowpass filter'


class Sinc(LinearTransformFilter):
    """ A nearly ideal lowpass filter, i.e. a windowed Sinc filter. The impulse response of the ideal lowpass filter
        is Sinc function, but because it is infinite length in both positive and negative time directions, it can not be
        used directly. Thus, the length of the impulse response is limited by using windowing. Properties of the filter
        depend on the width of the window and the type of the windows and must be written down. Too long window causes
        ripple to the signal in the time domain and too short window decreases the slope of the filter in the frequency
        domain. The default values are a good compromise. More details about windowing can be found from
        http://www.dspguide.com/ch16.htm and different options for the window can be visualized, for example, by using
        code in example/test 004_analog_signal_processors.ipynb
    """

    def __init__(self, f_cutoff, window_width = 3., window_type = 'blackman', normalization=None,
                 **kwargs):
        """
        :param f_cutoff: a cutoff frequency of the filter
        :param delay: a delay of the filter [s]
        :param window_width: a (half) width of the window in the units of zeros of Sinc(x) [2*pi*f_c]
        :param window_type: a shape of the window, blackman or hamming
        :param norm_type: see class LinearTransform
        :param norm_range: see class LinearTransform
        """
        scaling = 2. * pi * f_cutoff

        if normalization is None:
            normalization=('integral',(-window_width,window_width))

        self._impulse_response = abstract_filter_responses.normalized_sinc(window_type, window_width)

        super(self.__class__, self).__init__(scaling,normalization=normalization, **kwargs)
        self.label = 'Sinc filter'
