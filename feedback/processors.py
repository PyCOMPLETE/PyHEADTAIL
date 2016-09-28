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


# An alternative for np.dot because it slows down the calculations in LSF by a factor of two or more
gemm = linalg.get_blas_funcs("gemm")

def dot(A,B):
    return np.squeeze(gemm(1, A, B))

"""
    This file contains signal processors which can be used in the feedback module in PyHEADTAIL.

    A general requirement for the signal processor is that it is a class object containing a function, namely,
    process (signal, slice_set, phase_advance). The input parameters for the function process(signal, slice_set) are
    a numpy array 'signal',  a slice_set object of PyHEADTAIL and a phase advance of the signal in the units of absolute
    angle of betatron motion from the reference point of the accelerator. The function must return a numpy array with
    equal length to the input array. The other requirement is that the class object contains a list variable, namely
    'required_variables', which includes required variables for slicet_objects.

    The signals processors in this file are based on four abstract classes;
        1) in LinearTransform objects the input signal is multiplied with a matrix.
        2) in Multiplication objects the input signal is multiplied with an array with equal length to the input array
        3) in Addition objects to the input signal is added an array with equal length to the input array
        4) A normal signal processor doesn't store a signal (in terms of process() calls). Processors buffering,
           registering and/or delaying signals are namely Registers. The Registers have following properties in addition
           to the normal processor:
            a) the object is iterable
            b) the object contains a function namely combine(*args), which combines two signals returned by iteration
               together

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""

class LinearTransform(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on linear transformation. The signal is processed by
        calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced with an abstract
        method, namely response_function(*args), which returns an elements of the matrix (an effect of
        the ref_bin to the bin)
    """

    def __init__(self, norm_type=None, norm_range=None, matrix_symmetry = 'none', bin_check = False,
                 bin_middle = 'bin', recalculate_always = False):
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
        :param matrix_symmetry: symmetry of the matrix is used for minimizing the number of calculable elements
            in the matrix. Implemented options are:
            'none':             all elements are calculated separately
            'fully_diagonal':   all elements are identical in diagonal direction
        :param bin_check: if True, a change of the bin_set is checked every time process() is called and matrix is
            recalculated if any change is found
        :param bin_middle: defines if middle points of the bins are determined by a middle point of the bin
            (bin_middle = 'bin') or an average place of macro particles (bin_middle = 'particles')
        """

        self._norm_type = norm_type
        self._norm_range = norm_range
        self._bin_check = bin_check
        self._bin_middle = bin_middle
        self._matrix_symmetry = matrix_symmetry

        self._z_bin_set = None
        self._matrix = None

        self._recalculate_matrix = True
        self._recalculate_matrix_always = recalculate_always

        self.required_variables = ['z_bins','mean_z']


    @abstractmethod
    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Impulse response function of the processor
        pass

    def process(self,signal,slice_set, *args):

        # check if the bin set is changed
        if self._bin_check:
            self._recalculate_matrix = self.__check_bin_set(slice_set.z_bins)

        # recalculte the matrix if necessary
        if self._recalculate_matrix:
            if self._recalculate_matrix_always == False:
                self._recalculate_matrix = False
            if self._bin_middle == 'particles':
                bin_midpoints = np.array(copy.copy(slice_set.mean_z))
            elif self._bin_middle == 'bin':
                bin_midpoints = np.array([(i+j)/2. for i, j in zip(slice_set.z_bins, slice_set.z_bins[1:])])

            self.__generate_matrix(slice_set.z_bins,bin_midpoints)

        # process the signal
        return dot(self._matrix, signal)

        # np.dot can't be used, because it slows down the calculations in LSF by a factor of two or more
        # return np.dot(self._matrix,signal)

    def clear_matrix(self):
        self._matrix = np.array([])
        self._recalculate_matrix = True

    def __check_bin_set(self,z_bin_set):

        if self._z_bin_set is None:
            self._z_bin_set = copy.copy(z_bin_set)
            return True

        else:
            changed = False
            for old, new in itertools.izip(self._z_bin_set,z_bin_set):
                if old != new:
                    changed = True
                    self._z_bin_set = copy.copy(z_bin_set)
                    break
            return changed

    def print_matrix(self):
        for row in self._matrix:
            print "[",
            for element in row:
                print "{:6.3f}".format(element),
            print "]"

    def __generate_matrix(self,bin_set, bin_midpoints):

        self._matrix = np.identity(len(bin_midpoints))
        self._matrix *= np.nan

        bin_widths = []
        for i,j in zip(bin_set,bin_set[1:]):
            bin_widths.append(j-i)

        bin_widths = np.array(bin_widths)
        bin_std = np.std(bin_widths)/np.mean(bin_widths)

        if bin_std > 1e-3:
            'Dynamic slicer -> unoptimized matrix generation!'

        if self._matrix_symmetry == 'fully_diagonal' and bin_std < 1e-3:
            j = 0
            midpoint_j = bin_midpoints[0]
            for i, midpoint_i in enumerate(bin_midpoints):
                self._matrix[j][i] = self.response_function(midpoint_i, bin_set[i], bin_set[i + 1], midpoint_j,
                                                           bin_set[j]
                                                           , bin_set[j + 1])
                for val in xrange(1,len(bin_midpoints) - max(i,j)):
                    self._matrix[j + val][i + val] = self._matrix[j][i]

            i = 0
            midpoint_i = bin_midpoints[0]
            for j, midpoint_j in enumerate(bin_midpoints[1:], start=1):
                self._matrix[j][i] = self.response_function(midpoint_i, bin_set[i], bin_set[i + 1], midpoint_j,
                                                           bin_set[j]
                                                           , bin_set[j + 1])
                for val in xrange(1,len(bin_midpoints) - max(i,j)):
                    self._matrix[j + val][i + val] = self._matrix[j][i]

        else:
            counter = 0
            for i, midpoint_i in enumerate(bin_midpoints):
                for j, midpoint_j in enumerate(bin_midpoints):
                        if np.isnan(self._matrix[j][i]):
                            counter += 1
                            self._matrix[j][i] = self.response_function(midpoint_i,bin_set[i],bin_set[i+1],midpoint_j,bin_set[j]
                                                                   ,bin_set[j+1])
            # print str(counter) + ' elements is calculated'


        if self._norm_type == 'bunch_average':
            self._norm_coeff = bin_set[-1] - bin_set[0]
        elif self._norm_type == 'fixed_average':
            self._norm_coeff = self._norm_range[1] - self._norm_range[0]
        elif self._norm_type == 'bunch_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self._norm_coeff = self.response_function(bin_midpoints[center_idx], bin_set[center_idx],
                                                     bin_set[center_idx + 1], bin_midpoints[center_idx],
                                                     bin_set[0], bin_set[-1])
        elif self._norm_type == 'fixed_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self._norm_coeff = self.response_function(bin_midpoints[center_idx], bin_set[center_idx],
                                                     bin_set[center_idx + 1], bin_midpoints[center_idx],
                                                     self._norm_range[0], self._norm_range[-1])
        elif self._norm_type == 'max_column':
            self._norm_coeff= np.max(np.sum(self._matrix,0))

        elif self._norm_type is None:
            self._norm_coeff = 1.

        self._matrix = self._matrix / float(self._norm_coeff)


class Averager(LinearTransform):
    """ Returns a signal, which consists an average value of the input signal. A sums of the rows in the matrix
        are normalized to be one (i.e. a sum of the input signal doesn't change).
    """

    def __init__(self,norm_type = 'max_column', norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        return 1


class Delay(LinearTransform):
    """ Delays signal in the units of [second].
    """
    def __init__(self,delay, norm_type = None, norm_range = None,recalculate_always = False):
        self._delay = delay
        super(self.__class__, self).__init__(norm_type, norm_range, 'fully_diagonal', recalculate_always = recalculate_always)

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

    def __init__(self,filename, x_axis = 'time', norm_type = 'max_column', norm_range = None):
        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0]=self._data[:, 0]*c

        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
            return np.interp(bin_mid - ref_bin_mid, self._data[:, 0], self._data[:, 1])


class Filter(LinearTransform):
    __metaclass__ = ABCMeta
    """ A general class for (analog) filters. Impulse response of the filter must be determined by overwriting
        the function raw_impulse_response.

        This processor includes two additional properties.

    """

    def __init__(self, filter_type, f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range):
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

        self._f_cutoff = f_cutoff
        self._delay_z = delay * c
        self._filter_type = filter_type

        self._impulse_response = self.__impulse_response_generator(f_cutoff_2nd)
        super(Filter, self).__init__(norm_type, norm_range, 'fully_diagonal')


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
                    return self.raw_impulse_response(np.sign(x)*threshold_tau)
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

        scaling = 2.*pi*self._f_cutoff/c
        temp, _ = integrate.quad(self._impulse_response, scaling * (bin_from - (ref_bin_mid+self._delay_z)),
                       scaling * (bin_to - (ref_bin_mid+self._delay_z)))

        if ref_bin_mid == bin_mid:
            if self._filter_type == 'highpass':
                temp += 1.

        return temp


class Sinc(Filter):
    """ A nearly ideal lowpass filter, i.e. a windowed Sinc filter. The impulse response of the ideal lowpass filter
        is Sinc function, but because it is infinite length in both positive and negative time directions, it can not be
        used directly. Thus, the length of the impulse response is limited by using windowing. Properties of the filter
        depend on the width of the window and the type of the windows and must be written down. Too long window causes
        ripple to the signal in the time domain and too short window decreases the slope of the filter in the frequency
        domain. The default values are a good compromise. More details about windowing can be found from
        http://www.dspguide.com/ch16.htm and different options for the window can be visualized, for example, by using
        code in example/test 004_analog_signal_processors.ipynb
    """

    def __init__(self, f_cutoff, delay=0., window_width = 3, window_type = 'blackman' , norm_type=None, norm_range=None):
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
        super(self.__class__, self).__init__('lowpass', f_cutoff, delay, None, norm_type, norm_range)

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


class Lowpass(Filter):
    """ Classical first order lowpass filter (e.g. a RC filter), which impulse response can be described as exponential
        decay.
        """
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None):
        super(self.__class__, self).__init__('lowpass', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range)

    def raw_impulse_response(self, x):
        if x < 0:
            return 0
        else:
            return math.exp(-1. * x)

class Highpass(Filter):
    """The classical version of a highpass filter, which """
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None):
        super(self.__class__, self).__init__('highpass', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range)

    def raw_impulse_response(self, x):
        if x < 0:
            return 0
        else:
            return -1.*math.exp(-1. * x)

class PhaseLinearizedLowpass(Filter):
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None):
        super(self.__class__, self).__init__('lowpass', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range)

    def raw_impulse_response(self, x):
        if x == 0:
            return 0
        else:
            return special.k0(abs(x))

class Multiplication(object):
    __metaclass__ = ABCMeta
    """ An abstract class which multiplies the input signal by an array. The multiplier array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        multiplication_function(seed).
    """
    def __init__(self, seed, normalization = None, recalculate_multiplier = False):
        """
        :param seed: 'bin_length', 'bin_midpoint', 'signal' or a property of a slice, which can be found
            from slice_set
        :param normalization:
            'total_weight':  a sum of the multiplier array is equal to 1.
            'average_weight': an average in  the multiplier array is equal to 1,
            'maximum_weight': a maximum value in the multiplier array value is equal to 1
            'minimum_weight': a minimum value in the multiplier array value is equal to 1
        :param: recalculate_weight: if True, the weight is recalculated every time when process() is called
        """

        self._seed = seed
        self._normalization = normalization
        self._recalculate_multiplier = recalculate_multiplier

        self._multiplier = None

        self.required_variables = ['z_bins']

        if self._seed not in ['bin_length','bin_midpoint','signal']:
            self.required_variables.append(self._seed)

    @abstractmethod
    def multiplication_function(self, seed):
        pass

    def process(self,signal,slice_set, *args):

        if (self._multiplier is None) or self._recalculate_multiplier:
            self.__calculate_multiplier(signal,slice_set)

        return self._multiplier*signal

    def __calculate_multiplier(self,signal,slice_set):
        if self._seed == 'bin_length':
            bin_set = slice_set.z_bins
            self._multiplier = np.array([(j-i) for i, j in zip(bin_set, bin_set[1:])])
        elif self._seed == 'bin_midpoint':
            bin_set = slice_set.z_bins
            self._multiplier = np.array([(i+j)/2. for i, j in zip(bin_set, bin_set[1:])])
        elif self._seed == 'signal':
            self._multiplier = np.array(signal)
        else:
            self._multiplier = np.array(getattr(slice_set,self._seed))

        self._multiplier = self.multiplication_function(self._multiplier)

        if self._normalization == 'total_weight':
            norm_coeff = float(np.sum(self._multiplier))
        elif self._normalization == 'average_weight':
            norm_coeff = float(np.sum(self._multiplier))/float(len(self._multiplier))
        elif self._normalization == 'maximum_weight':
            norm_coeff = float(np.max(self._multiplier))
        elif self._normalization == 'minimum_weight':
            norm_coeff = float(np.min(self._multiplier))
        elif self._normalization == None:
            norm_coeff = 1.

        # TODO: try to figure out why this can not be written
        # self._multiplier /= norm_coeff
        self._multiplier =  self._multiplier / norm_coeff


class ChargeWeighter(Multiplication):
    """ weights signal with charge (macroparticles) of slices
    """

    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('n_macroparticles_per_slice', normalization)

    def multiplication_function(self,weight):
        return weight


class EdgeWeighter(Multiplication):
    """ Use an inverse of the Fermi-Dirac distribution function to increase signal strength on the edges of the bunch
    """

    def __init__(self,bunch_length,bunch_decay_length,maximum_weight = 10):
        """
        :param bunch_length: estimated width of the bunch
        :param bunch_decay_length: slope of the function on the edge of the bunch. Smaller value, steeper slope.
        :param maximum_weight: maximum value of the weight
        """
        self._bunch_length = bunch_length
        self._bunch_decay_length = bunch_decay_length
        self._maximum_weight=maximum_weight
        super(self.__class__, self).__init__('bin_midpoint', 'minimum_weight')

    def multiplication_function(self,weight):
        weight = np.exp((np.absolute(weight)-self._bunch_length/2.)/float(self._bunch_decay_length))+ 1.
        weight = np.clip(weight,1.,self._maximum_weight)
        return weight


class NoiseGate(Multiplication):
    """ Passes a signal which is greater/less than the threshold level.
    """

    def __init__(self,threshold, operator = 'greater', threshold_ref = 'amplitude'):

        self._threshold = threshold
        self._operator = operator
        self._threshold_ref = threshold_ref
        super(self.__class__, self).__init__('signal', None,recalculate_multiplier = True)

    def multiplication_function(self, seed):
        multiplier = np.zeros(len(seed))

        if self._threshold_ref == 'amplitude':
            comparable = np.abs(seed)
        elif self._threshold_ref == 'absolute':
            comparable = seed

        if self._operator == 'greater':
            multiplier[comparable > self._threshold] = 1
        elif self._operator == 'less':
            multiplier[comparable < self._threshold] = 1

        return multiplier


class MultiplicationFromFile(Multiplication):
    """ Multiplies the signal with an array, which is produced by interpolation from the loaded data. Note the seed for
        the interpolation can be any of those presented in the abstract function. E.g. a spatial weight can be
        determined by using a bin midpoint as a seed, nonlinear amplification can be modelled by using signal itself
        as a seed and etc...
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint',normalization = None, recalculate_multiplier = False):
        super(self.__class__, self).__init__(seed, normalization, recalculate_multiplier)
        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0] = self._data[:, 0] * c

    def multiplication_function(self, seed):
        return np.interp(seed, self._data[:, 0], self._data[:, 1])


class Addition(object):
    __metaclass__ = ABCMeta
    """ An abstract class which adds an array to the input signal. The addend array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        addend_function(seed).
    """

    def __init__(self, seed, normalization = None, recalculate_addend = False):
        """
        :param seed: 'bin_length', 'bin_midpoint', 'signal' or a property of a slice, which can be found
            from slice_set
        :param normalization:
            'total_weight':  a sum of the multiplier array is equal to 1.
            'average_weight': an average in  the multiplier array is equal to 1,
            'maximum_weight': a maximum value in the multiplier array value is equal to 1
            'minimum_weight': a minimum value in the multiplier array value is equal to 1
        :param: recalculate_weight: if True, the weight is recalculated every time when process() is called
        """

        self._seed = seed
        self._normalization = normalization
        self._recalculate_addend = recalculate_addend

        self._addend = None

        self.required_variables=['z_bins']

        if self._seed not in ['bin_length','bin_midpoint','signal']:
            self.required_variables.append(self._seed)

    @abstractmethod
    def addend_function(self, seed):
        pass

    def process(self,signal,slice_set, *args):

        if (self._addend is None) or self._recalculate_addend:
            self.__calculate_addend(signal,slice_set)

        return signal + self._addend

    def __calculate_addend(self,signal,slice_set):
        if self._seed == 'bin_length':
            bin_set = slice_set.z_bins
            self._addend = np.array([(j-i) for i, j in zip(bin_set, bin_set[1:])])
        elif self._seed == 'bin_midpoint':
            bin_set = slice_set.z_bins
            self._addend = np.array([(i+j)/2. for i, j in zip(bin_set, bin_set[1:])])
        elif self._seed == 'signal':
            self._addend = np.array(signal)
        else:
            self._addend = np.array(getattr(slice_set,self._seed))

        self._addend = self.addend_function(self._addend)

        if self._normalization == 'total':
            norm_coeff = float(np.sum(self._addend))
        elif self._normalization == 'average':
            norm_coeff = float(np.sum(self._addend))/float(len(self._addend))
        elif self._normalization == 'maximum':
            norm_coeff = float(np.max(self._addend))
        elif self._normalization == 'minimum':
            norm_coeff = float(np.min(self._addend))
        else:
            norm_coeff = 1.

        self._addend = self._addend / norm_coeff


class NoiseGenerator(Addition):
    """ Adds noise to a signal. The noise level is given as RMS value of the absolute level (reference_level = 'absolute'),
        a relative RMS level to the maximum signal (reference_level = 'maximum') or a relative RMS level to local
        signal values (reference_level = 'local'). Options for the noise distribution are a Gaussian normal distribution
        (distribution = 'normal') and an uniform distribution (distribution = 'uniform')
    """

    def __init__(self,RMS_noise_level,reference_level = 'absolute', distribution = 'normal'):

        self._RMS_noise_level = RMS_noise_level
        self._reference_level = reference_level
        self._distribution = distribution

        super(self.__class__, self).__init__('signal', None, True)

    def addend_function(self,seed):

        randoms = np.zeros(len(seed))

        if self._distribution == 'normal' or self._distribution is None:
            randoms = np.random.randn(len(seed))
        elif self._distribution == 'uniform':
            randoms = 1./0.577263*(-1.+2.*np.random.rand(len(seed)))

        if self._reference_level == 'absolute':
            addend = self._RMS_noise_level*randoms
        elif self._reference_level == 'maximum':
            addend = self._RMS_noise_level*np.max(seed)*randoms
        elif self._reference_level == 'local':
            addend = seed*self._RMS_noise_level*randoms

        return addend

class AdditionFromFile(Addition):
    """ Adds an array to the signal, which is produced by interpolation from the loaded data. Note the seed for
        the interpolation can be any of those presented in the abstract function.
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint',normalization = None, recalculate_multiplier = False):
        super(self.__class__, self).__init__(seed, normalization, recalculate_multiplier)
        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0] = self._data[:, 0] * c

    def addend_function(self, seed):
        return np.interp(seed, self._data[:, 0], self._data[:, 1])

class Register(object):
    __metaclass__ = ABCMeta

    """ An abstract class for a signal register. A signal is stored to the register, when the function process() is
        called. The register is iterable and returns values which have been kept in register longer than
        delay requires. Normally this means that a number of returned signals corresponds to a paremeter avg_length, but
        it is less during the first turns. The values from the register can be calculated together by using a abstract
        function combine(*). It manipulates values (in terms of a phase advance) such way they can be calculated
        together in the reader position.

        When the register is a part of a signal processor chain, the function process() returns np.array() which
        is an average of register values determined by a paremeter avg_length. The exact functionality of the register
        is determined by in the abstract iterator combine(*args).

    """

    def __init__(self, n_avg, tune, delay, in_processor_chain):
        """
        :param n_avg: a number of register values (in turns) have been stored after the delay
        :param tune: a real number value of a betatron tune (e.g. 59.28 in horizontal or 64.31 in vertical direction
                for LHC)
        :param delay: a delay between storing to reading values  in turns
        :param in_processor_chain: if True, process() returns a signal
        """
        self._delay = delay
        self._n_avg = n_avg
        self._phase_shift_per_turn = 2.*pi * tune
        self._phase_advance = None
        self._in_processor_chain = in_processor_chain
        self.combination = None


        self._max_reg_length = self._delay+self._n_avg
        self._register = deque()

        self._n_iter_left = -1

        self._reader_position = None

        # if n_slices is not None:
        #     self._register.append(np.zeros(n_slices))

        self.required_variables = None

    def __iter__(self):
        # calculates a maximum number of iterations. If there is no enough values in the register, sets -1, which
        # indicates that next() can return zero value

        self._n_iter_left =  len(self)
        if self._n_iter_left == 0:
            # return None
            self._n_iter_left = -1
        return self

    def __len__(self):
        # returns a number of signals in the register after delay
        return max((len(self._register) - self._delay), 0)

    def next(self):
        if self._n_iter_left < 1:
            raise StopIteration
        else:
            delay = -1. * (len(self._register) - self._n_iter_left) * self._phase_shift_per_turn
            self._n_iter_left -= 1
            return (self._register[self._n_iter_left],None,delay,self._phase_advance)

    def process(self,signal, slice_set, data_phase_advance ,*args):

        if self._phase_advance is None:
            self._phase_advance = data_phase_advance

        self._register.append(signal)

        if len(self._register) > self._max_reg_length:
            self._register.popleft()

        if self._in_processor_chain == True:
            temp_signal = np.zeros(len(signal))
            if len(self) > 0:
                prev = (np.zeros(len(self._register[0])),None,0,self._phase_advance)

                for value in self:
                    combined = self.combine(value,prev,None)
                    prev = value
                    temp_signal += combined / float(len(self))

            return temp_signal

    @abstractmethod
    def combine(self,x1,x2,reader_position,x_to_xp = False):

        pass


class VectorSumRegister(Register):

    def __init__(self, n_avg, tune, delay = 0, in_processor_chain=True):
        self.combination = 'combined'
        super(self.__class__, self).__init__(n_avg, tune, delay, in_processor_chain)
        self.required_variables = []

    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
        # determines a complex number representation from two signals (e.g. from two pickups or different turns), by using
        # knowledge about phase advance between signals. After this turns the vector to the reader's phase
        # TODO: Why not x2[3]-x1[3]?

        if (x1[3] is not None) and (x1[3] != x2[3]):
            phi_x1_x2 = x1[3]-x2[3]
            if phi_x1_x2 < 0:
                # print "correction"
                phi_x1_x2 += self._phase_shift_per_turn
        else:
            phi_x1_x2 = -1. * self._phase_shift_per_turn

        print "Delta phi: " + str(phi_x1_x2*360./(2*pi)%360.)

        s = np.sin(phi_x1_x2/2.)
        c = np.cos(phi_x1_x2/2.)

        re = 0.5 * (x1[0] + x2[0]) * (c + s * s / c)
        im = -s * x2[0] + c / s * (re - c * x2[0])

        delta_phi = x1[2]-phi_x1_x2/2.

        if reader_phase_advance is not None:
            delta_position = x1[3] - reader_phase_advance
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self._phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        s = np.sin(delta_phi)
        c = np.cos(delta_phi)


        return c*re-s*im

        # An old piece. It should work as well as the code above, but it has different forbidden values for phi_x1_x2
        # (where re or im variables go to infinity). Thus it is stored to here, so that it can be found easily but it
        # will be probably removed later.
        # if (x1[3] is not None) and (x1[3] != x2[3]):
        #     phi_x1_x2 = x1[3]-x2[3]
        #     if phi_x1_x2 < 0:
        #         # print "correction"
        #         phi_x1_x2 += self._phase_shift_per_turn
        # else:
        #     phi_x1_x2 = -1. * self._phase_shift_per_turn
        #
        # s = np.sin(phi_x1_x2)
        # c = np.cos(phi_x1_x2)
        # re = x1[0]
        # im = (c*x1[0]-x2[0])/float(s)
        #
        # # turns the vector to the reader's position
        # delta_phi = x1[2]
        # if reader_phase_advance is not None:
        #     delta_position = x1[3] - reader_phase_advance
        #     delta_phi += delta_position
        #     if delta_position > 0:
        #         delta_phi -= self._phase_shift_per_turn
        #     if x_to_xp == True:
        #         delta_phi -= pi/2.
        #
        # s = np.sin(delta_phi)
        # c = np.cos(delta_phi)
        #
        # # return np.array([c*re-s*im,s*re+c*im])
        #
        # return c*re-s*im


class CosineSumRegister(Register):
    """ Returns register values by multiplying the values with a cosine of the betatron phase angle from the reader.
        If there are multiple values in different phases, the sum approaches a value equal to half of the displacement
        in the reader's position.
    """
    def __init__(self, n_avg, tune, delay = 0, in_processor_chain=True):

        self.combination = 'individual'

        super(self.__class__, self).__init__(n_avg, tune, delay, in_processor_chain)
        self.required_variables = []

    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
        delta_phi = x1[2]
        if reader_phase_advance is not None:
            delta_position = self._phase_advance - reader_phase_advance
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self._phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        return 2.*math.cos(delta_phi)*x1[0]


class Bypass(object):
    """ A fast bypass processor, whichi does not modify the signal. A black sheep, which does not fit for
        the abstract classes.
    """

    def __init__(self):
        self.required_variables = []

    def process(self,signal, *args):
        return signal