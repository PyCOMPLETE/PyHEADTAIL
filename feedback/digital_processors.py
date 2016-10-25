from abc import ABCMeta, abstractmethod
from scipy import signal
import copy, math
import numpy as np
from scipy.constants import c, pi
from itertools import izip, count
from processors import Register
from scipy import linalg
import pyximport; pyximport.install()
from cython_functions import cython_matrix_product

"""
    This file contains signal processors which can be used for emulating digital signal processing in the feedback
    module. All the processors can be used separately, but digital filters assumes uniform slice spacing (bin width).
    If UniformCharge mode is used in the slicer, uniform bin width can be formed with ADC and DAC processors.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""

class Resampler(object):
    def __init__(self,type, sampling_rate, sync_method):

        """ Changes a sampling rate of the signal. Assumes that either the sampling of the incoming (ADC) or
            the outgoing (DAC) signal corresponds to the sampling found from the slice_set.

        :param type: type of the conversion, i.e. 'ADC' or 'DAC'
        :param sampling_rate: Samples per second
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
        self._type = type
        self._sampling_rate = sampling_rate
        self._sync_method = sync_method

        self._matrix = None

        self.required_variables = ['z_bins']


    def __generate_matrix(self, z_bins_input, z_bins_output):
        self._matrix = np.zeros((len(z_bins_output)-1, len(z_bins_input)-1))
        for i, bin_in_min, bin_in_max in izip(count(), z_bins_input,z_bins_input[1:]):
            for j, bin_out_min, bin_out_max in izip(count(), z_bins_output, z_bins_output[1:]):
                out_bin_length = bin_out_max - bin_out_min
                in_bin_length = bin_in_max - bin_in_min

                self._matrix[j][i] = (self.__CDF(bin_out_max, bin_in_min, bin_in_max) -
                                      self.__CDF(bin_out_min, bin_in_min, bin_in_max)) * in_bin_length / out_bin_length

    def __generate_new_binset(self,orginal_bins, n_signal_bins):

        new_bins = None
        signal_length = (orginal_bins[-1] - orginal_bins[0]) / c

        if self._sync_method == 'round':
            if self._sampling_rate is None:
                n_z_bins = n_signal_bins
            else:
                n_z_bins = int(math.ceil(signal_length * self._sampling_rate))
            new_bins = np.linspace(orginal_bins[0], orginal_bins[-1], n_z_bins + 1)

        elif self._sync_method == 'rising_edge':
            n_z_bins = int(math.floor(signal_length * self._sampling_rate))
            max_output_z = orginal_bins[0] + float(n_z_bins) * self._sampling_rate * c
            new_bins = np.linspace(orginal_bins[0], max_output_z, n_z_bins + 1)
        elif self._sync_method == 'falling_edge':
            n_z_bins = int(math.floor(signal_length * self._sampling_rate))
            min_output_z = orginal_bins[-1] - float(n_z_bins) * self._sampling_rate * c
            new_bins = np.linspace(min_output_z, orginal_bins[-1], n_z_bins + 1)
        elif self._sync_method == 'middle':
            n_z_bins = int(math.floor(signal_length * self._sampling_rate))
            delta_z = (orginal_bins[-1] - orginal_bins[0]) - float(n_z_bins) * self._sampling_rate * c
            new_bins = np.linspace(orginal_bins[0] + delta_z / 2., orginal_bins[-1] - delta_z / 2., n_z_bins + 1)

        return new_bins

    @staticmethod
    def __CDF(x,ref_bin_from, ref_bin_to):
        if x <= ref_bin_from:
            return 0.
        elif x < ref_bin_to:
            return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.

    def clear_matrix(self):
        self._matrix = None

    def process(self,signal,slice_set, *args):

        z_bins_input = None
        z_bins_output = None

        if self._matrix is None:
            if self._type == 'ADC':
                z_bins_input = slice_set.z_bins
                z_bins_output = self.__generate_new_binset(slice_set.z_bins, len(signal))
            elif self._type == 'DAC':
                z_bins_input = self.__generate_new_binset(slice_set.z_bins, len(signal))
                z_bins_output = slice_set.z_bins

            self.__generate_matrix(z_bins_input, z_bins_output)

        signal = np.array(signal)
        return np.array(cython_matrix_product(self._matrix, signal))
        # np.dot can't be used, because it slows down the calculations in LSF by a factor of two or three
        # return np.dot(self._matrix, signal)


class Quantizer(object):
    def __init__(self,n_bits,input_range):

        """ Quantizates signal to discrete levels determined by the number of bits and input range.
        :param n_bits: the signal is quantized (rounded) to 2^n_bits levels
        :param input_range: the maximum and minimum values for the levels in the units of input signal
        """

        self._n_bits = n_bits
        self._n_steps = np.power(2,self._n_bits)-1.
        self._input_range = input_range
        self._step_size = (self._input_range[1]-self._input_range[0])/float(self._n_steps)
        self.required_variables = []

    def process(self, signal, *args):
        signal = self._step_size*np.floor(signal/self._step_size+0.5)

        signal[signal < self._input_range[0]] = self._input_range[0]
        signal[signal > self._input_range[1]] = self._input_range[1]

        return signal


class ADC(object):
    def __init__(self,sampling_rate, n_bits = None, input_range = None, sync_method = 'round'):
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
        self._resampler = Resampler('ADC', sampling_rate, sync_method)
        self.required_variables = copy.copy(self._resampler.required_variables)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range)
            self.required_variables += self._digitizer.required_variables

    def process(self,signal,slice_set, *args):
        signal = self._resampler.process(signal,slice_set)

        if self._digitizer is not None:
            signal = self._digitizer.process(signal)

        return signal


class DAC(object):
    def __init__(self,sampling_rate, n_bits = None, output_range = None, sync_method = 'round'):
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
        self._resampler = Resampler('DAC', sampling_rate, sync_method)
        self.required_variables = copy.copy(self._resampler.required_variables)

        self._digitizer = None
        if (n_bits is not None) and (output_range is not None):
            self._digitizer = Quantizer(n_bits,output_range)
            self.required_variables += self._digitizer.required_variables

    def process(self,signal,slice_set, *args):

        if self._digitizer is not None:
            signal = self._digitizer.process(signal)

        signal = self._resampler.process(signal,slice_set)

        return signal


class DigitalFilter(object):
    def __init__(self,coefficients):
        """ Filters the signal by convolving the signal and the input array of filter (FIR) coefficients
        :param coefficients: A numpy array of filter (convolution) coefficients
        """
        self._coefficients = coefficients
        self.required_variables = []

    def process(self, signal, *args):
        return np.convolve(np.array(signal), np.array(self._coefficients), mode='same')


class FIR_Filter(DigitalFilter):
    def __init__(self,n_taps, f_cutoffs, sampling_rate):

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

        super(self.__class__, self).__init__(coefficients)


class FIR_Register(Register):
    def __init__(self, n_taps, tune, delay, zero_idx, in_processor_chain):
        """ A general class for the register object, which uses FIR (finite impulse response) method to calculate
            a correct signal for kick from the register values. Because the register can be used for multiple kicker
            (in different locations), the filter coefficients are calculated in every call with
            the function namely coeff_generator.

        :param n_taps: length of the register (and length of filter)
        :param tune: a real number value of a betatron tune (e.g. 59.28 in horizontal or 64.31 in vertical direction
                for LHC)
        :param delay: a delay between storing to reading values  in turns
        :param zero_idx: location of the zero index of the filter coeffients
            'middle': an index of middle value in the register is 0. Values which have spend less time than that
                    in the register have negative indexes and vice versa
        :param in_processor_chain: if True, process() returns a signal, if False saves computing time
        """
        self.combination = 'individual'
        # self.combination = 'combined'
        self._zero_idx = zero_idx
        self._n_taps = n_taps

        super(FIR_Register, self).__init__(n_taps, tune, delay, in_processor_chain)
        self.required_variables = []

    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
        delta_phi = -1. * float(self._delay) * self._phase_shift_per_turn

        if self._zero_idx == 'middle':
            delta_phi -= float(self._n_taps/2) * self._phase_shift_per_turn

        if reader_phase_advance is not None:
            delta_position = self._phase_advance - reader_phase_advance
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self._phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        n = self._n_iter_left

        if self._zero_idx == 'middle':
            n -= self._n_taps/2
        # print delta_phi
        h = self.coeff_generator(n, delta_phi)
        h *= self._n_taps

        # print str(len(self)/2) + 'n: ' + str(n) + ' -> ' + str(h)  + ' (phi = ' + str(delta_phi) + ') from ' + str(self._phase_advance) + ' to ' + str(reader_phase_advance)

        return h*x1[0]

    def coeff_generator(self, n, delta_phi):
        """ Calculates filter coefficients
        :param n: index of the value
        :param delta_phi: total phase advance to the kicker for the value which index is 0
        :return: filter coefficient h
        """
        return 0.


class HilbertPhaseShiftRegister(FIR_Register):
    """ A register used in some damper systems at CERN. The correct signal is calculated by using FIR phase shifter,
    which is based on the Hilbert transform. It is recommended to use odd number of taps (e.g. 7) """

    def __init__(self,n_taps, tune, delay = 0, in_processor_chain=True):
        super(self.__class__, self).__init__(n_taps, tune, delay, 'middle', in_processor_chain)

    def coeff_generator(self, n, delta_phi):
        h = 0.

        if n == 0:
            h = np.cos(delta_phi)
        elif n % 2 == 1:
            h = -2. * np.sin(delta_phi) / (pi * float(n))

        return h