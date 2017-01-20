import math, copy
from collections import deque
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi

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

    def __init__(self, n_avg, tune, delay, in_processor_chain,store_signal = False):
        """
        :param n_avg: a number of register values (in turns) have been stored after the delay
        :param tune: a real number value of a betatron tune (e.g. 59.28 in horizontal or 64.31 in vertical direction
                for LHC)
        :param delay: a delay between storing to reading values  in turns
        :param in_processor_chain: if True, process() returns a signal
        """
        self.signal_parameters = None
        self.beam_parameters = None
        self._delay = delay
        self._n_avg = n_avg
        self._phase_shift_per_turn = 2.*pi * tune
        self._in_processor_chain = in_processor_chain
        self.combination = None

        self._max_reg_length = self._delay+self._n_avg
        self._register = deque()

        self._n_iter_left = -1

        self._reader_position = None

        # if n_bins is not None:
        #     self._register.append(np.zeros(n_bins))


        self.extensions = ['store', 'register']

        self.label = None
        self._store_signal = store_signal
        self.input_signal = None
        self.input_signal_parameters = None
        self.output_signal = None
        self.output_signal_parameters = None

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
            return (self._register[self._n_iter_left],None,delay,self.beam_parameters.phase_advance)

    def process(self,signal_parameters, signal, *args, **kwargs):

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_signal_parameters = copy.copy(signal_parameters)

        if self.beam_parameters is None:
            self.signal_parameters = signal_parameters
            self.beam_parameters = signal_parameters.beam_parameters

        self._register.append(signal)

        if len(self._register) > self._max_reg_length:
            self._register.popleft()

        if self._in_processor_chain == True:
            temp_signal = np.zeros(len(signal))
            if len(self) > 0:
                prev = (np.zeros(len(self._register[0])),None,0,self.beam_parameters.phase_advance)

                for value in self:
                    combined = self.combine(value,prev,None)
                    prev = value
                    temp_signal += combined / float(len(self))

            if self._store_signal:
                self.output_signal = np.copy(temp_signal)
                self.output_signal_parameters = copy.copy(signal_parameters)

            return signal_parameters, temp_signal

    @abstractmethod
    def combine(self,x1,x2,reader_position,x_to_xp = False):

        pass


class VectorSumRegister(Register):

    def __init__(self, n_avg, tune, delay = 0, in_processor_chain=True,**kwargs):
        self.combination = 'combined'
        super(self.__class__, self).__init__(n_avg, tune, delay, in_processor_chain,**kwargs)
        self.label = 'Vector sum register'

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
    def __init__(self, n_avg, tune, delay = 0, in_processor_chain=True,**kwargs):

        self.combination = 'individual'

        super(self.__class__, self).__init__(n_avg, tune, delay, in_processor_chain,**kwargs)
        self.label = 'Cosine sum register'

    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
        delta_phi = x1[2]
        if reader_phase_advance is not None:
            delta_position = self.beam_parameters.phase_advance - reader_phase_advance
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self._phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        return 2.*math.cos(delta_phi)*x1[0]






class FIR_Register(Register):
    def __init__(self, n_taps, tune, delay, zero_idx, in_processor_chain,**kwargs):
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

        super(FIR_Register, self).__init__(n_taps, tune, delay, in_processor_chain,**kwargs)
        self.required_variables = []

    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
        delta_phi = -1. * float(self._delay) * self._phase_shift_per_turn

        if self._zero_idx == 'middle':
            delta_phi -= float(self._n_taps/2) * self._phase_shift_per_turn

        if reader_phase_advance is not None:
            delta_position = self.beam_parameters.phase_advance - reader_phase_advance
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

    def __init__(self,n_taps, tune, delay = 0, in_processor_chain=True,**kwargs):
        super(self.__class__, self).__init__(n_taps, tune, delay, 'middle', in_processor_chain,**kwargs)
        self.label = 'HilbertPhaseShiftRegister'

    def coeff_generator(self, n, delta_phi):
        h = 0.

        if n == 0:
            h = np.cos(delta_phi)
        elif n % 2 == 1:
            h = -2. * np.sin(delta_phi) / (pi * float(n))

        return h