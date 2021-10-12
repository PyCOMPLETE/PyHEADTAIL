import math, copy
from collections import deque
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import pi

from ..core import Parameters, default_macros

"""Signal processors based on registers and combiners.

@author Jani Komppula
@date: 11/10/2017
"""

class Register(object):
    """
    Stores signals to the register. The obejct is iterable, i.e. iteration
    returns the stored signals after the given delay.
    """
    def __init__(self, n_values, tune, delay=0, **kwargs):
        """
        Parameters
        ----------
        n_values : number
          A maximum number of signals stored and returned (in addition to
          the delay)
        tune : number
          A real number value of a betatron tune
        delay : number
          A number of turns the signal kept in the register before returning it

        """

        self._n_values = n_values
        self._delay = delay
        self._phase_advance_per_turn = 2. * np.pi * tune

        self._n_iter_left = 0
        self._signal_register = deque(maxlen=(n_values + delay))
        self._parameter_register = deque(maxlen=(n_values + delay))

        self.extensions = ['register']
        self._macros = [] + default_macros(self, 'Register', **kwargs)

    @property
    def parameters(self):
        if len(self._parameter_register) > 0:
            return self._parameter_register[0]
        else:
            return None

    @property
    def phase_advance_per_turn(self):
        return self._phase_advance_per_turn

    @property
    def delay(self):
        return self._delay

    @property
    def maxlen(self):
        return self._n_values

    def __len__(self):
        """
        Returns a number of signals in the register after the delay.
        """
        return max((len(self._signal_register) - self._delay), 0)

    def __iter__(self):
        """
        Calculates how many iterations are required
        """
        self._n_iter_left = len(self)

        return self

    def __next__(self):
        if self._n_iter_left < 1:
            raise StopIteration

        else:
            delay = -1. * (len(self._signal_register) - self._n_iter_left) \
                            * self._phase_advance_per_turn
            self._n_iter_left -= 1

            return (self._parameter_register[self._n_iter_left],
                    self._signal_register[self._n_iter_left], delay)

    def process(self, parameters, signal, *args, **kwargs):
        self._parameter_register.append(parameters)
        self._signal_register.append(signal)

        return parameters, signal


class UncorrectedDelay(object):
    """ Delays the signal in the units of turns without any betatron pahse
    advance correction
    """
    def __init__(self, delay, **kwargs):

        self._delay = delay
        self._register = Register(n_values=1, tune=1., delay=self._delay)

        self.extensions = ['register']
        self._macros = [] + default_macros(self, 'UncorrectedDelay', **kwargs)

    @property
    def delay(self):
        return self._delay

    def process(self, parameters, signal, *args, **kwargs):
        self._register.process(parameters, signal, *args, **kwargs)
        output_parameters = None
        output_signal = None

        for (parameters_i, signal_i, delay_i) in self._register:
            output_parameters = parameters_i
            output_signal = signal_i

        if output_parameters is None:
            output_parameters = parameters
            output_signal = np.zeros(len(signal))

        return output_parameters, output_signal



class Combiner(object, metaclass=ABCMeta):
    def __init__(self, registers, target_location, target_beta=None,
                 additional_phase_advance=0., beta_conversion = '0_deg', **kwargs):
        """
        Parameters
        ----------
        registers : list
          A list of registers, which are a source for the signal
        target_location : number
          A target phase advance in radians of betatron motion
        additional_phase_advance : number
          Additional phase advance for the target location.
          For example, np.pi/2. for shift from displacement in the pick up to
          divergenve in the kicker
        """

        self._registers = registers
        self._target_location = target_location
        self._target_beta = target_beta
        self._additional_phase_advance = additional_phase_advance
        self._beta_conversion = beta_conversion

        if self._beta_conversion == '0_deg':
            pass
        elif self._beta_conversion == '90_deg':
            self._additional_phase_advance += pi/2.
        else:
            raise ValueError('Unknown beta conversion type.')

        self._combined_parameters = None


        self.extensions = ['combiner']
        self._macros = [] + default_macros(self, 'Combiner', **kwargs)

    @abstractmethod
    def combine(self, registers, target_location, target_beta, additional_phase_advance, beta_conversion):
        pass

    def process(self, parameters=None, signal=None, *args, **kwargs):

        output_signal = self.combine(self._registers,
                              self._target_location,
                              self._target_beta,
                              self._additional_phase_advance,
                              self._beta_conversion)

        if self._combined_parameters is None:
            self._combined_parameters = copy.copy(self._registers[0].parameters)
            self._combined_parameters['location'] = self._target_location
            self._combined_parameters['beta'] = self._target_beta

        return self._combined_parameters, output_signal

class CosineSumCombiner(Combiner):
    """ A combiner, which utilizes "Cosine sum"- algorithm for the betatron
    phase advance correction.

    In the other words, it can be proven that, the sum of the singnals
    multiplied by cos(phase advance to the target) approaches a half value of
    the correct signal, when the number of the singal with equally distributed
    (random) phase advances increases.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        registers : list
          A list of registers, which are a source for the signal
        target_location : number
          A target phase advance in radians of betatron motion
        additional_phase_advance : number
          Additional phase advance for the target location.
          For example, np.pi/2. for shift from displacement in the pick up to
          divergenve in the kicker
        """
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'Cosine sum combiner'

    def combine(self, registers, target_location, target_beta, additional_phase_advance, beta_conversion):
        combined_signal = None
        n_signals = 0

        for register in registers:
            for (parameters, signal, delay) in register:
                if combined_signal is None:
                    combined_signal = np.zeros(len(signal))
                delta_position = parameters['location'] \
                                - target_location

                if delta_position > 0:
                    delta_position -= register.phase_advance_per_turn

                delta_phi = delay + delta_position - additional_phase_advance
                n_signals += 1

                if target_beta is not None:
                    beta_correction = 1. / np.sqrt(parameters['beta'] * target_beta)
                else:
                    beta_correction = 1.

                combined_signal += beta_correction * 2. * math.cos(delta_phi) * signal

        if combined_signal is not None:
            combined_signal = combined_signal/float(n_signals)

        return combined_signal

class DummyCombiner(Combiner):
    """ A combiner, which by passes the signal without any corrections
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        registers : list
          A list of registers, which are a source for the signal
        target_location : number
          A target phase advance in radians of betatron motion
        additional_phase_advance : number
          Additional phase advance for the target location.
          For example, np.pi/2. for shift from displacement in the pick up to
          divergenve in the kicker
        """
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'Dummy combiner'

    def combine(self, registers, target_location, target_beta, additional_phase_advance, beta_conversion):
        combined_signal = None
        
        if len(registers[0]) > 0:
        
            for (parameters, signal, delay) in registers[0]:
                combined_signal = signal
    
            if target_beta is not None:
                beta_correction = 1. / np.sqrt(parameters['beta'] * target_beta)
            else:
                beta_correction = 1.

            return beta_correction*combined_signal
    
        else:
            return combined_signal

class HilbertCombiner(Combiner):
    """ A combiner, which utilizes a algorithm based on the Hilbert transform.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        registers : list
          A list of registers, which are a source for the signal
        target_location : number
          A target phase advance in radians of betatron motion
        additional_phase_advance : number
          Additional phase advance for the target location.
          For example, np.pi/2. for shift from displacement in the pick up to
          divergenve in the kicker
        """
        if 'n_taps' in kwargs:
            self._n_taps = kwargs['n_taps']
        else:
            self._n_taps = None

        self._coefficients = None
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'Hilbert combiner'

    @property
    def n_taps(self):
        return self._n_taps

    @n_taps.setter
    def n_taps(self, value):
        self._n_taps = value


    def combine(self, registers, target_location, target_beta, additional_phase_advance, beta_conversion):
        if self._coefficients is None:
#            print registers
            if self._n_taps is None:
                self._n_taps = registers[0].maxlen
            self._coefficients = [None]*len(registers)

        combined_signal = None

        for i, register in enumerate(registers):
            if len(register) >= len(self._coefficients):
                if self._coefficients[i] is None:
                    self._coefficients[i] = self.__generate_coefficients(
                            register, target_location, target_beta,
                            additional_phase_advance)

                for j, (parameters, signal, delay) in enumerate(register):

                    if target_beta is not None:
                        beta_correction = 1. / np.sqrt(parameters['beta'] * target_beta)
                    else:
                        beta_correction = 1.

                    if combined_signal is None:
                        combined_signal = np.zeros(len(signal))

                    combined_signal += beta_correction * self._coefficients[i][j] * signal

        if combined_signal is not None:
            combined_signal = combined_signal/float(len(registers))

        return combined_signal

    def __generate_coefficients(self, register, target_location, target_beta, additional_phase_advance):
        parameters = register.parameters

        delta_phi = -1. * float(register.delay) \
                    * register.phase_advance_per_turn

        delta_phi -= float(self._n_taps/2) * register.phase_advance_per_turn

        delta_position = parameters['location'] - target_location
        delta_phi += delta_position
        if delta_position > 0:
            delta_phi -= register.phase_advance_per_turn

        delta_phi -= additional_phase_advance

        coefficients = np.zeros(self._n_taps)

        for i in range(self._n_taps):
            n = self._n_taps-i-1
            n -= self._n_taps/2
            h = 0.

            if n == 0:
                h = np.cos(delta_phi)
            elif n % 2 == 1:
                h = -2. * np.sin(delta_phi) / (pi * float(n))
            coefficients[i] = h
        return coefficients


class VectorSumCombiner(Combiner):
    """ A combiner, which utilizes vector calculus for the correction.

    It can be proven that if the oscillation amplitude doesn't change
    turn by turn (e.g. the damper gain is low), the correction is
    ideal if the signal from two different phase advances (e.g. turns or
    pickups) are available.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        registers : list
          A list of registers, which are a source for the signal
        target_location : number
          A target phase advance in radians of betatron motion
        additional_phase_advance : number
          Additional phase advance for the target location.
          For example, np.pi/2. for shift from displacement in the pick up to
          divergenve in the kicker
        """
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'Vector sum combiner'
        self._warning_printed = False

    def combine(self, registers, target_location, target_beta,
                additional_phase_advance, beta_conversion):

        combined_signal = None
        n_signals = 0

        if len(registers) == 1:
            prev_parameters = None
            prev_signal = None
            prev_delay = None

            if len(registers[0]) > 1:

                for i, (parameters, signal, delay) in enumerate(registers[0]):
                    if i == 0:
                        combined_signal = np.zeros(len(signal))
                        prev_signal = np.zeros(len(signal))
                    else:
                        phase_advance_per_turn = (
                                registers[0].phase_advance_per_turn)
                        location_1 = prev_parameters['location']
                        beta_1 = prev_parameters['beta']
                        delay_1 = prev_delay
                        location_2 = parameters['location']
                        beta_2 = prev_parameters['beta']
                        delay_2 = delay

                        combined_signal = combined_signal + \
                                        self.__combine_signals(prev_signal, delay_1,
                                                               location_1, beta_1,
                                                               signal, delay_2,
                                                               location_2, beta_2,
                                                               target_location,
                                                               target_beta,
                                                               beta_conversion,
                                                               phase_advance_per_turn,
                                                               additional_phase_advance)
                        n_signals += 1

                    np.copyto(prev_signal,signal)
                    prev_parameters = parameters
                    prev_delay = delay

        elif len(registers) > 1:
            prev_register = registers[0]

            for register in registers[1:]:
                for (parameters_1, signal_1, delay_1), (parameters_2, signal_2, delay_2) in zip(prev_register,register):
                        if combined_signal is None:
                            combined_signal = np.zeros(len(signal_1))

                        phase_advance_per_turn = (
                                prev_register.phase_advance_per_turn)
                        location_1 = parameters_1['location']
                        beta_1 = parameters_1['beta']
                        location_2 = parameters_2['location']
                        beta_2 = parameters_2['beta']

                        combined_signal = combined_signal + \
                                        self.__combine_signals(signal_1, delay_1,
                                                               location_1, beta_1,
                                                               signal_2, delay_2,
                                                               location_2, beta_2,
                                                               target_location,
                                                               target_beta,
                                                               beta_conversion,
                                                               phase_advance_per_turn,
                                                               additional_phase_advance
                                                               )
                        n_signals += 1

                prev_register = register
        else:
            raise ValueError('At least one register must be given.')

        if combined_signal is not None:
            combined_signal = combined_signal / float(n_signals)

        return combined_signal

    def __combine_signals(self, signal_1, delay_1, location_1, beta_1,
                          signal_2, delay_2, location_2, beta_2,
                          target_location, target_beta, beta_conversion,
                          phase_advance_per_turn, additional_phase_advance):

        readings_angle_diff, final_rotation_angle = (
                self.__determine_angles(target_location, phase_advance_per_turn,
                                location_1, delay_1, location_2, delay_2)
                )
        final_rotation_angle += additional_phase_advance

        re, im = self.__determine_vector(signal_1, beta_1, signal_2, beta_2,
                                         readings_angle_diff)

        calculated_signal = self.__rotate_vector(re, im, final_rotation_angle)

        if target_beta is not None:
            if beta_conversion == '90_deg':
                beta_correction = 1./np.sqrt(beta_1*target_beta)
            elif beta_conversion == '0_deg':
                beta_correction = np.sqrt(target_beta/beta_1)
        else:
            beta_correction = 1.

        return beta_correction * calculated_signal

    def __determine_angles(self, target_location, phase_advance_per_turn,
                           signal_1_location, signal_1_delay,
                           signal_2_location, signal_2_delay):

        readings_location_difference = signal_2_location - signal_1_location
        if readings_location_difference < 0.:
            readings_location_difference += readings_location_difference

        readings_delay_difference = signal_2_delay - signal_1_delay
        readings_phase_difference = readings_location_difference \
                                        + readings_delay_difference

        if self._warning_printed == False:
            if (readings_phase_difference%(-1.*np.pi) > 0.2) or (readings_phase_difference%np.pi < 0.2):
                self._warning_printed = True
                print("WARNING: It is recommended that the angle between the readings is at least 12 deg")

        target_location_difference = target_location - signal_1_location
        if target_location_difference < 0.:
            target_location_difference += readings_location_difference

        target_delay_difference = -1. * signal_1_delay
        target_phase_difference = target_location_difference \
                                        + target_delay_difference

        return readings_phase_difference, target_phase_difference

    def __determine_vector(self, signal_1, beta_1, signal_2, beta_2,
                           angle_difference):
        """
        """
        s = np.sin(angle_difference)
        c = np.cos(angle_difference)

        re = signal_1
        im = (1./s) * np.sqrt(beta_1/beta_2) * signal_2 - (c/s) * signal_1
        return re, im

    def __rotate_vector(self, re, im, rotation_angle):

        s = np.sin(rotation_angle)
        c = np.cos(rotation_angle)

        return c*re+s*im


class FIRCombiner(Combiner):
    """ A combiner object, which correct the betatron phase advance by using
    the given coefficient.
    """

    def __init__(self, coefficients, *args, **kwargs):
        """
        Parameters
        ----------
        coefficients: list
            A list of FIR coefficients
        """
        self._coefficients = coefficients
        super(FIRCombiner, self).__init__(*args, **kwargs)
        self.label = 'FIR combiner'

    def combine(self, registers, target_location, target_beta,
                additional_phase_advance, beta_conversion):
        combined_signal = None

        for register in registers:
            if len(register) >= len(self._coefficients):
                for i, (parameters, signal, delay) in enumerate(register):
                    if combined_signal is None:
                        combined_signal = np.zeros(len(signal))
                    if i < len(self._coefficients):
                        combined_signal += self._coefficients[i] * signal

        return combined_signal



class DCRemovedVectorSumCombiner(FIRCombiner):
    """ A 'notch filttered', i.e. DC-level removed, version of the vector sum
        combiner. It is a three tap FIR filter, which has been derived by using
        asumptions that a beam is a rotating vector in (x, xp)-plane and 
        x-values can be measured in different turns, but they contains an
        unknown constant DC-offset.
         
        This version gives mathematically exact correction when tune
        is well known. When tune error exists the version induces only low 
        noise in comparison to other types of combiners.
        
        Developed by J. Komppula @ 2017.
    """
    def __init__(self, tune, delay=0, *args, **kwargs):
        def calculate_coefficients(tune, delay):
            ppt = -tune * 2.* np.pi
            c12 = np.cos(1.*ppt)
            s12 = np.sin(1.*ppt)
            c13 = np.cos(2.*ppt)
            s13 = np.sin(2.*ppt)
            c14 = np.cos((2+delay)*ppt)
            s14 = np.sin((2+delay)*ppt)
            
            divider = -1.*(-c12*s13+c13*s12-s12+s13)
        
            cx1 = c14*(1-(c12*s13-c13*s12)/divider)+s14*(-c12+c13)/divider
            cx2 = (c14*(-(-s13))+s14*(-c13+1))/divider
            cx3 = (c14*(-(s12))+s14*(c12-1))/divider

            return [cx3, cx2, cx1]
        
        coefficients = calculate_coefficients(tune, delay)
        
        super(DCRemovedVectorSumCombiner, self).__init__(coefficients,*args, **kwargs)
        self.label = 'FIR combiner'
    
    

class TurnFIRFilter(object):
    """A signal processor, which can be used as a FIR filer in turn domain.
    """

    def __init__(self, coefficients, tune, delay = 0, additional_phase_advance = 0., **kwargs):
        """
        Parameters
        ----------
        coefficients: list
            A list of FIR coefficients
        tune: float
            A betatron tune of the plane
        delay: int
            A delay of the signal in the units of turn before the filter
        addtional_phase_advance: float
            An additional betatron phase advance in radians to be taken into
            account to the betatron phase correction.
        """
        self._coefficients = coefficients
        self._tune = tune
        self._additional_phase_advance = additional_phase_advance
        self._register = Register(len(self._coefficients), self._tune, delay)
        self._combiner = None

        self.extensions = []
        self._macros = [] + default_macros(self, 'TurnFIRFilter', **kwargs)

    def process(self, parameters, signal, *args, **kwargs):
        self._register.process(parameters, signal, *args, **kwargs)
        if self._combiner is None:
            self.__init_combiner(parameters)

        output_parameters, output_signal = self._combiner.process(parameters,
                                                                  signal,
                                                                  *args,
                                                                  **kwargs)

        if output_signal is None:
            output_parameters = parameters
            output_signal = np.zeros(len(signal))

        return output_parameters, output_signal

    def __init_combiner(self, parameters):
        registers = [self._register]
        target_location = parameters['location']
        target_beta = parameters['beta']
        extra_phase = self._additional_phase_advance
        self._combiner = FIRCombiner(self._coefficients,registers, target_location,
                                                   target_beta, extra_phase)

class TurnDelay(object):
    """ Delays the signal a number of turns given as an input parameter.
    """
    def __init__(self, delay, tune, n_taps=2, combiner='vector_sum',
                 additional_phase_advance=0, **kwargs):
        """
        Parameters
        ----------
        delay: int
            A number of turns signal is delayed
        tune: float
            A betatron tune of the plane
        n_taps: int
            A number of turns of data used for betatron phase advance
            correction of the delay. Note that typically the group delay is
            delay + n_taps/2 depending on the correction algorithm.
        combiner: str or object
            Combiner object, which is used for betatron correction
        addtional_phase_advance: float
            An additional betatron phase advance in radians to be taken into
            account to the betatron phase correction.

        """

        self._delay = delay
        self._tune = tune
        self._n_taps = n_taps
        self._combiner_type = combiner
        self._additional_phase_advance = additional_phase_advance

        self._register = Register(self._n_taps, self._tune, self._delay)
        self._combiner = None

        self.extensions = []
        self._macros = [] + default_macros(self, 'TurnDelay', **kwargs)

    def process(self, parameters, signal, *args, **kwargs):
        self._register.process(parameters, signal, *args, **kwargs)

        if self._combiner is None:
            self.__init_combiner(parameters)

        output_parameters, output_signal = self._combiner.process(parameters,
                                                                  signal,
                                                                  *args,
                                                                  **kwargs)
#        print output_signal
        if output_signal is None:
            output_parameters = parameters
            output_signal = np.zeros(len(signal))

        return output_parameters, output_signal

    def __init_combiner(self, parameters):
        registers = [self._register]
        target_location = parameters['location']
        target_beta = parameters['beta']
        extra_phase = self._additional_phase_advance

        if isinstance(self._combiner_type, str):
            if self._combiner_type == 'vector_sum':
                self._combiner = VectorSumCombiner(registers, target_location,
                                                   target_beta, extra_phase)
            elif self._combiner_type == 'cosine_sum':
                self._combiner = CosineSumCombiner(registers, target_location,
                                                   target_beta, extra_phase)
            elif self._combiner_type == 'hilbert':
                self._combiner = HilbertCombiner(registers, target_location,
                                                 target_beta, extra_phase)
            elif self._combiner_type == 'DCrem_vector_sum':
                self._combiner = DCRemovedVectorSumCombiner(self._tune,
                                                            self._delay,
                                                            registers,
                                                            target_location,
                                                            target_beta,
                                                            extra_phase)
            else:
                raise ValueError('Unknown combiner type')
        else:
            self._combiner = self._combiner_type(registers, target_location,
                                                 extra_phase)
