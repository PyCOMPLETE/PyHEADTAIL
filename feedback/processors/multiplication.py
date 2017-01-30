from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import copy

class Multiplication(object):
    __metaclass__ = ABCMeta
    """ An abstract class which multiplies the input signal by an array. The multiplier array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        multiplication_function(seed).
    """
    def __init__(self, seed, normalization = None, recalculate_multiplier = False, store_signal = False):
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

        self.signal_classes = (0,0)

        self.extensions = ['store']
        self._store_signal = store_signal

        if self._seed not in ['bin_length','bin_midpoint','signal']:
            self.extensions.append('bunch')
            self.required_variables = [self._seed]

        self.input_signal = None
        self.input_signal_parameters = None

        self.output_signal = None
        self.output_signal_parameters = None

    @abstractmethod
    def multiplication_function(self, seed):
        pass

    def process(self,signal_parameters, signal, slice_sets = None, *args, **kwargs):

        if (self._multiplier is None) or self._recalculate_multiplier:
            self.__calculate_multiplier(signal_parameters, signal, slice_sets)

        output_signal =  self._multiplier*signal

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_signal_parameters = copy.copy(signal_parameters)
            self.output_signal = np.copy(output_signal)
            self.output_signal_parameters = copy.copy(signal_parameters)

        # process the signal
        return signal_parameters, output_signal

    def __calculate_multiplier(self,signal_parameters, signal, slice_sets):
        self._multiplier = np.zeros(len(signal))

        if self._seed == 'ones':
            self._multiplier = self._multiplier + 1.
        elif self._seed == 'bin_length':
            np.copyto(self._multiplier, (signal_parameters.bin_edges[:,1]-signal_parameters.bin_edges[:,0]))
        elif self._seed == 'bin_midpoint':
            np.copyto(self._multiplier, ((signal_parameters.bin_edges[:,1]+signal_parameters.bin_edges[:,0])/2.))
        elif self._seed == 'normalized_bin_midpoint':

            for i in xrange(signal_parameters.n_segments):
                i_from = i * signal_parameters.n_bins_per_segment
                i_to = (i + 1) * signal_parameters.n_bins_per_segment

                np.copyto(self._multiplier[i_from:i_to], ((signal_parameters.bin_edges[i_from:i_to,1]+
                                                           signal_parameters.bin_edges[i_from:i_to,0])/2.
                                                          -signal_parameters.original_z_mids[i]))

        elif self._seed == 'signal':
            np.copyto(self._multiplier,signal)
        else:
            if len(signal) == len(slice_sets) * (len(slice_sets[0].z_bins) - 1):
                start_idx = 0
                for slice_set in slice_sets:
                    seed = getattr(slice_set,self._seed)
                    np.copyto(self._multiplier[start_idx:(start_idx+len(seed))],seed)
                    start_idx += len(seed)
            else:
                raise ValueError('Signal length does not correspond to the original signal length '
                                 'from the slice sets in the method Multiplication')

        # print 'self._multiplier: ' + str(self._multiplier)
        self._multiplier = self.multiplication_function(self._multiplier)

        if self._normalization is None:
            norm_coeff = 1.
        elif self._normalization == 'total':
            norm_coeff = float(np.sum(self._multiplier))
        elif self._normalization == 'average':
            norm_coeff = float(np.sum(self._multiplier))/float(len(self._multiplier))
        elif self._normalization == 'maximum':
            norm_coeff = float(np.max(self._multiplier))
        elif self._normalization == 'minimum':
            norm_coeff = float(np.min(self._multiplier))
        else:
            raise  ValueError('Unknown value in Multiplication._normalization')

        # TODO: try to figure out why this can not be written as
        # TODO:      self._multiplier /= norm_coeff
        self._multiplier =  self._multiplier / norm_coeff

    def clear(self):
        self._multiplier = None


class ChargeWeighter(Multiplication):
    """ weights signal with charge (macroparticles) of slices
    """

    def __init__(self, normalization = 'maximum', **kwargs):
        super(self.__class__, self).__init__('n_macroparticles_per_slice', normalization,recalculate_multiplier = True
                                             , **kwargs)
        self.label = 'Charge weighter'

    def multiplication_function(self,weight):
        return weight


class EdgeWeighter(Multiplication):
    """ Use an inverse of the Fermi-Dirac distribution function to increase signal strength on the edges of the bunch
    """

    def __init__(self,bunch_length,bunch_decay_length,maximum_weight = 10., **kwargs):
        """
        :param bunch_length: estimated width of the bunch
        :param bunch_decay_length: slope of the function on the edge of the bunch. Smaller value, steeper slope.
        :param maximum_weight: maximum value of the weight
        """
        self._bunch_length = bunch_length
        self._bunch_decay_length = bunch_decay_length
        self._maximum_weight=maximum_weight
        super(self.__class__, self).__init__('bin_midpoint', 'minimum', **kwargs)
        self.label = 'Edge weighter'

    def multiplication_function(self,weight):
        weight = np.exp((np.absolute(weight)-self._bunch_length/2.)/float(self._bunch_decay_length))+ 1.
        weight = np.clip(weight,1.,self._maximum_weight)
        return weight


class NoiseGate(Multiplication):
    """ Passes a signal which is greater/less than the threshold level.
    """

    def __init__(self,threshold, operator = 'greater', threshold_ref = 'amplitude', **kwargs):

        self._threshold = threshold
        self._operator = operator
        self._threshold_ref = threshold_ref
        super(self.__class__, self).__init__('signal',recalculate_multiplier = True, **kwargs)
        self.label = 'Noise gate'

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


class SignalMixer(Multiplication):
    def __init__(self,frequency,phase_shift, **kwargs):

        self._frequency = frequency
        self._phase_shift = phase_shift

        super(self.__class__, self).__init__('normalized_bin_midpoint', **kwargs)
        self.label = 'Signal mixer'

    def multiplication_function(self, seed):
        multiplier = np.sin(2.*pi*self._frequency*seed/c + self._phase_shift)
        return multiplier


class IdealAmplifier(Multiplication):
    def __init__(self,gain, **kwargs):

        self._gain = gain

        super(self.__class__, self).__init__('ones', **kwargs)
        self.label = 'IdealAmplifier'

    def multiplication_function(self, seed):
        return seed * self._gain


class MultiplicationFromFile(Multiplication):
    """ Multiplies the signal with an array, which is produced by interpolation from the loaded data. Note the seed for
        the interpolation can be any of those presented in the abstract function. E.g. a spatial weight can be
        determined by using a bin midpoint as a seed, nonlinear amplification can be modelled by using signal itself
        as a seed and etc...
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint', **kwargs):
        super(self.__class__, self).__init__(seed, **kwargs)
        self.label = 'Multiplication from file'

        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)

        if self._x_axis == 'time':
            self._data[:, 0] = self._data[:, 0] * c
        elif self._x_axis == 'position':
            pass
        else:
            raise ValueError('Unknown value in MultiplicationFromFile._x_axis')

    def multiplication_function(self, seed):
        return np.interp(seed, self._data[:, 0], self._data[:, 1])
