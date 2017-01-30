from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import copy
from scipy.constants import c, pi

class Addition(object):
    __metaclass__ = ABCMeta
    """ An abstract class which adds an array to the input signal. The addend array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        addend_function(seed).
    """

    def __init__(self, seed, normalization = None, recalculate_addend = False, store_signal = False):
        """
        :param seed: 'bin_length', 'bin_midpoint', 'signal' or a property of a slice, which can be found
            from slice_set
        :param normalization:
            'total_weight':  a sum of the multiplier array is equal to 1.
            'average_weight': an average in  the multiplier array is equal to 1,
            'maximum_weight': a maximum value in the multiplier array value is equal to 1
            'minimum_weight': a minimum value in the multiplier array value is equal to 1
        :param: recalculate_addend: if True, the weight is recalculated every time when process() is called
        """

        self._seed = seed
        self._normalization = normalization
        self._recalculate_addend = recalculate_addend

        self._addend = None

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
    def addend_function(self, seed):
        pass

    def process(self,signal_parameters, signal, slice_sets = None, *args, **kwargs):

        if (self._addend is None) or self._recalculate_addend:
            self.__calculate_addend(signal_parameters, signal, slice_sets)

        output_signal = signal + self._addend

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_signal_parameters = copy.copy(signal_parameters)
            self.output_signal = np.copy(output_signal)
            self.output_signal_parameters = copy.copy(signal_parameters)

        # process the signal
        return signal_parameters, output_signal

    def __calculate_addend(self,signal_parameters, signal, slice_sets):
        self._addend = np.zeros(len(signal))

        if self._seed == 'ones':
            self._addend = self._addend + 1.
        elif self._seed == 'bin_length':
            np.copyto(self._addend, (signal_parameters.bin_edges[:,1]-signal_parameters.bin_edges[:,0]))
        elif self._seed == 'bin_midpoint':
            np.copyto(self._addend, ((signal_parameters.bin_edges[:,1]+signal_parameters.bin_edges[:,0])/2.))
        elif self._seed == 'normalized_bin_midpoint':

            for i in xrange(signal_parameters.n_segments):
                i_from = i * signal_parameters.n_bins_per_segment
                i_to = (i + 1) * signal_parameters.n_bins_per_segment

                np.copyto(self._addend[i_from:i_to], ((signal_parameters.bin_edges[i_from:i_to,1]+
                                                           signal_parameters.bin_edges[i_from:i_to,0])/2.
                                                          -signal_parameters.original_z_mids[i]))

        elif self._seed == 'signal':
            np.copyto(self._addend,signal)
        else:
            if len(signal) == len(slice_sets) * (len(slice_sets[0].z_bins) - 1):
                start_idx = 0
                for slice_set in slice_sets:
                    seed = getattr(slice_set,self._seed)
                    np.copyto(self._addend[start_idx:(start_idx+len(seed))],seed)
                    start_idx += len(seed)
            else:
                raise ValueError('Signal length does not correspond to the original signal length '
                                 'from the slice sets in the method Addition')

        self._addend = self.addend_function(self._addend)

        if self._normalization is None:
            norm_coeff = 1.
        elif self._normalization == 'total':
            norm_coeff = float(np.sum(self._addend))
        elif self._normalization == 'average':
            norm_coeff = float(np.sum(self._addend))/float(len(self._addend))
        elif self._normalization == 'maximum':
            norm_coeff = float(np.max(self._addend))
        elif self._normalization == 'minimum':
            norm_coeff = float(np.min(self._addend))
        else:
            raise  ValueError('Unknown value in Addition._normalization')

        # TODO: try to figure out why this can not be written as
        # TODO:      self._addend /= norm_coeff
        self._addend = self._addend / norm_coeff

    def clear(self):
        self._addend = None

class AdditionFromFile(Addition):
    """ Adds an array to the signal, which is produced by interpolation from the loaded data. Note the seed for
        the interpolation can be any of those presented in the abstract function.
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint', **kwargs):
        super(self.__class__, self).__init__(seed, **kwargs)
        self.label = 'Addition from file'

        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)

        if self._x_axis == 'time':
            self._data[:, 0] = self._data[:, 0] * c
        elif self._x_axis == 'position':
            pass
        else:
            raise ValueError('Unknown value in AdditionFromFile._x_axis')

    def addend_function(self, seed):
        return np.interp(seed, self._data[:, 0], self._data[:, 1])


class NoiseGenerator(Addition):
    """ Adds noise to a signal. The noise level is given as RMS value of the absolute level (reference_level = 'absolute'),
        a relative RMS level to the maximum signal (reference_level = 'maximum') or a relative RMS level to local
        signal values (reference_level = 'local'). Options for the noise distribution are a Gaussian normal distribution
        (distribution = 'normal') and an uniform distribution (distribution = 'uniform')
    """

    def __init__(self,RMS_noise_level,reference_level = 'absolute', distribution = 'normal', **kwargs):
        super(self.__class__, self).__init__('signal', recalculate_addend=True, **kwargs)
        self.label = 'Noise generator'

        self._RMS_noise_level = RMS_noise_level
        self._reference_level = reference_level
        self._distribution = distribution

    def signal_classes(self):
        return (0,0)

    def addend_function(self,seed):

        if self._distribution == 'normal' or self._distribution is None:
            randoms = np.random.randn(len(seed))
        elif self._distribution == 'uniform':
            randoms = 1./0.577263*(-1.+2.*np.random.rand(len(seed)))
        else:
            raise ValueError('Unknown value in NoiseGenerator._distribution')

        if self._reference_level == 'absolute':
            addend = self._RMS_noise_level*randoms
        elif self._reference_level == 'maximum':
            addend = self._RMS_noise_level*np.max(seed)*randoms
        elif self._reference_level == 'local':
            addend = seed*self._RMS_noise_level*randoms
        else:
            raise ValueError('Unknown value in NoiseGenerator._reference_level')

        return addend
