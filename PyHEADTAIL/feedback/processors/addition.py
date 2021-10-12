from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy.constants import c, pi
from ..core import default_macros

""" This file contains dimensionless impulse responses function for different
analog filters, which can be used in different signal processor implementations
(e.g. based on convolution and linear transformation).

@author Jani Komppula
@date: 11/10/2017
"""

class Addition(object, metaclass=ABCMeta):
    """ An abstract class which adds an array to the input signal. The addend array is produced by taking
        a slice property (determined by the input parameter 'seed') and passing it through the abstract method
        addend_function(seed).
    """

    def __init__(self, seed, normalization = None, recalculate_addend = False, label='Addition', **kwargs):
        """
        :param seed: a seed for the addend, which can be 'bin_length', 'bin_midpoint', 'signal' or any slice
            property found from slice_set
        :param normalization:
            None:
            'total_sum': The sum over the addend is equal to 1
            'segment_sum': The sum of the addend over each signal segment is equal to 1
            'total_average': The total average of the addend is equal to 1
            'segment_average': The average addend of each signal segment is equal to 1
            'total_integral': The total integral over the addend is equal to 1
            'segment_integral': The integral of the addend over each signal segment is equal to 1
            'total_min': The minimum of the addend is equal to 1
            'segment_min': The minimum of the addend in each signal segment is equal to 1
            'total_max': The minimum of the addend is equal to 1
            'segment_max': The minimum of the addend in each signal segment is equal to 1
        :param: recalculate_addend: if True, the weight is recalculated every time when process() is called
        """

        self._seed = seed
        self._normalization = normalization
        self._recalculate_addend = recalculate_addend

        self._addend = None

        self.signal_classes = (0,0)

        self.extensions = []
        self._macros = [] + default_macros(self, 'Addition', **kwargs)

        if self._seed not in ['bin_length','bin_midpoint','signal']:
            self.extensions.append('bunch')
            self.required_variables = [self._seed]

    @abstractmethod
    def addend_function(self, seed):
        pass

    def process(self,parameters, signal, slice_sets = None, *args, **kwargs):

        if (self._addend is None) or self._recalculate_addend:
            self.__calculate_addend(parameters, signal, slice_sets)

        output_signal = signal + self._addend

        # process the signal
        return parameters, output_signal

    def __calculate_addend(self,parameters, signal, slice_sets):
        self._addend = np.zeros(len(signal))

        if self._seed == 'ones':
            self._addend = self._addend + 1.
        elif self._seed == 'bin_length':
            np.copyto(self._addend, (parameters.bin_edges[:,1]-parameters.bin_edges[:,0]))
        elif self._seed == 'bin_midpoint':
            np.copyto(self._addend, ((parameters.bin_edges[:,1]+parameters.bin_edges[:,0])/2.))
        elif self._seed == 'normalized_bin_midpoint':

            for i in range(parameters.n_segments):
                i_from = i * parameters.n_bins_per_segment
                i_to = (i + 1) * parameters.n_bins_per_segment

                np.copyto(self._addend[i_from:i_to], ((parameters.bin_edges[i_from:i_to,1]+
                                                           parameters.bin_edges[i_from:i_to,0])/2.
                                                          -parameters.original_z_mids[i]))

        elif self._seed == 'signal':
            np.copyto(self._addend,signal)
        else:
            if len(signal) == len(slice_sets) * (len(slice_sets[0].z_bins) - 1):
                start_idx = 0
                for slice_set in slice_sets:
                    seed = getattr(slice_set,self._seed)
                    np.copyto(self._addend[start_idx:(start_idx+len(seed))],seed)
                    start_idx += len(seed)
                np.copyto(self._addend, self._addend[::-1])
            else:
                raise ValueError('Signal length does not correspond to the original signal length '
                                 'from the slice sets in the method Addition')

        self._addend = self.addend_function(self._addend)

        # NOTE: add options for average bin integrals?
        if self._normalization is None:
            norm_coeff = 1.

        elif self._normalization == 'total_sum':
            norm_coeff = float(np.sum(self._addend))

        elif self._normalization == 'segment_sum':
            norm_coeff = np.ones(len(self._addend))
            for i in range(parameters.n_segments):
                i_from = i*parameters.n_bins_per_segment
                i_to = (i+1)*parameters.n_bins_per_segment
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.sum(self._addend[i_from:i_to]))

        elif self._normalization == 'total_average':
            norm_coeff = float(np.sum(self._addend))/float(len(self._addend))

        elif self._normalization == 'segment_average':
            norm_coeff = np.ones(len(self._addend))
            for i in range(parameters.n_segments):
                i_from = i*parameters.n_bins_per_segment
                i_to = (i+1)*parameters.n_bins_per_segment
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.sum(self._addend[i_from:i_to]))/float(parameters.n_bins_per_segment)

        elif self._normalization == 'total_integral':
            bin_widths = parameters.bin_edges[:,1] - parameters.bin_edges[:,0]
            norm_coeff = np.sum(self._addend*bin_widths)

        elif self._normalization == 'segment_integral':
            bin_widths = parameters.bin_edges[:,1] - parameters.bin_edges[:,0]
            norm_coeff = np.ones(len(self._addend))
            for i in range(parameters.n_segments):
                i_from = i*parameters.n_bins_per_segment
                i_to = (i+1)*parameters.n_bins_per_segment
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.sum(self._addend[i_from:i_to]*bin_widths[i_from:i_to]))

        elif self._normalization == 'total_min':
            norm_coeff = float(np.min(self._addend))

        elif self._normalization == 'segment_min':
            norm_coeff = np.ones(len(self._addend))
            for i in range(parameters.n_segments):
                i_from = i*parameters.n_bins_per_segment
                i_to = (i+1)*parameters.n_bins_per_segment
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.min(self._addend[i_from:i_to]))

        elif self._normalization == 'total_max':
            norm_coeff = float(np.max(self._addend))

        elif self._normalization == 'segment_max':
            norm_coeff = np.ones(len(self._addend))
            for i in range(parameters.n_segments):
                i_from = i*parameters.n_bins_per_segment
                i_to = (i+1)*parameters.n_bins_per_segment
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.max(self._addend[i_from:i_to]))
        else:
            raise  ValueError('Unknown value in Addition._normalization')

        # TODO: try to figure out why this can not be written as
        # TODO:      self._addend /= norm_coeff
        self._addend = self._addend / norm_coeff

    def clear(self):
        self._addend = None

class AdditionFromFile(Addition):
    """ Adds an array to the signal, which is produced by interpolation from the external data file. Note the seed
        (unit) for the interpolation can be any of those available for the seed.
        (i.e. location, sigma, or a number of macroparticles per slice, etc.)
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint', **kwargs):
        super(self.__class__, self).__init__(seed, label = 'Addition from file', **kwargs)

        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)

        if self._x_axis == 'time':
            pass
        elif self._x_axis == 'position':
            self._data[:, 0] = self._data[:, 0] / c
        else:
            raise ValueError('Unknown value in AdditionFromFile._x_axis')

    def addend_function(self, seed):
        return np.interp(seed, self._data[:, 0], self._data[:, 1])


class NoiseGenerator(Addition):
    """ Adds noise. The noise level is given as an absolute RMS noise level in the units of signal
        (reference_level = 'absolute') or a relative RMS level from the maximum value of the signal
        (reference_level = 'maximum'). Options for the noise distribution are a Gaussian (normal) distribution
        (distribution = 'normal') or an uniform distribution (distribution = 'uniform')
    """

    def __init__(self,RMS_noise_level,reference_level = 'absolute', distribution = 'normal', **kwargs):
        super(self.__class__, self).__init__('signal', recalculate_addend=True,
             label = 'Noise generator', **kwargs)

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
