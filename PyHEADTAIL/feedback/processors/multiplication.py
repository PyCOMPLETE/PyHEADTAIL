from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
from ..core import default_macros

"""Signal processors based on multiplication operation.

@author Jani Komppula
@date: 11/10/2017
"""

class Multiplication(object, metaclass=ABCMeta):
    """ An abstract class which multiplies the input signal by an array. The multiplier array is produced by taking
        a slice property (determined by the input parameter 'seed') and passing it through the abstract method
        multiplication_function(seed).
    """
    def __init__(self, seed, normalization = None, recalculate_multiplier = False, **kwargs):
        """
        :param seed: a seed for the multiplier, which can be 'bin_length', 'bin_midpoint', 'signal' or any slice
            property found from slice_set
        :param normalization: normalization of the multiplier
            'total_sum': The sum over the multiplier is equal to 1
            'segment_sum': The sum of the multiplier over each signal segment is equal to 1
            'total_average': The average of the multiplier is equal to 1
            'segment_average': The average multiplier of each signal segment is equal to 1
            'total_integral': The total integral over the multiplier is equal to 1
            'segment_integral': The integral of the multiplier over each signal segment is equal to 1
            'total_min': The minimum of the multiplier is equal to 1
            'segment_min': The minimum of the multiplier in each signal segment is equal to 1
            'total_max': The minimum of the multiplier is equal to 1
            'segment_max': The minimum of the multiplier in each signal segment is equal to 1
        :param recalculate_weight: if True, the weight is recalculated every time when process() is called
        """

        self._seed = seed
        self._normalization = normalization
        self._recalculate_multiplier = recalculate_multiplier

        self._multiplier = None

        self.signal_classes = (0,0)

        self.extensions = []
        self._macros = [] + default_macros(self, 'Multiplication', **kwargs)

        if self._seed not in ['bin_length','bin_midpoint','signal','ones']:
            self.extensions.append('bunch')
            self.required_variables = [self._seed]


    @abstractmethod
    def multiplication_function(self, seed):
        pass

    def process(self,parameters, signal, slice_sets = None, *args, **kwargs):

        if (self._multiplier is None) or self._recalculate_multiplier:
            self.__calculate_multiplier(parameters, signal, slice_sets)

        output_signal =  self._multiplier*signal

        # process the signal
        return parameters, output_signal

    def __calculate_multiplier(self,parameters, signal, slice_sets):
        self._multiplier = np.zeros(len(signal))

        if self._seed == 'ones':
            self._multiplier = self._multiplier + 1.
        elif self._seed == 'bin_width':
            np.copyto(self._multiplier, (parameters['bin_edges'][:,1]-parameters['bin_edges'][:,0]))
        elif self._seed == 'bin_midpoint':
            np.copyto(self._multiplier, ((parameters['bin_edges'][:,1]+parameters['bin_edges'][:,0])/2.))
        elif self._seed == 'normalized_bin_midpoint':

            for i in range(parameters['n_segments']):
                i_from = i * parameters['n_bins_per_segment']
                i_to = (i + 1) * parameters['n_bins_per_segment']

                np.copyto(self._multiplier[i_from:i_to], ((parameters['bin_edges'][i_from:i_to,1]+
                                                           parameters['bin_edges'][i_from:i_to,0])/2.
                                                          -parameters['segment_midpoints'][i]))

        elif self._seed == 'signal':
            np.copyto(self._multiplier,signal)
        else:
            if len(signal) == len(slice_sets) * (len(slice_sets[0].z_bins) - 1):
                start_idx = 0
                for slice_set in slice_sets:
                    seed = getattr(slice_set,self._seed)
                    np.copyto(self._multiplier[start_idx:(start_idx+len(seed))],seed)
                    start_idx += len(seed)
                np.copyto(self._multiplier, self._multiplier[::-1])
            else:
                raise ValueError('Signal length does not correspond to the original signal length '
                                 'from the slice sets in the method Multiplication')

        self._multiplier = self.multiplication_function(self._multiplier)

        # NOTE: add options for average bin integrals?
        if self._normalization is None:
            norm_coeff = 1.

        elif self._normalization == 'total_sum':
            norm_coeff = float(np.sum(self._multiplier))

        elif self._normalization == 'segment_sum':
            norm_coeff = np.ones(len(self._multiplier))
            for i in range(parameters['n_segments']):
                i_from = i*parameters['n_bins_per_segment']
                i_to = (i+1)*parameters['n_bins_per_segment']
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.sum(self._multiplier[i_from:i_to]))

        elif self._normalization == 'total_average':
            norm_coeff = float(np.sum(self._multiplier))/float(len(self._multiplier))

        elif self._normalization == 'segment_average':
            norm_coeff = np.ones(len(self._multiplier))
            for i in range(parameters['n_segments']):
                i_from = i*parameters['n_bins_per_segment']
                i_to = (i+1)*parameters['n_bins_per_segment']
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.sum(self._multiplier[i_from:i_to]))/float(parameters['n_bins_per_segment'])

        elif self._normalization == 'total_integral':
            bin_widths = parameters['bin_edges'][:,1] - parameters['bin_edges'][:,0]
            norm_coeff = np.sum(self._multiplier*bin_widths)

        elif self._normalization == 'segment_integral':
            bin_widths = parameters['bin_edges'][:,1] - parameters['bin_edges'][:,0]
            norm_coeff = np.ones(len(self._multiplier))
            for i in range(parameters['n_segments']):
                i_from = i*parameters['n_bins_per_segment']
                i_to = (i+1)*parameters['n_bins_per_segment']
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.sum(self._multiplier[i_from:i_to]*bin_widths[i_from:i_to]))

        elif self._normalization == 'total_min':
            norm_coeff = float(np.min(self._multiplier))

        elif self._normalization == 'segment_min':
            norm_coeff = np.ones(len(self._multiplier))
            for i in range(parameters['n_segments']):
                i_from = i*parameters['n_bins_per_segment']
                i_to = (i+1)*parameters['n_bins_per_segment']
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.min(self._multiplier[i_from:i_to]))

        elif self._normalization == 'total_max':
            norm_coeff = float(np.max(self._multiplier))

        elif self._normalization == 'segment_max':
            norm_coeff = np.ones(len(self._multiplier))
            for i in range(parameters['n_segments']):
                i_from = i*parameters['n_bins_per_segment']
                i_to = (i+1)*parameters['n_bins_per_segment']
                norm_coeff[i_from:i_to] = norm_coeff[i_from:i_to]*float(np.max(self._multiplier[i_from:i_to]))
        else:
            raise  ValueError('Unknown value in Multiplication._normalization')

        # TODO: try to figure out why this can not be written as
        # TODO:      self._multiplier /= norm_coeff
        self._multiplier =  self._multiplier / norm_coeff

    def clear(self):
        self._multiplier = None


class ChargeWeighter(Multiplication):
    """ The signal is weighted by charge (a number of macroparticles per slice)
    """

    def __init__(self, normalization = 'segment_max', **kwargs):
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
    """ Multiplies signal with a sine wave. Phase is locked to the midpoint of the each bunch shifted by the value of
     phase_shift [radians]"""
    def __init__(self,frequency,phase_shift, **kwargs):

        self._frequency = frequency
        self._phase_shift = phase_shift

        super(self.__class__, self).__init__('normalized_bin_midpoint', **kwargs)
        self.label = 'Signal mixer'

    def multiplication_function(self, seed):
        multiplier = np.sin(2.*pi*self._frequency*seed + self._phase_shift)
        return multiplier


class IdealAmplifier(Multiplication):
    """ An ideal amplifier/attenuator, which multiplies signal by a value of gain"""

    def __init__(self,gain, **kwargs):

        self._gain = gain

        super(self.__class__, self).__init__('ones', **kwargs)
        self.label = 'IdealAmplifier'

    def multiplication_function(self, seed):
        return seed * self._gain


class SegmentAverage(Multiplication):
    """An average of each signal segment is set to equal to 1. """
    def __init__(self,**kwargs):

        super(self.__class__, self).__init__('ones',normalization = 'segment_sum', **kwargs)
        self.label = 'SegmentAverage'

    def multiplication_function(self, seed):
        return seed

class MultiplicationFromFile(Multiplication):
    """ Multiplies the signal with an array, which is produced by interpolation from the external data file. Note the seed
        (unit) for the interpolation can be any of those available for the seed
        (i.e. location, sigma, or a number of macroparticles per slice, etc.)
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint', **kwargs):
        super(self.__class__, self).__init__(seed, **kwargs)
        self.label = 'Multiplication from file'

        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)

        if self._x_axis == 'time':
            pass
        elif self._x_axis == 'position':
            self._data[:, 0] = self._data[:, 0] / c
        else:
            raise ValueError('Unknown value in MultiplicationFromFile._x_axis')

    def multiplication_function(self, seed):
        return np.interp(seed, self._data[:, 0], self._data[:, 1])
