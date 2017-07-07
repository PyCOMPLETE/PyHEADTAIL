import copy
import numpy as np
from ..core import debug_extension

class Bypass(object):
    """ A fast bypass processor, whichi does not modify the signal. A black sheep, which does not fit for
        the abstract classes.
    """

    def __init__(self, **kwargs):
        self.signal_classes = (0, 0)

        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, 'Bypass', **kwargs)]

    def process(self, parameters, signal, *args, **kwargs):

        for extension in self._extension_objects:
            extension(self, parameters, signal, parameters, signal,
                      *args, **kwargs)


        return parameters, signal


class Average(object):

    def __init__(self, avg_type = 'bunch', **kwargs):
        self.label = 'Average'
        self._avg_type = avg_type


        self.signal_classes = (0, 0)
        self.extensions = ['debug']
        self._extension_objects = [debug_extension(self, 'Average', **kwargs)]


    def process(self, parameters, signal, *args, **kwargs):

        if self._avg_type == 'bunch':
            n_segments = signal_parameters.n_segments
            n_slices_per_segment = signal_parameters.n_slices_per_segment

            output_signal = np.zeros(len(signal))
            ones = np.ones(n_slices_per_segment)

            for i in xrange(n_segments):
                idx_from = i * n_slices_per_segment
                idx_to = (i + 1) * n_slices_per_segment
                np.copyto(output_signal[idx_from:idx_to], ones * np.mean(signal[idx_from:idx_to]))

        elif self._avg_type == 'total':
            output_signal = np.ones(len(signal))*np.mean(signal)

        else:
            raise ValueError('Unknown value in Average._avg_type')

        for extension in self._extension_objects:
            extension(self, parameters, signal, parameters, output_signal,
                      *args, **kwargs)


        return parameters, output_signal