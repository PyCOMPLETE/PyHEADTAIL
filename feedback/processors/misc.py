import copy
import numpy as np

class Bypass(object):
    """ A fast bypass processor, whichi does not modify the signal. A black sheep, which does not fit for
        the abstract classes.
    """

    def __init__(self, store_signal = False):
        self.signal_classes = (0, 0)
        self.extensions = ['store']

        self.label = 'Bypass'
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    def process(self,parameters, signal, *args, **kwargs):

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(signal)
            self.output_parameters = copy.copy(parameters)


        return parameters, signal


class Average(object):

    def __init__(self, avg_type = 'bunch', store_signal = False):
        self.label = 'Average'
        self._avg_type = avg_type


        self.signal_classes = (0, 0)
        self.extensions = ['store']

        self._store_signal = store_signal
        self.input_signal = None
        self.input_signal_parameters = None
        self.output_signal = None
        self.output_signal_parameters = None


    def process(self,signal_parameters, signal, *args, **kwargs):

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

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_signal_parameters = copy.copy(signal_parameters)
            self.output_signal = np.copy(output_signal)
            self.output_signal_parameters = copy.copy(signal_parameters)


        return signal_parameters, output_signal