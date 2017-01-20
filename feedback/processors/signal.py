import collections

SignalParameters = collections.namedtuple('SignalParameters', ['signal_class','bin_edges','n_segments',
                                                               'n_bins_per_segment',
                                                               'original_segment_mids', 'beam_parameters'])

BeamParameters = collections.namedtuple('BeamParameters', ['phase_advance','beta_function'])