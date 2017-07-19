import numpy as np
import copy

""" This file contains the core functions and variables for signal processing.
"""

# TODO: change the base unit from distance to time


version = '0.2_b2'

def Parameters(signal_class=0, bin_edges=np.array([]), n_segments=0,
               n_bins_per_segment=0, segment_ref_points=np.array([]),
               previous_parameters = [], location=0, beta=1.):
    """
    Returns a prototype for signal parameters.

    Parameters
    ----------
    class : int
        A signal class
    bin_edges : NumPy array
        A 2D numpy array, which is equal length to the signal. Each row
        includes two floating point numbers, the edge positions of
        the bin in the physical space.
    n_segments : int
        A number of equal length and equally binned segments where to
        the signal can be divided
    n_bins_per_segment : int
        A number of bins per segment. `len(bin_edges)/n_segments`
    segment_ref_points : NumPy array
        A numpy array of the reference point for the segments
    previous_parameters : array
        A list of Parameters, which tracks how the samping has been changed
        during the signal processing
    location : float
        A location of the signal in betatron phase.
    beta : float
        A vale of beta function in the source of the signal. Value 1
        is neutral for signal processing
    """
    return {'class': signal_class,
            'bin_edges': bin_edges,
            'n_segments': n_segments,
            'n_bins_per_segment': n_bins_per_segment,
            'segment_ref_points': segment_ref_points,
            'previous_parameters': previous_parameters,
            'location': location,
            'beta': beta
            }


def Signal(signal=[]):
    """ Returns a prototype for a signal."""
    return np.array(signal)


def process(parameters, signal, processors, **kwargs):
    """
    Processes a signal through the given signal processors

    Parameters
    ----------
    parameters : dict
        A standardized dict of the additional parameters describing the signal
    signal : NumPy array
        The signal
    processors : list
        A list of signal processors.
    **kwargs : -
        Other arguments which will be passed to the signal processors

    Returns
    -------
    dict
        Possibly modified dict of the signal parameters
    NumPy array
        The processed signal
    """

    for processor in processors:
        parameters, signal = processor.process(parameters, signal, **kwargs)
#        if signal is None:
#            print 'None signal!'
#            break

    return parameters, signal


def bin_widths(bin_edges):
    return (bin_edges[:, 1]-bin_edges[:, 0])


def bin_mids(bin_edges):
    return (bin_edges[:, 0]+bin_edges[:, 1])/2.


def bin_edges_to_z_bins(bin_edges):
    return np.append(bin_edges[:, 0], bin_edges[-1, 1])


def z_bins_to_bin_edges(z_bins):
    return np.transpose(np.array([z_bins[:-1], z_bins[1:]]))


def append_bin_edges(bin_edges_1, bin_edges_2):
    return np.concatenate((bin_edges_1, bin_edges_2), axis=0)


def get_processor_extensions(processors, external_extensions=None):
    """
    A function, which checks available extensions from the processors.

    Parameters
    ----------
    processors : list
        A list of signal processors.
    external_extensions : list
        A list of external extensions, which will be added to the list

    Returns
    -------
    list
        A list of found extensions
    """

    if external_extensions is None:
        available_extensions = []
    else:
        available_extensions = external_extensions

    for processor in processors:
        if processor.extensions is not None:
            available_extensions.extend(processor.extensions)

    available_extensions = list(set(available_extensions))

    return available_extensions


#--- Extension specific functions ------
#---------------------------------------

def get_processor_variables(processors, required_variables=None):
    """
    A function which checks the required PyHEADTAIL slice set variables
    from the signal processors.

    Parameters
    ----------
    processors : list
        A list of signal processors.
    external_variables : list
        A list of external variables, which will be added to the list

    Returns
    -------
    list
        A list of found statistical variables
    """

    if required_variables is None:
        required_variables = []

    for processor in processors:
        if 'bunch' in processor.extensions:
            required_variables.extend(processor.required_variables)

    required_variables = list(set(required_variables))

    if 'z_bins' in required_variables:
        required_variables.remove('z_bins')

    return required_variables


def debug_extension(target_object, label=None, **kwargs):
    """
    A debug extension, which can be added to the extension object list in the
    signal processors. If input parameter debug = True is given to the signal
    processor, the input and output parameters and signals are stored to
    the signal processor.

    Parameters
    ----------
    processors : list
        A list of signal processors.
    external_variables : list
        A list of external variables, which will be added to the list

    Returns
    -------
    list
        A list of found statistical variables
    """
    if 'debug' in kwargs:
        setattr(target_object, 'debug', kwargs['debug'])
    else:
        setattr(target_object, 'debug', False)
    setattr(target_object, 'label', label)
    setattr(target_object, 'input_parameters', None)
    setattr(target_object, 'input_signal', None)
    setattr(target_object, 'output_parameters', None)
    setattr(target_object, 'output_signal', None)

    def store_data(target_object, input_parameters, input_signal,
                   output_parameters, output_signal, *args, **kwargs):
        if target_object.debug:
            target_object.input_parameters = copy.copy(input_parameters)
            target_object.input_signal = np.copy(input_signal)
            target_object.output_parameters = copy.copy(output_parameters)
            target_object.output_signal = np.copy(output_signal)

    return store_data
