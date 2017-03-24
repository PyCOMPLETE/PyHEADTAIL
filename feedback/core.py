import numpy as np

""" This file contains the core functions and variables for signal processing.
"""

# TODO: automated Debug extension
# TODO: change the base unit from distance to time


def Parameters(signal_class=0, bin_edges=np.array([]), n_segments=0,
               n_bins_per_segment=0, segment_midpoints=np.array([]),
               location=0, beta=1.):
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
    segment_midpoints : NumPy array
        A numpy array of original midpoints of the segments
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
            'segment_midpoints': segment_midpoints,
            'location': location,
            'beta': beta
            }


def Signal(signal=[]):
    """ Returns a prototype for a signal."""
    return np.array(signal)


def process(parameters, signal, processors, **kwargs):
    """
    Returns a prototype for signal parameters.

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

    return parameters, signal


def get_processor_extensions(processors, available_extensions=None):
    """
    A function, which checks available extensions from the processors.

    Parameters
    ----------
    processors : list
        A list of signal processors.
    available_extensions : list
        A list of external extensions, which will be added to the list

    Returns
    -------
    list
        A list of found extensions
    """

    if available_extensions is None:
        available_extensions = []

    for processor in processors:
        if processor.extensions is not None:
            available_extensions.extend(processor.extensions)

    available_extensions = list(set(available_extensions))

    return available_extensions


# Extension specific functions
#########################

def get_processor_variables(processors, required_variables=None):
    """
    A function which checks the required PyHEADTAIL slice set variables
    from the signal processors.

    Parameters
    ----------
    processors : list
        A list of signal processors.
    required_variables : list
        A list of external extensions, which will be added to the list

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
