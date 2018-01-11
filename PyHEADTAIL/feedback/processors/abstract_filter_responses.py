import numpy as np
import scipy.special as special

""" This file contains dimensionless impulse responses function for different
analog filters, which can be used in different signal processor implementations
(e.g. based on convolution and linear transformation).

@author Jani Komppula
@date: 11/10/2017
"""

def normalized_lowpass(max_impulse_length):

    def response_function(x):
        if x < 0.:
            return 0.
        elif x > max_impulse_length:
            return 0.
        else:
            return np.exp(-1. * x)

    return response_function

def normalized_highpass(max_impulse_length):

    def response_function(x):
        if x < 0.:
            return 0.
        elif x > max_impulse_length:
            return 0.
        else:
            return np.exp(-1. * x)

    return response_function

def normalized_phase_linearized_lowpass(max_impulse_length):
    """Phase linearized version of the normal lowpass (RC,
    singe poll roll-off) filter. Formula derived by Gerd Kotzian."""

    def response_function(x):
        if x == 0.:
            return 0.
        elif x < -max_impulse_length:
            return 0.
        elif x > max_impulse_length:
            return 0.
        else:
            return special.k0(abs(x))

    return response_function

def normalized_Gaussian(max_impulse_length):

    def response_function(x):
        if x < -max_impulse_length:
            return 0.
        elif x > max_impulse_length:
            return 0.
        else:
            return np.exp(-x ** 2. / 2.* (2.*np.log(2))) / np.sqrt(2. * np.pi)

    return response_function

def normalized_sinc(window_type, window_width):

    def blackman_window(x):
        return 0.42-0.5*np.cos(2.*np.pi*(x/np.pi+window_width)/(2.*window_width))\
               +0.08*np.cos(4.*np.pi*(x/np.pi+window_width)/(2.*window_width))

    def hamming_window(x):
        return 0.54-0.46*np.cos(2.*np.pi*(x/np.pi+window_width)/(2.*window_width))

    def response_function(x):
        if np.abs(x/np.pi) > window_width:
            return 0.
        else:
            if window_type == 'blackman':
                return np.sinc(x/np.pi)*blackman_window(x)
            elif window_type == 'hamming':
                return np.sinc(x/np.pi)*hamming_window(x)

    return response_function