from __future__ import division

import h5py
import matplotlib
import pylab as plt
import numpy as np

#--------------------------------------------------------------------------------------------------
# Try to understand numpy.fft
#print('easy fft')

# Sampling frequency [Hz] and Nyquist frequency
#freq_s = 10
#freq_N = freq_s/2.

# Number of samples to be used.
#n_s    = 2048

# Determines time range
#t = np.arange(0., n_s/freq_s, 1./freq_s)

# Signal frequency [Hz]
#freq = 0.8

#signal = np.sin(2.*np.pi*freq*t)
#fft_y  = np.fft.rfft(signal)

# Frequency scale (for Fourier space)
#f = np.fft.rfftfreq(n_s, d=1./freq_s)

#fig2, ( ax21, ax22 ) = plt.subplots(2)
#ax21.plot(t, signal)
#ax22.plot(f, np.absolute(fft_y))

#print(signal.shape)
#print(fft_y.shape)

#plt.show()
