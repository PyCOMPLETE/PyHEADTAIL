# This file can be run by using the following command:
#$ mpirun -np 4 python 007_multibunch_ideal_feedback.py

"""
    This test is used for testing a bandwidth limited damper with multi turn wakes. The test is
    based on the code in the file '009_multibunch_bandwidth_limited_feedback.py'
"""

from __future__ import division

import sys, os
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)

import time
import numpy as np
import seaborn as sns
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p, pi

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
from PyHEADTAIL.impedances.wakes import ResistiveWall, CircularResistiveWall
from PyHEADTAIL.feedback.feedback import OneboxFeedback
from PyHEADTAIL.feedback.processors.multiplication import ChargeWeighter
from PyHEADTAIL.feedback.processors.convolution import Sinc, Lowpass, GaussianLowpass
from PyHEADTAIL.feedback.processors.misc import Bypass
from PyHEADTAIL.feedback.processors.resampling import ADC, DAC, UpSampler


plt.switch_backend('TkAgg')
sns.set_context('talk', font_scale=1.3)
sns.set_style('darkgrid', {
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'lines.markeredgewidth': 1})



def pick_signals(processor, source = 'input'):
    """
    A function which helps to visualize the signals passing the signal processors.
    :param processor: a reference to the signal processor
    :param source: source of the signal, i.e, 'input' or 'output' signal of the processor
    :return: (t, z, bins, signal), where 't' and 'z' are time or position values for the signal values (which can be used
        as x values for plotting), 'bins' are data for visualizing sampling and 'signal' is the actual signal.
    """

    if source == 'input':
        bin_edges = processor.input_parameters['bin_edges']
        raw_signal = processor.input_signal
    elif source == 'output':
        bin_edges = processor.output_parameters['bin_edges']
        raw_signal = processor.output_signal
    else:
        raise ValueError('Unknown value for the data source')

    z = np.zeros(len(raw_signal)*4)
    bins = np.zeros(len(raw_signal)*4)
    signal = np.zeros(len(raw_signal)*4)
    value = 1.

    for i, edges in enumerate(bin_edges):
        z[4*i] = edges[0]
        z[4*i+1] = edges[0]
        z[4*i+2] = edges[1]
        z[4*i+3] = edges[1]
        bins[4*i] = 0.
        bins[4*i+1] = value
        bins[4*i+2] = value
        bins[4*i+3] = 0.
        signal[4*i] = 0.
        signal[4*i+1] = raw_signal[i]
        signal[4*i+2] = raw_signal[i]
        signal[4*i+3] = 0.
        value *= -1

    t = z/c

    return (t, z, bins, signal)


def kicker(bunch):
    """
    A function which sets initial kicks for the bunches. The function is passed to the bunch generator
    in the machine object.
    """
#    bunch.x[:] += 1e-3
#    bunch.y[:] += 1e-3
    bunch.x *= 0
    bunch.xp *= 0
    bunch.y *= 0
    bunch.yp *= 0
    f = (1./20e6)*c
    bunch.x[:] += 0e-3 * np.sin(2.*pi*(np.mean(bunch.z)-bunch.z[0])*f)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# SIMULATION, BEAM AND MACHNINE PARAMETERS
# ========================================
n_turns = 20
n_segments = 1
n_macroparticles = 1000

from test_tools import MultibunchMachine
machine = MultibunchMachine(n_segments=n_segments)

intensity = 2.3e11
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081


# FILLING SCHEME
# ==============
# Bunches are created by creating a list of numbers representing the RF buckets to be filled.

n_bunches = 61
filling_scheme = [401 + 10*i for i in range(n_bunches)]

# Machine returns a super bunch, which contains particles from all of the bunches
# and can be split into separate bunches
bunches = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme)


# CREATE BEAM SLICERS
# ===================
slicer = UniformBinSlicer(50, n_sigma_z=3)
slicer_for_wakefields = UniformBinSlicer(20, z_cuts=(-0.4, 0.4))


# FEEDBACK MAP
# ==============
# Actual PyHEADTAIL map for the feedback system is created here. It is exactly same as presented for a single bunch
# in the file '001_ideal_feedbacks.ipynb'. Only difference is that 'mpi' flag is set into 'True' in OneboxFeedback
# object.
#
# Flags 'store_signal' are set into 'True' in the signal processors in order to visualize signal processing after the
# simulation, However, the flag does not affect the actual simulation.

fc=40e6
bunch_length = 2.49507468767912e-08/5.
bunch_spacing = 2.49507468767912e-08
f_ADC = 10./bunch_spacing

processors_x = [
#        Bypass(store_signal=True),
        ChargeWeighter(normalization='segment_average', store_signal=True),

        # It is recommended to resample the bunch in order to synchronize the slices
        # with bunch spacing, which helps the convolution. Parameters f_ADC and signal_length
        # should not affect significantly the reults, when signal_length is longer than
        # bunch_length and f_ADC is a couple of times higher than bunch frequency
        ADC(f_ADC, signal_length=0.5*bunch_spacing, store_signal=True),

        # It is recommended to use a gaussian lowpass filter. Sharp edges in impulse responses
        # are challenging from a simulation point of view (e.g. Lowpass and PhaseLinearizedLowpass)
        # and Sinc filter is too sensitive to cut off frequency, because oscillations in
        # the impulse response might be in resonance with the bunch spacing. Thus the gaussian
        # filter is the most stable solution.
        GaussianLowpass(fc, normalization=('bunch_by_bunch', bunch_length, bunch_spacing),
                        store_signal=True),
#        Lowpass(fc, normalization=('bunch_by_bunch', bunch_length, bunch_spacing),
#               store_signal=True),
#        Sinc(fc,normalization=('bunch_by_bunch', bunch_length,bunch_spacing),
#             store_signal=True),

        # DAC returs the signal to the original bin set.
        DAC(store_signal=True)
]
processors_y = [
#        Bypass(store_signal=True),
        ChargeWeighter(normalization='segment_average', store_signal=True),
        ADC(f_ADC, signal_length=0.5*bunch_spacing, store_signal=True),
        GaussianLowpass(fc, normalization=('bunch_by_bunch', bunch_length, bunch_spacing),
                        store_signal=True),
#        Lowpass(fc, normalization=('bunch_by_bunch', bunch_length, bunch_spacing),
#               store_signal=True),
#        Sinc(fc,normalization=('bunch_by_bunch', bunch_length,bunch_spacing),
#             store_signal=True),
        DAC(store_signal=True)
]
gain = 0.01
feedback_map = OneboxFeedback(gain, slicer, processors_x, processors_y, axis='displacement', mpi = True)

# The map is included directly into the total map in the machine.
a = machine.one_turn_map.pop()
a = machine.one_turn_map.pop()
a = machine.one_turn_map.pop()
machine.one_turn_map.append(feedback_map)
# WAKES
# =======
wakes = CircularResonator(1e7, 50e6, 50, n_turns_wake=10)
wake_field = WakeField(slicer_for_wakefields, wakes,
                       circumference=machine.circumference, mpi=True)


#wakes = CircularResistiveWall(pipe_radius=5e-2, resistive_wall_length=machine.circumference,
#                                    conductivity=1e6, dt_min=1e-3/c, mpi=True)
#wake_field = WakeField(slicer_for_wakefields, wakes)


w_function = wake_field.wake_kicks[0].wake_function
w_factor = wake_field.wake_kicks[0]._wake_factor
# The map is included directly into the total map in the machine.
machine.one_turn_map.append(wake_field)

# TRACKING LOOP
# =============
s_cnt = 0
monitorswitch = False

if rank == 0:

    import cProfile
    print '\n--> Begin tracking...\n'
    pr = cProfile.Profile()
    pr.enable()

for i in range(n_turns):

    if rank == 0:
        t0 = time.clock()
    machine.track(bunches)

    if rank == 0:
        t1 = time.clock()
        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))
if rank == 0:
    pr.disable()
    pr.print_stats(sort='time')


# VISUALIZATION
# =============
if rank == 0:
    # On the first processor, the script plots signals passed each signal processor from
    # the last simulated turn of the simulation

    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 14), sharex=False)

    for i, processor in enumerate(processors_x):
        t, z, bins, signal = pick_signals(processor,'output')
        ax1.plot(z, bins*(0.9**i), label =  processor.label)
        ax2.plot(z, signal, label =  processor.label)
#	if i == 0:
#		print z
#		print feedback_map._mpi_gatherer.total_data
#		print feedback_map._mpi_gatherer.total_data.z_bins


    # The first plot represents sampling in the each signal processor. The magnitudes of the curves do not represent
    # anything, but the change of the polarity represents a transition from one bin to another.
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_xlabel('Z position [m]')
    ax1.set_ylabel('Bin set')
    ax1.legend(loc='upper left')

    # Actual signals
    ax2.set_xlabel('Z position [m]')
    ax2.set_ylabel('Signal')
    ax2.legend(loc='upper left')

    plt.legend()
    plt.show()

