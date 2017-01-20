# This file can be run by using the following command:
#$ mpirun -np 4 009_multibunch_separated_pickup_and_kicker_resampling.py

"""
    This an extended version of the example presented in the file python 008_multibunch_separated_pickup_and_kicker.py.
    It demonstrates the use of resamplers, which allow simulations of bandwidth limited bunch by bunch feedback systems.
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
from PyHEADTAIL.feedback.feedback import Kicker, PickUp
from PyHEADTAIL.feedback.processors.multiplication import ChargeWeighter
from PyHEADTAIL.feedback.processors.register import HilbertPhaseShiftRegister
from PyHEADTAIL.feedback.processors.signal import BeamParameters
from PyHEADTAIL.feedback.processors.convolution import Sinc
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
        bin_edges = processor.input_signal_parameters.bin_edges
        raw_signal = processor.input_signal
    elif source == 'output':
        bin_edges = processor.output_signal_parameters.bin_edges
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
    bunch.x *= 0
    bunch.xp *= 0
    bunch.y *= 0
    bunch.yp *= 0
    bunch.x[:] += 2e-2 * np.sin(2.*pi*np.mean(bunch.z)/1000.)


# MPI objects, which are used for visualizing the data only on the first processor
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# SIMULATION, BEAM AND MACHNINE PARAMETERS
# ========================================
n_turns = 100
n_segments = 1
n_macroparticles = 40000

from test_tools import MultibunchMachine
machine = MultibunchMachine(n_segments=n_segments)

intensity = 2.3e11
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081


# FILLING SCHEME
# ==============
# Bunches are created by creating a list of numbers which represent bunches in the RF buckets.
# In the other words, this means that the length of the list corresponds to the number of bunches are simulated and
# the numbers in the list correspond to the locations of the bunches in the machine.

n_bunches = 13
filling_scheme = [401 + 20*i for i in range(n_bunches)]

# multiple bunches are created by passing the filling scheme to the generator. It returns a super bunch, which contains
# particles from the all of the bunches, but can be split into separated bunches
bunches = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme, kicker=kicker)


# CREATE BEAM SLICERS
# ===================
slicer = UniformBinSlicer(50, n_sigma_z=3)


# FEEDBACK MAP
# ==============
# This uses the code from the previous example ('008_multibunch_separated_pickup_and_kicker.py'). Only signal
# processors are different.

delay = 1
n_values = 3

bunch_spacing = 2.49507468767912e-08
f_ADC = 1./(bunch_spacing)
signal_length = 2.*bunch_spacing

f_c = 50e6

pickup_processors_x = [
    ChargeWeighter(normalization = 'average',store_signal  = True),
    # In this example a sample per bunch sampling rate is used for the ADC
    ADC(f_ADC, n_bits = 8, input_range = (-1e-3,1e-3), signal_length = signal_length,store_signal  = True),
    HilbertPhaseShiftRegister(n_values, machine.accQ_x, delay,store_signal  = True)
]
pickup_processors_y = [
    ChargeWeighter(normalization = 'average',store_signal  = True),
    ADC(f_ADC, n_bits = 8, input_range = (-1e-3,1e-3), signal_length = signal_length,store_signal  = True),
    HilbertPhaseShiftRegister(n_values, machine.accQ_x, delay,store_signal  = True)
]


pickup_beam_parameters_x = BeamParameters(1.*2.*pi/float(n_segments)*machine.accQ_x,machine.beta_x_inj)
pickup_beam_parameters_y = BeamParameters(1.*2.*pi/float(n_segments)*machine.accQ_y,machine.beta_y_inj)

pickup_map = PickUp(slicer,pickup_processors_x,pickup_processors_y,
       pickup_beam_parameters_x, pickup_beam_parameters_y, mpi = True)

registers_x = [pickup_processors_x[2]]
registers_y = [pickup_processors_y[2]]

kicker_processors_x = [
    # multiplies the sampling rate by a factor of three by adding zeros values between the samples
    UpSampler(3,kernel=[0,1,0],store_signal  = True),
    Sinc(1*f_c,store_signal  = True),
    # returns to the bin set of the original slice set. The values for the slices are determined by using spline interpolation
    DAC(store_signal  = True)
]
kicker_processors_y = [
    UpSampler(3,kernel=[0,1,0],store_signal  = True),
    Sinc(1*f_c,store_signal  = True),
    DAC(store_signal  = True)
]

kicker_beam_parameters_x = BeamParameters(2.*2.*pi/float(n_segments)*machine.accQ_x,machine.beta_x_inj)
kicker_beam_parameters_y = BeamParameters(2.*2.*pi/float(n_segments)*machine.accQ_y,machine.beta_y_inj)

gain = 0.1

kicker_map = Kicker(gain, slicer, kicker_processors_x, kicker_processors_y,
                    registers_x, registers_y, kicker_beam_parameters_x, kicker_beam_parameters_y, mpi = True)


# The one turn map of the machine is reconstructed and the kicker and the pickup are placed into the correct slots.
# The one turn map is created machine in suck a way that the first elements of the map are transverse elements
# (but this might vary!).

new_one_turn_map = []
for i, m in enumerate(machine.one_turn_map):

    if i == 1:
        new_one_turn_map.append(pickup_map)

    if i == 2:
        new_one_turn_map.append(kicker_map)

    new_one_turn_map.append(m)

machine.one_turn_map = new_one_turn_map

# TRACKING LOOP
# =============
s_cnt = 0
monitorswitch = False
if rank == 0:
    print '\n--> Begin tracking...\n'

print 'Tracking'
for i in range(n_turns):

    if rank == 0:
        t0 = time.clock()
    machine.track(bunches)

    if rank == 0:
        t1 = time.clock()
        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))

if rank == 0:
    # On the first processor, the script plots signals passed each signal processor from the last simulated turn

    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 14), sharex=False)
    fig.suptitle('Pickup processors', fontsize=20)

    for i, processor in enumerate(pickup_processors_x):
        t, z, bins, signal = pick_signals(processor,'output')
        ax1.plot(z, bins*(0.9**i), label =  processor.label)
        ax2.plot(z, signal, label =  processor.label)

    # The first plot represents sampling in the each signal processor. The magnitudes of the curves do not represent
    # anything, but the change of the polarity represents a transition from one bin to other.
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_xlabel('Z position [m]')
    ax1.set_ylabel('Bin set')
    ax1.legend(loc='upper left')

    # Actual signals are plotted in this figure
    ax2.set_xlabel('Z position [m]')
    ax2.set_ylabel('Signal')
    ax2.legend(loc='upper left')

    fig, (ax3, ax4) = plt.subplots(2, figsize=(14, 14), sharex=False)
    fig.suptitle('Kicker processors', fontsize=20)

    for i, processor in enumerate(kicker_processors_x):
        t, z, bins, signal = pick_signals(processor,'output')
        ax3.plot(z, bins*(0.9**i), label =  processor.label)
        ax4.plot(z, signal, label =  processor.label)

    # The first plot represents sampling in the each signal processor. The magnitudes of the curves do not represent
    # anything, but the change of the polarity represents a transition from one bin to other.
    ax3.set_ylim([-1.1, 1.1])
    ax3.set_xlabel('Z position [m]')
    ax3.set_ylabel('Bin set')
    ax3.legend(loc='upper left')

    # Actual signals are plotted in this figure
    ax4.set_xlabel('Z position [m]')
    ax4.set_ylabel('Signal')
    ax4.legend(loc='upper left')

    plt.show()
