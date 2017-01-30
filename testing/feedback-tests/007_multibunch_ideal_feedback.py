# This file can be run by using the following command:
#$ mpirun -np 4 python 007_multibunch_ideal_feedback.py

"""
    This a simple example for a multi bunch MPI feedback. It is based on the ideal bunch feedback presented
    in the file 001_ideal_feedbacks.ipynb. The only difference is that multiple bunches are simulated in parallel
    in this example.
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
from PyHEADTAIL.feedback.feedback import OneboxFeedback
from PyHEADTAIL.feedback.processors.multiplication import ChargeWeighter
from PyHEADTAIL.feedback.processors.misc import Bypass

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
# Actual PyHEADTAIL map for the feedback system is created here. It is exactly same as presented for a single bunch
# in the file '001_ideal_feedbacks.ipynb'. Only difference is that 'mpi' flag is set into 'True' in OneboxFeedback
# object.
#
# Flags 'store_signal' are set into 'True' in the signal processors in order to visualize signal processing after the
# simulation, However, the flag does not affect the actual simulation.

processors_x = [
    Bypass(store_signal = True),
    ChargeWeighter(normalization = 'average',store_signal  = True),
]
processors_y = [
    Bypass(store_signal = True),
    ChargeWeighter(normalization = 'average',store_signal  = True),
]
gain = 0.1
feedback_map = OneboxFeedback(gain, slicer, processors_x, processors_y, axis='displacement', mpi = True)

# The map is included directly into the total map in the machine.
machine.one_turn_map.append(feedback_map)

# TRACKING LOOP
# =============
s_cnt = 0
monitorswitch = False

if rank == 0:
    print '\n--> Begin tracking...\n'

for i in range(n_turns):

    if rank == 0:
        t0 = time.clock()
    machine.track(bunches)

    if rank == 0:
        t1 = time.clock()
        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))

# VISUALIZATION
# =============
if rank == 0:
    # On the first processor, the script plots signals passed each signal processor from the last simulated turn

    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 14), sharex=False)

    for i, processor in enumerate(processors_x):
        t, z, bins, signal = pick_signals(processor,'output')
        ax1.plot(z, bins*(0.9**i), label =  processor.label)
        ax2.plot(z, signal, label =  processor.label)
	if i == 0:
		print z
		print feedback_map._mpi_gatherer.total_data
		print feedback_map._mpi_gatherer.total_data.z_bins

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

    plt.legend()
    plt.show()
