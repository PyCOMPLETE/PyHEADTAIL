from __future__ import division

import sys, os
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)

import time, copy
import numpy as np
from mpi4py import MPI
from scipy.constants import c, e, m_p
import matplotlib.pyplot as plt

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeTable, WakeField


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def generate_machine_and_bunches(n_bunches, n_macroparticles, intensity, chroma):

    # BEAM AND MACHNINE PARAMETERS
    # ============================
    from HLLHC import HLLHC
    machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                    machine_configuration='7_TeV_collision_tunes',
                    longitudinal_focusing='non-linear',
                    Qp_x=chroma, Qp_y=chroma, wrap_z=True,h_RF = 350,
                 circumference = 266.58883)
#    machine.circumference = 300.
    C = machine.circumference
    h = np.min(machine.longitudinal_map.harmonics) * 1.
    filling_scheme = sorted([20*i for i in range(n_bunches) for j in range(1)])

    # BEAM
    # ====
    epsn_x = 2.e-6
    epsn_y = 2.e-6
    sigma_z = 0.081
    allbunches = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
        filling_scheme=filling_scheme, matched=True)

#    bunch_list = allbunches.split()
#    for bunch in bunch_list:
#        bunch.x[:] = bunch.x[:] + (np.random.random() - 0.5)*1e-6
#
#    return machine, sum(bunch_list)
    return machine, allbunches

def track_n_turns(machine, bunches, mpi_settings, n_turns, n_slices):


    # CREATE BEAM SLICERS
    # ===================
    slicer_for_wakefields = UniformBinSlicer(n_slices, z_cuts=(-0.5, 0.5))
    # slicers_list = [UniformBinSlicer(20, z_cuts=(-0.4-f*C/h, 0.4-f*C/h)) for f in filling_scheme]


    # CREATE WAKES
    # ============
    if mpi_settings is not None:
        wakes = CircularResonator(1e12, 10e6, 10, n_turns_wake=10)
#        wakefile = '../interactive-tests/wake_table.dat'
#        wakes = WakeTable(wakefile,
#                          ['time', 'dipole_x', 'dipole_y', 'noquadrupole_x', 'noquadrupole_y',
#                         'nodipole_xy', 'nodipole_yx' ], n_turns_wake=10,
#                           circumference=machine.circumference)

        wake_field = WakeField(slicer_for_wakefields, wakes,
                               circumference=machine.circumference, h_bunch=int(machine.h_RF/10), mpi=mpi_settings)


        machine.one_turn_map.append(wake_field)

    bunch_list = bunches.split()

    track_time_data = np.zeros(n_turns)
    x_data = np.zeros((n_turns, len(bunch_list)))


    for i in range(n_turns):
#        if rank == 0:
        t0 = time.clock()

        machine.track(bunches)

#        if rank == 0:
        if i == 0:
            print 'mpi_settings = ' + str(mpi_settings) + ', n_bunches = ' + str(n_bunches)
        t1 = time.clock()
        track_time_data[i] = (t1-t0)*1e3
        bunch_list = bunches.split()
        for j, bunch in enumerate(bunch_list):
            x_data[i,j] = bunch.mean_x()


        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))

#    if rank == 0:
    return track_time_data, x_data

number_of_bunches = [8]
# number_of_bunches = [4,8,16]
chroma = 0
intensity = 2.3e11
n_turns = 1000
n_macroparticles = 200
n_slices = 10

output_filename = '2performance_data_resonator_n_slices_' + str(n_slices) + '.txt'

mpi_settings_for_testing = [
#    None,
    'dummy',
#    'memory_optimized',

    # if you want to compare accuratyly the loop minimized version to the other versions
    # please uncomment the line marked with 'UNCOMMENT THIS' (~line 370) in wake_kicks.py
    # because of the small rounding errors
#    'loop_minimized',

    'mpi_full_ring_fft',
#    True,
    ]

mpi_setting_labels = [
#    'without wake objects',
    'without wakes',
#    'memory_optimized',
#    'loop_minimized',
    'mpi_full_ring_fft',
#    'original',
    ]

data = []
data_x = []
for i, n_bunches in enumerate(number_of_bunches):
    data_x.append([])
    ref_machine, ref_bunches = generate_machine_and_bunches(n_bunches, n_macroparticles, intensity, chroma)
    for j, mpi_settings in enumerate(mpi_settings_for_testing):
        machine = copy.deepcopy(ref_machine)
        bunches = copy.deepcopy(ref_bunches)
        if i == 0:
            data.append(np.zeros((n_turns,len(number_of_bunches))))
        track_time_data, x_data = track_n_turns(machine, bunches, mpi_settings, n_turns, n_slices)
        data[j][:,i] = track_time_data
        data_x[-1].append(x_data)

ref_values = np.zeros(n_turns)
for i in xrange(n_turns):
    ref_values[i] = np.sum(np.abs(data_x[0][1][i,:]))

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(16, 9), sharex=True, tight_layout=False)
for i in xrange(len(data_x[0])):
    ax1.plot(data_x[0][i][:,0], label = mpi_setting_labels[i])
    ax2.plot(data_x[0][i][:,1])

    if i > 1:
        values = np.zeros(n_turns)
        for j in xrange(n_turns):
            values[j] = np.sum(np.abs(data_x[0][i][j,:]))

        ref_diff_1 = (values-ref_values)/ref_values
        ax3.plot(ref_diff_1, label = mpi_setting_labels[i])

ax1.legend()
ax3.legend()
ax1.set_ylabel('Bunch ' +str(int(rank*number_of_bunches[0]/size)) + ', mean_x')
ax2.set_ylabel('Bunch ' +str(int(rank*number_of_bunches[0]/size)+1) + ', mean_x')
ax3.set_ylabel('Relative total error')

plt.show()

# -*- coding: utf-8 -*-

