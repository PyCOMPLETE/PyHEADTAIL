from __future__ import division

#import sys, os
#BIN = os.path.expanduser("../../../")
#sys.path.append(BIN)

import time
import numpy as np
from mpi4py import MPI
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeTable, WakeField



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def track_n_turns(mpi_settings, n_turns, n_bunches, n_macroparticles, n_slices):
    chroma = 0
    intensity = 2.3e11
    n_macroparticles = 20000


    # BEAM AND MACHNINE PARAMETERS
    # ============================
    from HLLHC import HLLHC
    machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                    machine_configuration='7_TeV_collision_tunes',
                    longitudinal_focusing='non-linear',
                    Qp_x=chroma, Qp_y=chroma, wrap_z=True)
    C = machine.circumference
    h = np.min(machine.longitudinal_map.harmonics) * 1.
    filling_scheme = sorted([5 + 10*i for i in range(n_bunches) for j in range(1)])

    # BEAM
    # ====
    epsn_x = 2.e-6
    epsn_y = 2.e-6
    sigma_z = 0.081
    allbunches = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
        filling_scheme=filling_scheme, matched=True)

    # CREATE BEAM SLICERS
    # ===================
    slicer_for_wakefields = UniformBinSlicer(n_slices, z_cuts=(-0.4, 0.4))
    # slicers_list = [UniformBinSlicer(20, z_cuts=(-0.4-f*C/h, 0.4-f*C/h)) for f in filling_scheme]


    # CREATE WAKES
    # ============
    if mpi_settings is not None:
        wakes = CircularResonator(1e6, 20e6, 10, n_turns_wake=10)
#        wakefile = '../interactive-tests/wake_table.dat'
#        wakes = WakeTable(wakefile,
#                          ['time', 'dipole_x', 'dipole_y', 'noquadrupole_x', 'noquadrupole_y',
#                         'nodipole_xy', 'nodipole_yx' ], n_turns_wake=10,
#                           circumference=machine.circumference)

        wake_field = WakeField(slicer_for_wakefields, wakes,
                               circumference=machine.circumference, mpi=mpi_settings)


        machine.one_turn_map.append(wake_field)

    track_time_data = np.zeros(n_turns)


    for i in range(n_turns):
        if rank == 0:
            t0 = time.clock()

        machine.track(allbunches)

        if rank == 0:
            if i == 0:
                print 'mpi_settings = ' + str(mpi_settings) + ', n_bunches = ' + str(n_bunches)
            t1 = time.clock()
            track_time_data[i] = (t1-t0)*1e3

            print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
                "%d/%m/%Y %H:%M:%S", time.localtime())))

    if rank == 0:
        return track_time_data

number_of_bunches = [8, 16, 32, 64, 128]
# number_of_bunches = [4,8,16]
n_turns = 20
n_macroparticles = 200
n_slices = 10

output_filename = 'performance_data_resonator_n_slices_10.txt'

mpi_settings_for_testing = [
    None,
    True,
    'optimized',
    ]

mpi_setting_labels = [
    'without wakes',
    'original',
    'optimized',
    ]

data = []

for i, n_bunches in enumerate(number_of_bunches):
    for j, mpi_settings in enumerate(mpi_settings_for_testing):
        if i == 0:
            data.append(np.zeros((n_turns,len(number_of_bunches))))
        data[j][:,i] = track_n_turns(mpi_settings, n_turns, n_bunches, n_macroparticles, n_slices)

if rank == 0:
    row_labels = np.linspace(1,n_turns,n_turns)
    with open(output_filename, "w") as text_file:
        text_file.write('n_turns: ' + str(n_turns) + '\n')
        text_file.write('n_macroparticles per bunch: ' + str(n_macroparticles) + '\n')
        text_file.write('n_slices per bunch: ' + str(n_slices) + '\n')
        text_file.write('number_of_bunches: ' + str(number_of_bunches) + '\n')

        for i in xrange(len(mpi_settings_for_testing)):
            text_file.write('' + '\n')
            text_file.write('' + '\n')
            text_file.write('     Turn by turn tracking time [ms], ' + mpi_setting_labels[i]  + '\n')
            text_file.write('%s  %s\n' % ('   ', ' '.join('%05s' % i for i in number_of_bunches)))
            for row_label, row in zip(row_labels, data[i]):
                text_file.write('%s [%s]\n' % ('% 3.0f' %row_label, ' '.join('% 5.0f' %i  for i in row)))
