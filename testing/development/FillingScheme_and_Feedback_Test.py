# run this file by using command:
#$ mpirun -np 4 python FillingScheme_and_Feedback_Test.py


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
from PyHEADTAIL.feedback.processors import Sinc, Lowpass,Bypass

def kicker(bunch):
    bunch.x *= 0
    bunch.xp *= 0
    bunch.y *= 0
    bunch.yp *= 0
    bunch.x[:] += 2e-2 * np.sin(2.*pi*np.mean(bunch.z)/1000.)

plt.switch_backend('TkAgg')
sns.set_context('talk', font_scale=1.3)
sns.set_style('darkgrid', {
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'lines.markeredgewidth': 1})


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_turns = 100
chroma = 0

n_bunches = 13
filling_scheme = [401 + 200*i for i in range(n_bunches)]
n_macroparticles = 40000
intensity = 2.3e11


# BEAM AND MACHNINE PARAMETERS
# ============================
from HLLHC import HLLHC
machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                machine_configuration='7_TeV_collision_tunes',
                longitudinal_focusing='non-linear',
                Qp_x=chroma, Qp_y=chroma, wrap_z=True)
# C = machine.circumference
# h = np.min(machine.longitudinal_map.harmonics) * 1.
# z = -1.*(np.array(filling_scheme))/h*C


'''extract_bunches is exteremely expensive so you would want to do this exactly
once (if, at all, since anyway, you are generating bunches)! In this case, we
might still have to live with the wake track method able to accept a bunches
list - you won't want it to extract bunches at every turn!

Try another method implementing a bunches.split method. The, we can keep all
track methods generic acting on a particles object.

'''


# Distribute among processors
# ===========================
# size = 4
# n_bunches_per_proc = n_bunches//size
# n_bunches_remainder = n_bunches % size
# bunches_on_proc_list = [filling_scheme[i:i+n_bunches_per_proc+1]
#                         if i < n_bunches_remainder
#                         else filling_scheme[i+1:i+n_bunches_per_proc+1]
#                         for i in range(0, n_bunches, n_bunches_per_proc)
#                         if filling_scheme[i+1:i+n_bunches_per_proc+1]]
# print("\n\n*** Summary of bunches on processors {:s}".format(
#     bunches_on_proc_list))


# BEAM
# ====
# allbunches = []
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081

bunches = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme, kicker=kicker)

print 'Bunches are ready'

# bunches = [machine.generate_6D_Gaussian_bunch_matched(
#     n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
#     bucket=bucket) for bucket in bunches_on_proc_list[rank]]
# for b in bunches:
#     b.x *= 0
#     b.xp *= 0
#     b.y *= 0
#     b.yp *= 0
#     b.x[:] += 2e-2 * np.sin(2.*pi*np.mean(b.z)/1000.)
# bunches = sum(bunches)


# bunch_tmp = machine.generate_6D_Gaussian_bunch_matched(
#     n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
#     bunch_id=rank+10)
# print "\n--> Buch length and emittance: {:g} m, {:g} eVs.".format(
#     bunch.sigma_z(), bunch.epsn_z())
# bunch_tmp.z += z[rank+1]
# bunch_tmp.dt = z[rank+1]/c

# bunch += bunch_tmp
# print bunch.bunch_id


# CREATE BEAM SLICERS
# ===================
slicer_for_diagnostics = UniformBinSlicer(50000, n_sigma_z=3)
slicer_for_wakefields = UniformBinSlicer(50, z_cuts=(-0.2, 0.2))

f_c = 100e6

gain = 0.3

processors_x = [
    # Lowpass(f_c)
    # Sinc(f_c, bunch_spacing=25e-9)
    Lowpass(f_c, bunch_spacing=25e-9)
    # Bypass()
]
processors_y = [
    # Lowpass(f_c)
    # Sinc(f_c, bunch_spacing=25e-9)
    Lowpass(f_c, bunch_spacing=25e-9)
    # Bypass()
]
feedback_map = OneboxFeedback(gain, slicer_for_wakefields, processors_x, processors_y, axis='displacement', mpi = True,extra_statistics = ['mean_xp','mean_yp'])


# statistics = ['mean_x','mean_y','mean_z']
# maps = [
#     feedback_map
# ]
# mpi_map = MPI_map(maps,slicer,bunch_offsets,statistics)





# TRACKING LOOP
# =============
machine.one_turn_map.append(feedback_map)
# machine.one_turn_map.append(wake_field)

s_cnt = 0
monitorswitch = False
if rank == 0:
    print '\n--> Begin tracking...\n'

print 'Tracking'
for i in range(n_turns):

    if rank == 0:
        t0 = time.clock()
    # bunchmonitor.dump(bunch)

    # if not monitorswitch:
    #     if (bunch.mean_x() > 1e3 or bunch.mean_y() > 1e3 or
    #             i > n_turns - 2048):
    #         print "--> Activate monitor"
    #         monitorswitch = True
    # else:
    #     if s_cnt < 2048:
    #         slicemonitor.dump(bunch)
    #         s_cnt += 1

    machine.track(bunches)

    if rank == 0:
        t1 = time.clock()
        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))

if rank == 0:



    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(14, 14), sharex=False)
    # ax1.plot(bunch_offsets, init_positions, 'ro')

    # this gets data from local variables... but for debugging

    ax1.set_title('Cut-off frequency: ' + str(f_c/1e6) + ' MHz')
    ax1.plot(feedback_map._mpi_gatherer.total_data.mean_z, feedback_map._mpi_gatherer.total_data.mean_x * 1e3, 'b.')
    ax1.set_xlabel('Z position [m]')
    ax1.set_ylabel('Mean_x [mm]')

    ax2.plot(feedback_map._mpi_gatherer.total_data.mean_z, feedback_map._mpi_gatherer.total_data.mean_xp * 1e6, 'r.')
    ax2.set_xlabel('Z position [m]')
    ax2.set_ylabel('Mean_xp [mm mrad]')

    ax3.plot(processors_x[0]._CDF_time,processors_x[0]._PDF,'r-')
    ax3.set_xlabel('Z position [m]')
    ax3.set_ylabel('Mean_xp [mm mrad]')

    ax4.plot(processors_x[0]._CDF_time,processors_x[0]._CDF_value,'r-')
    ax4.set_xlabel('Z position [m]')
    ax4.set_ylabel('Mean_xp [mm mrad]')
    plt.show()
