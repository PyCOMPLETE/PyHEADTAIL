from __future__ import division

import time
import numpy as np
import seaborn as sns
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p
import h5py

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
from PyHEADTAIL.monitors.monitors import BunchMonitor


plt.switch_backend('TkAgg')
sns.set_context('talk', font_scale=1.3)
sns.set_style('darkgrid', {
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'lines.markeredgewidth': 1,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Avant Garde']})
    # 'text.usetex': True})


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_turns = 100
chroma = 0
n_bunches = 7
intensity = 2.3e11
n_macroparticles = 200000


# BEAM AND MACHNINE PARAMETERS
# ============================
from HLLHC import HLLHC
machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                circumference=70, h_RF=[40],
                machine_configuration='7_TeV_collision_tunes',
                longitudinal_focusing='non-linear',
                Qp_x=chroma, Qp_y=chroma, wrap_z=True)
C = machine.circumference
h = np.min(machine.longitudinal_map.harmonics) * 1.
filling_scheme = sorted([5 + 2*i + h*j for i in range(n_bunches) for j in range(1)])


# BEAM
# ====
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081
allbunches = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme)


# CREATE BEAM SLICERS
# ===================
slicer_for_diagnostics = UniformBinSlicer(10000/2.*n_turns, z_cuts=(-C*n_turns, 0))
slicer_for_wakefields = UniformBinSlicer(20, z_cuts=(-0.4, 0.4))


# CREATE WAKES
# ============
wakes = CircularResonator(1e6, 20e6, 10, n_turns_wake=10)
wake_field = WakeField(slicer_for_wakefields, wakes,
                       circumference=machine.circumference, mpi=True)
w_function = wake_field.wake_kicks[0].wake_function
w_factor = wake_field.wake_kicks[0]._wake_factor

# TRACKING LOOP
# =============
# machine.one_turn_map.append(damper)
machine.one_turn_map.append(wake_field)

#bunchmonitor = BunchMonitor('bunchmon_test_parallel', n_turns)
#bunchmonitor = h5py.File('bunchmon_test_parallel.h5', 'w', driver='mpio', comm=comm)
bunchmonitor = BunchMonitor(
    'bunchmon_test_parallel', n_turns, mpi=True, filling_scheme=filling_scheme)

# According to docs, every rank must act in the same manner on the file structure!
# This seems to be true even if a specific rank would always write only to one subgroup of
# the file.
#stats_to_store = [
#    'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
#    'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
#    'epsn_z', 'macroparticlenumber' ]
#h5mainGroup = bunchmonitor.create_group('Bunches')
#for bid in np.int_(filling_scheme): # Shall we use as bucket IDs the integer part only? Otherwise it's a bit weird.
#    h5group = h5mainGroup.create_group(repr(bid))
#    for stats in sorted(stats_to_store):
#        h5group.create_dataset(stats, shape=(n_turns,)) #,
#            compression='gzip', compression_opts=9) According to Andrew Collette, compression opts not available
# when using h5py parallel... Possible compression at the end, with h5repack command line tool.

s_cnt = 0
if rank == 0:
    print '\n--> Begin tracking...\n'
for i in range(n_turns):

    if rank == 0:
        t0 = time.clock()

    machine.track(allbunches)
    bunchmonitor.dump(allbunches)
    #bunch_list = allbunches.split()
    #for b in bunch_list:
    #    bunchmonitor['Bunches'][repr(int(b.bunch_id[0]))]['mean_x'][i] = b.mean_x()
    #    bunchmonitor['Bunches'][repr(int(b.bunch_id[0]))]['mean_z'][i] = b.mean_z()
    
    #for bid in set(np.int_(allbunches.bunch_id)):
    #    msk = (allbunches.bunch_id == bid)
    #    bunchmonitor['Bunches'][repr(bid)]['mean_x'][i] = np.mean(allbunches.x[msk])
    #    bunchmonitor['Bunches'][repr(bid)]['mean_z'][i] = np.mean(allbunches.z[msk])

    if rank == 0:
        t1 = time.clock()
        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))

