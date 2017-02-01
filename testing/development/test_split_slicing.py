from __future__ import division

import time
import numpy as np
import seaborn as sns
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField


plt.switch_backend('TkAgg')
sns.set_context('talk', font_scale=1.3)
sns.set_style('darkgrid', {
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'lines.markeredgewidth': 1})


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_turns = 1
chroma = 0
n_bunches = 3
intensity = 2.3e11
n_macroparticles = 1000000


# BEAM AND MACHNINE PARAMETERS
# ============================
from HLLHC import HLLHC
# from TSLHC import HLLHC
machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                circumference=40, h_RF=[30],
                machine_configuration='7_TeV_collision_tunes',
                longitudinal_focusing='non-linear',
                Qp_x=chroma, Qp_y=chroma, wrap_z=True)
C = machine.circumference
h = np.min(machine.longitudinal_map.harmonics) * 1.
filling_scheme = sorted([5 + 10*i + h*j for i in range(n_bunches) for j in range(n_turns)])


# BEAM
# ====
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081
allbunches = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme)
allbunches.x *= 0
allbunches.xp *= 0
allbunches.y *= 0
allbunches.yp *= 0
allbunches.x[:] += 2e-2

bunches = allbunches.split()


# CREATE BEAM SLICERS
# ===================
slicer_for_diagnostics = UniformBinSlicer(10000, z_cuts=(-C*n_turns, 0))
slicer_for_wakefields = UniformBinSlicer(20, z_cuts=(-0.3, 0.3))
slicers_list = [UniformBinSlicer(20, z_cuts=(-0.3-f*C/h, 0.3-f*C/h)) for f in filling_scheme]


# CREATE WAKES
# ============
wakes = CircularResonator(1e6, 50e6, 50, n_turns_wake=1)
wake_field = WakeField(slicer_for_wakefields, wakes,
                       circumference=machine.circumference, mpi=True)
# wake_field = ParallelWakes(slicer_for_wakefields, wake_sources_list=None,
#                            circumference=machine.circumference,
#                            filling_scheme=filling_scheme,
#                            comm=comm)
w_function = wake_field.wake_kicks[0].wake_function
w_factor = wake_field.wake_kicks[0]._wake_factor



# REFERENCE
# =========
slices = allbunches.get_slices(slicer_for_diagnostics,
                               statistics=['mean_x', 'mean_y'])
zbins = slices.z_bins
times = slices.t_centers
locations = slices.z_centers
moments = slices.n_macroparticles_per_slice * slices.mean_x
tmin, tmax = times[0], times[-1]
t = np.hstack((times-tmax, (times-tmin)[1:]))
x_kick = (np.convolve(w_function(t, allbunches.beta), moments,
                      mode='valid'))
xkickmax = np.max(x_kick)

# # MOVE SLICERS
# # ============
# bins_slicers = []
# times_slicers = []
# moments_slicers = []
# for l, s in enumerate(slicers_list):
#     sl = allbunches.get_slices(s, statistics=['mean_x'])
#     bins_slicers.append(sl.z_bins)
#     times_slicers.append(sl.t_centers)
#     moments_slicers.append(sl.n_macroparticles_per_slice * sl.mean_x)
# bins_slicers = np.array(bins_slicers).T
# times_slicers = np.array(times_slicers).T
# moments_slicers = np.array(moments_slicers).T

# MOVE BUNCHES
# ============
slices_list = []
for l, b in enumerate(bunches):
    z_delay = -filling_scheme[l]*C/h
    # z_delay = -b.mean_z()
    b.z -= z_delay
    s = b.get_slices(slicer_for_wakefields, statistics=['mean_x', 'mean_y'])
    b.z += z_delay
    s.z_bins += z_delay #+ s.slice_widths[0]/2.
    slices_list.append(s)
bins_list = np.array([s.z_bins for s in slices_list])
times_list = np.array([s.t_centers for s in slices_list])
locations_list = np.array([s.z_centers for s in slices_list])
moments_list = np.array([s.n_macroparticles_per_slice*s.mean_x for s in slices_list])
x_kick_list = []
for j, bt in enumerate(bunches):
    signal = 0
    for i, bs in enumerate(bunches):
        t_source = times_list[i]
        t_target = times_list[j]
        tmin, tmax = t_source[0], t_source[-1]
        dt = np.hstack((t_target-tmax, (t_target-tmin)[1:]))
        mm = moments_list[i]
        signal += np.convolve(w_function(dt, bs.beta), mm, mode='valid')
    x_kick_list.append(signal)
x_kick_list = np.array(x_kick_list)


# PLOT RESULTS
# ============
plt.close('all')
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True)
# [ax1.axvline(a, c='b') for a in zbins]
# [ax1.axvline(a, c='g') for b in bins_slicers for a in b]
# [ax1.axvline(a, c='darkred', ls='--') for b in bins_list for a in b]

ax1.plot(times*allbunches.beta*c, moments/np.max(moments), c='b', ls='-')
# ax1.plot(times_slicers, moments_slicers/np.max(moments_slicers), c='g', marker='o', ls='None')
ax1.plot(times_list*allbunches.beta*c, moments_list/np.max(moments_list), c='darkred', marker='o', ls='None')

# ax2.plot(times, w_function(times, allbunches.beta))

ax2.plot(locations, moments/np.max(moments), c='b')
ax2.plot(locations_list[::-1], moments_list/np.max(moments_list), c='darkred', marker='o', ls='None')
# ax3.plot(times, x_kick, c='b', marker='o', ls='None')
# ax3.plot(times_list, x_kick_list/np.max(x_kick_list), c='darkred', marker='d', ls='None')

# ax1.set_xlim(-C*n_turns, 0)
# ax1.legend([l1[0], l2[0], l3[0]], ["Reference", "Move slicers", "Move bunches"])
plt.show()







# allbunches = comm.gather(bunches, root=0)
# if rank == 0:
#     allslices = allbunches[0].get_slices(slicer_for_diagnostics, statistics=['mean_x'])
#     alltimes = allslices.t_centers
#     allmoments = allslices.mean_x * allslices.n_macroparticles_per_slice

#     alltimes_list = []
#     allmoments_list = []
#     for b in allbunches[0].split():
#         delay = b.mean_z()
#         b.z -= delay
#         s = b.get_slices(slicer_for_wakefields, statistics=['mean_x'])
#         b.z += delay
#         s.z_bins += delay
#         alltimes_list.append(s.t_centers)
#         allmoments_list.append(s.mean_x * s.n_macroparticles_per_slice)
#     alltimes_list = np.array(alltimes_list).T
#     allmoments_list = np.array(allmoments_list).T

    # ax1.plot(times_list.T, moments_list.T/np.max(moments_list), 'o', label="Bad manual")
    # ax1.plot(alltimes, allmoments/np.max(allmoments), 'd', label="Good PyHEADTAIL")
    # ax1.plot(alltimes_list, allmoments_list/np.max(allmoments_list), '*', label="Bad PyHEADTAIL")
    # ax1.plot(times, moments/np.max(moments), label="Reference")
    # ax1.plot(times_slicers, moments_slicers/np.max(moments_slicers), label="Wurst")
    # ax2.plot(times, w_function(times, bunches.beta))
    # [ax2.plot(b.z/c, w_function(b.z/c, b.beta), 'o') for b in allbunches]
    # ax3.plot(times, x_kick/xkickmax)
    # ax3.plot(times_list.T, x_kick_list.T/xkickmax, 'd', ms=10)
    # [ax3.plot(b.z/c, b.xp/xkickmax, 'o') for b in allbunches]
    # ax1.legend(loc=0)
    # plt.show()
