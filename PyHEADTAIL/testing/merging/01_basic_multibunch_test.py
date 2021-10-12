

import os
import sys

BIN = os.path.expanduser("../../../")
sys.path.append(BIN)

import time, copy
import numpy as np
import seaborn as sns
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField

plt.switch_backend('TkAgg')

sns.set(context='talk', style='darkgrid', palette='deep',
        font='sans-serif', font_scale=1.8, color_codes=False,
        rc={'axes.linewidth': 2,
            'axes.edgecolor': 'black',
            'lines.markeredgewidth': 1,
            'text.usetex': True,
            'font.sans-serif': 'Helvetica',
})

np.random.seed(0)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_turns = 2
chroma = 0
n_bunches = 4
intensity = 2.3e11
n_macroparticles = 20000

# BEAM AND MACHNINE PARAMETERS
# ============================
from HLLHC import HLLHC

# from TSLHC import HLLHC
machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                circumference=70, h_RF=[40],
                machine_configuration='7_TeV_collision_tunes',
                longitudinal_focusing='non-linear',
                Qp_x=chroma, Qp_y=chroma, wrap_z=True)
h_bunch = 40
C = machine.circumference
h = np.min(machine.longitudinal_map.harmonics) * 1.
filling_scheme = sorted([5 + 2 * i ** 2 + h * j for i in range(n_bunches) for j in range(1)])
filling_scheme_synth = sorted([5 + 2 * i ** 2 + h * j for i in range(n_bunches) for j in range(n_turns)])

# BEAM
# ====
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081
allbunches = machine.generate_6D_Gaussian_bunch(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme, matched=True)
allbunches.x *= 0
allbunches.xp *= 0
allbunches.y *= 0
allbunches.yp *= 0
allbunches.x[:] += 2e-2

synthbunches = machine.generate_6D_Gaussian_bunch(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme_synth, matched=True)
synthbunches.x *= 0
synthbunches.xp *= 0
synthbunches.y *= 0
synthbunches.yp *= 0
synthbunches.x[:] += 2e-2

# CREATE BEAM SLICERS
# ===================
slicer_for_diagnostics = UniformBinSlicer(10000 / 2. * n_turns, z_cuts=(-C * n_turns, 0),
                                          circumference=machine.circumference, h_bunch=40)
slicer_for_wakefields = UniformBinSlicer(20, z_cuts=(-0.4, 0.4),
                                         circumference=machine.circumference, h_bunch=40)

# CREATE WAKES
# ============
wakes = CircularResonator(1.1e6, 50.6e6, 18, n_turns_wake=n_turns)

wake_field = WakeField(slicer_for_wakefields, wakes, mpi='memory_optimized')
# wake_field = WakeField(slicer_for_wakefields, wakes, mpi='linear_mpi_full_ring_fft')
# wake_field = WakeField(slicer_for_wakefields, wakes, mpi='circular_mpi_full_ring_fft',
#                       Q_x=machine.Q_x, Q_y=machine.Q_y,
#                       beta_x=machine.beta_x, beta_y=machine.beta_y)
# wake_field = WakeField(slicer_for_wakefields, wakes, mpi=True)

w_function = wake_field.wake_kicks[0].wake_function
w_factor = wake_field.wake_kicks[0]._wake_factor

# TRACKING
# ========

# SYNTHETIC TRACKING LOOP
# =======================
synthbunches = comm.gather(synthbunches, root=0)
if rank == 0:
    # REFERENCE
    # =========
    synthbunches = sum(synthbunches)
    synthbunches2 = synthbunches.split()
    for b in synthbunches2:
        b.z -= b.bucket_id[0] * C / float(h_bunch)
        print(('b.mean_z(): ' + str(b.mean_z())))

    synthbunches2 = sum(synthbunches2)
    slices = synthbunches2.get_slices(slicer_for_diagnostics,
                                      statistics=['mean_x', 'mean_y'])
    zbins = slices.z_bins
    times = slices.t_centers + slices.bucket_id * slices.circumference / float(slices.h_bunch) / c
    locations = slices.z_centers + slices.bucket_id * slices.circumference / float(slices.h_bunch)
    moments = slices.n_macroparticles_per_slice * slices.mean_x
    tmin, tmax = times[0], times[-1]
    t = np.hstack((times - tmax, (times - tmin)[1:]))
    x_kick = (np.convolve(w_function(t, allbunches.beta), moments,
                          mode='valid'))  # * slices.slice_widths[0]
    xkickmax = np.max(x_kick)

    # MOVE BUNCHES
    # ============
    bunches = synthbunches.split()
    slices_list = []
    for l, b in enumerate(bunches):
        s = b.get_slices(slicer_for_wakefields, statistics=['mean_x', 'mean_y'])
        slices_list.append(s)
    sw = np.array([s.slice_widths[0] for s in slices_list])
    bins_list = np.array([s.z_bins for s in slices_list])
    times_list = np.array([s.t_centers for s in slices_list])
    locations_list = np.array([s.z_centers for s in slices_list])
    moments_list = np.array([s.n_macroparticles_per_slice * s.mean_x for s in slices_list])
    x_kick_list = []
    for j, bt in enumerate(bunches):
        signal = 0
        for i, bs in zip(list(range(len(bunches)))[::-1], bunches[::-1]):
            # for i, bs in enumerate(bunches[::-1]):
            t_source = times_list[i]
            t_target = times_list[j]
            tmin, tmax = t_source[0], t_source[-1]

            dt = np.hstack((t_target - tmax, (t_target - tmin)[1:]))
            mm = moments_list[i]
            signal += np.convolve(w_function(dt, bs.beta), mm, mode='valid')  # * sw[i]
        x_kick_list.append(signal)
    x_kick_list = np.array(x_kick_list)

# TRACKING LOOP
# =============
a = machine.one_turn_map.pop()
a = machine.one_turn_map.pop()
a = machine.one_turn_map.pop()
machine.one_turn_map.append(wake_field)

s_cnt = 0
monitorswitch = False
delta_xp = np.zeros((n_turns, len(allbunches.xp)))
delta_zz = np.zeros((n_turns, len(allbunches.xp)))
if rank == 0:
    print('\n--> Begin tracking...\n')
for i in range(n_turns):

    if rank == 0:
        t0 = time.clock()

    xp_old = allbunches.xp.copy()
    machine.track(allbunches)
    delta_xp[i, :] = allbunches.xp - xp_old

    allbunches_c = copy.deepcopy(allbunches)
    allbunches_c = allbunches_c.split()

    for b in allbunches_c:
        b.z -= b.bucket_id[0] * C / float(h_bunch)
    allbunches_c = sum(allbunches_c)
    delta_zz[i, :] = allbunches_c.z.copy()

    if rank == 0:
        t1 = time.clock()
        print(('Turn {:d}, {:g} ms, {:s}'.format(i, (t1 - t0) * 1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime()))))

# PLOT RESULTS
# ============
delta_xp = comm.gather(delta_xp, root=0)
delta_zz = comm.gather(delta_zz, root=0)
bunches = comm.gather(allbunches, root=0)
if rank == 0:
    bunches = [b.split() for b in bunches]
    bunches = sum(bunches, [])
    delta_xp = np.hstack(delta_xp)
    delta_zz = np.hstack(delta_zz)
    print((delta_xp.shape, delta_zz.shape))

    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(16, 9), sharex=True, tight_layout=False)
    fig.suptitle("Parallelized PyHEADTAIL multi-bunch, multi-turn wakes\n" +
                 "with {:d} bunches over {:d} turns on {:d} processors... :D!".format(
                     n_bunches, n_turns, comm.size), fontsize=20)

    slices_list = []
    for l, b in enumerate(bunches):
        s = b.get_slices(slicer_for_wakefields, statistics=['mean_x', 'mean_y'])
        slices_list.append(s)
    sw_pyht = np.array([s.slice_widths[0] for s in slices_list])
    bins_list_pyht = np.array([s.z_bins for s in slices_list])
    times_list_pyht = np.array([s.t_centers for s in slices_list]) - machine.circumference / machine.beta / c * i
    locations_list_pyht = np.array([s.z_centers for s in slices_list])
    moments_list_pyht = np.array([s.n_macroparticles_per_slice * s.mean_x for s in slices_list])

    ax1.plot(times_list, moments_list / np.max(moments_list), c='darkred', marker='o', ls='None')
    ax1.plot(times_list_pyht.T, moments_list_pyht.T / np.max(moments_list_pyht), marker='o', ms='6', ls='None')

    ax2.plot(times, w_function(times, allbunches.beta))

    ax3.plot(times, x_kick / xkickmax, c='darkolivegreen', ls='-')
    ax3.plot(times_list, x_kick_list / xkickmax, c='darkred', marker='d', ls='None')
    [ax3.plot(delta_zz[i, :] / machine.beta / c - i * machine.circumference / machine.beta / c,
              delta_xp[i, :] / w_factor(bunches[0]) / xkickmax, marker='o', ms='6', ls='None') for i in range(n_turns)]

    [[ax.axvline(-i * machine.circumference / machine.beta / c, ls='--', lw=2, c='0.2') for i in range(n_turns)]
     for ax in [ax1, ax2, ax3]]

    # ax1.set_xlim(-C*n_turns, 0)
    # ax1.legend([l1[0], l2[0], l3[0]], ["Reference", "Move slicers", "Move bunches"])
    ax1.set_ylabel("Bunch profiles\n[normalized]")
    ax2.set_ylabel("Wake field\n[arb. units]")
    ax3.set_ylabel("Wake field kick\n[normalized]")
    ax3.set_xlabel("Time [s]")

    ww = w_function(times, allbunches.beta)
    ax2.set_ylim((-np.max(np.abs(ww)), np.max(np.abs(ww))))
    ax3.set_xlim(-n_turns * machine.circumference / machine.beta / c, 0)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

#    plt.savefig("Multibunch_PyHEADTAIL.png", dpi=200)
plt.show()
