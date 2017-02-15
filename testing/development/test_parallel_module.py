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
    'lines.markeredgewidth': 1,
    'font.family': serif,
    'text.usetex': True})


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_turns = 1
chroma = 0

n_bunches = 3
# filling_scheme = [401 + 200*i for i in range(n_bunches)]
filling_scheme = [5 + 10*i for i in range(n_bunches)]
n_macroparticles = 200000
intensity = 2.3e11


# BEAM AND MACHNINE PARAMETERS
# ============================
from HLLHC import HLLHC
# from TSLHC import HLLHC
machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                circumference=50, h_RF=[40],
                machine_configuration='7_TeV_collision_tunes',
                longitudinal_focusing='non-linear',
                Qp_x=chroma, Qp_y=chroma, wrap_z=True)
# machine.circumference = 30
# machine.longitudinal_map.harmonics[0] = 40.
C = machine.circumference
h = np.min(machine.longitudinal_map.harmonics) * 1.
z = -1.*(np.array(filling_scheme))/h*C


# BEAM
# ====
# allbunches = []
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081
bunches = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme)
bunches.x *= 0
bunches.xp *= 0
bunches.y *= 0
bunches.yp *= 0
bunches.x[:] += 2e-2


# CREATE BEAM SLICERS
# ===================
slicer_for_diagnostics = UniformBinSlicer(100000, n_sigma_z=3)
slicer_for_diagnostics = UniformBinSlicer(10000, z_cuts=(-C*n_turns, 0))
slicer_for_wakefields = UniformBinSlicer(200, z_cuts=(-0.3, 0.3))


# CREATE WAKES
# ============
wakes = CircularResonator(1e6, 50e6, 50, n_turns_wake=10)
wake_field = WakeField(slicer_for_wakefields, wakes,
                       circumference=machine.circumference, mpi=True)
# wake_field = ParallelWakes(slicer_for_wakefields, wake_sources_list=None,
#                            circumference=machine.circumference,
#                            filling_scheme=filling_scheme,
#                            comm=comm)
w_function = wake_field.wake_kicks[0].wake_function
w_factor = wake_field.wake_kicks[0]._wake_factor


# CREATE DAMPER
# =============
dampingrate = 50
damper = TransverseDamper(dampingrate, dampingrate)


# CREATE MONITORS
# ===============
try:
    bucket = machine.longitudinal_map.get_bucket(bunches)
except AttributeError:
    bucket = machine.rfbucket

simulation_parameters_dict = {
    'gamma': machine.gamma,
    'intensity': intensity,
    'Qx': machine.Q_x,
    'Qy': machine.Q_y,
    'Qs': bucket.Qs,
    'beta_x': bunches.beta_Twiss_x(),
    'beta_y': bunches.beta_Twiss_y(),
    'beta_z': bucket.beta_z,
    'epsn_x': bunches.epsn_x(),
    'epsn_y': bunches.epsn_y(),
    'sigma_z': bunches.sigma_z(),
}
# bunchmonitor = BunchMonitor(
#     outputpath+'/bunchmonitor_{:04d}_chroma={:g}'.format(it, chroma),
#     n_turns, simulation_parameters_dict,
#     write_buffer_to_file_every=512, buffer_size=4096)
# slicemonitor = SliceMonitor(
#     outputpath+'/slicemonitor_{:04d}_chroma={:g}'.format(it, chroma),
#     2048, slicer_for_slicemonitor,
#     simulation_parameters_dict, write_buffer_to_file_every=512, buffer_size=4096)




# SYNTHETIC TRACKING LOOP
if rank == 0:
    filling_scheme = sorted([5 + 10*i + h*j for i in range(n_bunches) for j in range(n_turns)])
    allbunches = machine.generate_6D_Gaussian_bunch_matched(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
        filling_scheme=filling_scheme)
    allbunches.x *= 0
    allbunches.xp *= 0
    allbunches.y *= 0
    allbunches.yp *= 0
    allbunches.x[:] += 2e-2


    # bunches = allbunches.split()
    # print(filling_scheme)

    # [plt.plot(b.z, b.bunch_id) for b in bunches]
    # plt.plot(allbunches.z, allbunches.bunch_id, 'o')

    # wurstel


    slices = allbunches.get_slices(slicer_for_diagnostics,
                                   statistics=['mean_x', 'mean_y'])
    times = slices.t_centers
    tmin, tmax = times[0], times[-1]
    t = np.hstack((times-tmax, (times-tmin)[1:]))
    moments = slices.mean_x * slices.n_macroparticles_per_slice
    x_kick = (np.convolve(w_function(t, allbunches.beta), moments,
                          mode='valid'))
    xkickmax = np.max(x_kick)

    # # we *need* to weight here - this is physical! Else, the convolution is effectively different.
    # moments2 = slices.mean_x * slices.n_macroparticles_per_slice
    # x_kick2 = (np.convolve(w_function(t, allbunches.beta), moments2,
    #                       mode='valid'))
    # xkickmax2 = np.max(x_kick2)


    ###############################################################################################
    slicers_list = [UniformBinSlicer(200, z_cuts=(-0.3-f*C/h, 0.3-f*C/h)) for f in filling_scheme]
    times_wurst = []
    moments_wurst = []
    for l, s in enumerate(slicers_list):
        sl = allbunches.get_slices(s, statistics=['mean_x'])
        times_wurst.append(sl.t_centers)
        moments_wurst.append(sl.mean_x*sl.n_macroparticles_per_slice)
    times_wurst = np.array(times_wurst).T
    moments_wurst = np.array(moments_wurst).T


    bunches_list = allbunches.split()
    slices_list = []
    for l, b in enumerate(bunches_list):
        z_delay = b.mean_z()
        z_delay = -filling_scheme[l]*C/h
        b.z -= z_delay
        s = b.get_slices(slicer_for_wakefields, statistics=['mean_x', 'mean_y'])
        b.z += z_delay
        s.z_bins += z_delay #+ s.slice_widths[0]/2.
        slices_list.append(s)
    times_list = np.array([s.t_centers for s in slices_list])
    # we *need* to weight here - physical! Else, the convolution is effectively different.
    moments_list = np.array(
        [s.n_macroparticles_per_slice*s.mean_x for s in slices_list])

    x_kick_list = []
    for j, bt in enumerate(bunches_list):
        signal = 0
        for i, bs in enumerate(bunches_list):
            t_source = times_list[i]
            t_target = times_list[j]
            tmin, tmax = t_source[0], t_source[-1]
            dt = np.hstack((t_target-tmax, (t_target-tmin)[1:]))
            mm = moments_list[i]
            signal += np.convolve(w_function(dt, bs.beta), mm, mode='valid')
        x_kick_list.append(signal)
    x_kick_list = np.array(x_kick_list)
    ###############################################################################################


# # allbunches = comm.gather(bunches, root=0)
# if rank == 0:
#     fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True)
#     ax1.plot(times, moments/np.max(moments))
#     ax1.plot(times_list.T, moments_list.T/np.max(moments_list), 'o')
#     ax2.plot(times, w_function(times, bunches.beta))
#     # [ax2.plot(b.z/c, w_function(b.z/c, b.beta), 'o') for b in allbunches]
#     ax3.plot(times, x_kick/xkickmax)
#     ax3.plot(times_list.T, x_kick_list.T/xkickmax, 'd', ms=10)
#     plt.show()

# print '\n*** Successfully completed!'



# wurstel


# TRACKING LOOP
# =============
# machine.one_turn_map.append(damper)
a = machine.one_turn_map.pop()
a = machine.one_turn_map.pop()
a = machine.one_turn_map.pop()
machine.one_turn_map.append(wake_field)

# s_cnt = 0
# monitorswitch = False
# if rank == 0:
#     print '\n--> Begin tracking...\n'
# for i in range(n_turns):

#     if rank == 0:
#         t0 = time.clock()
#     # bunchmonitor.dump(bunch)

#     # if not monitorswitch:
#     #     if (bunch.mean_x() > 1e3 or bunch.mean_y() > 1e3 or
#     #             i > n_turns - 2048):
#     #         print "--> Activate monitor"
#     #         monitorswitch = True
#     # else:
#     #     if s_cnt < 2048:
#     #         slicemonitor.dump(bunch)
#     #         s_cnt += 1

#     machine.track(bunches)

#     if rank == 0:
#         t1 = time.clock()
#         print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
#             "%d/%m/%Y %H:%M:%S", time.localtime())))


allbunches = comm.gather(bunches, root=0)
if rank == 0:
    allslices = allbunches[0].get_slices(slicer_for_diagnostics, statistics=['mean_x'])
    alltimes = allslices.t_centers
    allmoments = allslices.mean_x * allslices.n_macroparticles_per_slice

    alltimes_list = []
    allmoments_list = []
    for b in allbunches[0].split():
        delay = b.mean_z()
        b.z -= delay
        s = b.get_slices(slicer_for_wakefields, statistics=['mean_x'])
        b.z += delay
        s.z_bins += delay
        alltimes_list.append(s.t_centers)
        allmoments_list.append(s.mean_x * s.n_macroparticles_per_slice)
    alltimes_list = np.array(alltimes_list).T
    allmoments_list = np.array(allmoments_list).T

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True)
    ax1.plot(times_list.T, moments_list.T/np.max(moments_list), 'o', label="Bad manual")
    ax1.plot(alltimes, allmoments/np.max(allmoments), 'd', label="Good PyHEADTAIL")
    ax1.plot(alltimes_list, allmoments_list/np.max(allmoments_list), '*', label="Bad PyHEADTAIL")
    ax1.plot(times, moments/np.max(moments), label="Reference")
    ax1.plot(times_wurst, moments_wurst/np.max(moments_wurst), label="Wurst")
    ax2.plot(times, w_function(times, bunches.beta))
    [ax2.plot(b.z/c, w_function(b.z/c, b.beta), 'o') for b in allbunches]
    ax3.plot(times, x_kick/xkickmax)
    ax3.plot(times_list.T, x_kick_list.T/xkickmax, 'd', ms=10)
    [ax3.plot(b.z/c, b.xp/xkickmax, 'o') for b in allbunches]
    ax1.legend(loc=0)
    plt.show()

print '\n*** Successfully completed!'
