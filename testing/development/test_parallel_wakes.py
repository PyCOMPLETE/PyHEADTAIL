from __future__ import division

import time
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField


plt.switch_backend('TkAgg')


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_turns = 2
chroma = 0

n_bunches = 13
filling_scheme = [401 + 200*i for i in range(n_bunches)]
n_macroparticles = 10000
intensity = 2.3e11


# BEAM AND MACHNINE PARAMETERS
# ============================
from HLLHC import HLLHC
machine = HLLHC(charge=e, mass=m_p, n_segments=1,
                machine_configuration='7_TeV_collision_tunes',
                longitudinal_focusing='non-linear',
                Qp_x=chroma, Qp_y=chroma, wrap_z=True)
C = machine.circumference
h = np.min(machine.longitudinal_map.harmonics)
z = C/h * (np.array(filling_scheme)-h) + C/(2*h)*0


'''extract_bunches is exteremely expensive so you would want to do this exactly
once (if, at all, since anyway, you are generating bunches)! In this case, we
might still have to live with the wake track method able to accept a bunches
list - you won't want it to extract bunches at every turn!

Try another method implementing a bunches.split method. The, we can keep all
track methods generic acting on a particles object.

'''


# Distribute among processors
# ===========================
size = 4
n_bunches_per_proc = n_bunches//size
n_bunches_remainder = n_bunches % size
bunches_on_proc_list = [filling_scheme[i:i+n_bunches_per_proc+1]
                        if i < n_bunches_remainder
                        else filling_scheme[i+1:i+n_bunches_per_proc+1]
                        for i in range(0, n_bunches, n_bunches_per_proc)
                        if filling_scheme[i+1:i+n_bunches_per_proc+1]]
print("\n\n*** Summary of bunches on processors {:s}".format(
    bunches_on_proc_list))


# BEAM
# ====
# allbunches = []
epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081
bunches = [machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    bunch_id=b_id) for b_id in bunches_on_proc_list[rank]]

# Bunches on processors
for i, b_id in enumerate(bunches_on_proc_list[rank]):
    bunches[i].z += (b_id-h) * C/h
    bunches[i].dt = (b_id-h) * C/h/c
bunches = sum(bunches)


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
buncher_for_wakefields = machine.buncher
slicer_for_wakefields = UniformBinSlicer(300, n_sigma_z=3)


# CREATE WAKES
# ============
wakes = CircularResonator(1e6, 1e9, 1)
wake_field = WakeField(slicer_for_wakefields, wakes,
                       circumference=machine.circumference, comm=comm)
# wake_field = ParallelWakes(slicer_for_wakefields, wake_sources_list=None,
#                            circumference=machine.circumference,
#                            filling_scheme=filling_scheme,
#                            comm=comm)


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


# TRACKING LOOP
# =============
machine.one_turn_map.append(damper)
machine.one_turn_map.append(wake_field)

s_cnt = 0
monitorswitch = False
if rank == 0:
    print '\n--> Begin tracking...\n'
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

allbunches = comm.gather(bunches, root=0)
if rank == 0:
    fig, (ax1) = plt.subplots(1, figsize=(16, 9))
    for b in allbunches:
        ax1.plot(b.z, b.dp, '.')
    ax1.set_xlim(0, 1000)
    # plt.show()

print '\n*** Successfully completed!'
