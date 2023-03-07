import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.constants import c, e, m_p
import time

from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.monitors.monitors import BunchMonitor
from PyHEADTAIL.particles.slicing import UniformBinSlicer


# Machine settings
n_turns = 300

n_macroparticles = 1000 # per bunch 
intensity = 2.3e11
intensity = 2.3e13

alpha = 53.86**-2

p0 = 7000e9 * e / c

accQ_x = 62.31
accQ_y = 60.32
Q_s = 2.1e-3
chroma = 0

circumference = 26658.883

beta_x = circumference / (2.*np.pi*accQ_x)
beta_y = circumference / (2.*np.pi*accQ_y)

h_RF = 35640
h_bunch = 3564

epsn_x = 2e-6
epsn_y = 2e-6
sigma_z = 0.09

machine = Synchrotron(
        optics_mode='smooth', circumference=circumference,
        n_segments=1, s=None, name=None,
        alpha_x=None, beta_x=beta_x, D_x=0,
        alpha_y=None, beta_y=beta_y, D_y=0,
        accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=chroma, Qp_y=chroma,
        app_x=0, app_y=0, app_xy=0,
        alpha_mom_compaction=alpha, longitudinal_mode='linear',
        h_RF=np.atleast_1d(h_RF), p0=p0,
        charge=e, mass=m_p, wrap_z=False, Q_s=Q_s)


# Filling scheme

filling_scheme = [] # A list of filled buckets
for i in range(3):
    for j in range(72):
        filling_scheme.append(i*80+j)

allbunches = machine.generate_6D_Gaussian_bunch(n_macroparticles, intensity,
                                                epsn_x, epsn_y, sigma_z=sigma_z,
                                                filling_scheme=filling_scheme,
                                                matched=False)


# Wakes

slicer = UniformBinSlicer(20, z_cuts=(-2.*sigma_z, 2.*sigma_z),
                          circumference=machine.circumference, h_bunch=h_bunch)

n_turns_wake = 3

# from PyHEADTAIL.impedances.wakes import CircularResistiveWall, WakeField
# # pipe radius [m]
# b = 13.2e-3
# # length of the pipe [m]
# L = 100000.
# # conductivity of the pipe 1/[Ohm m]
# sigma = 1. / 7.88e-10
# wakes = CircularResistiveWall(b, L, sigma, b/c, beta_beam=machine.beta)

from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
R_shunt = 135e6
frequency = 1.97e9
Q = 3100
wakes = CircularResonator(R_shunt, frequency, Q, n_turns_wake=n_turns_wake)


# Choose optimization algorithm for wake field calculation

# mpi_settings = False
# mpi_settings = True
# mpi_settings = 'memory_optimized'
mpi_settings = 'linear_mpi_full_ring_fft'
wake_field = WakeField(slicer, wakes, mpi=mpi_settings)

# mpi_settings = 'circular_mpi_full_ring_fft'
# wake_field = WakeField(slicer, wakes, mpi=mpi_settings, Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)

machine.one_turn_map.append(wake_field)


# Tracking

simulation_parameters_dict = {'gamma': machine.gamma,
                              'intensity': intensity,
                              'Qx': accQ_x,
                              'Qy': accQ_y,
                              'Qs': Q_s,
                              'beta_x': beta_x,
                              'beta_y': beta_y,
                              'epsn_x': epsn_x,
                              'epsn_y': epsn_y,
                              'sigma_z': sigma_z,
                             }

bunchmonitor = BunchMonitor('bunch_by_bunch_data', n_turns,
                            simulation_parameters_dict,
                            write_buffer_every=10, buffer_size=10,
                            mpi=True, filling_scheme=filling_scheme)

for i in range(n_turns):
    t0 = time.time()

    machine.track(allbunches)
    bunchmonitor.dump(allbunches)

    bunch_list = allbunches.split_to_views()
    bunch = bunch_list[0]
    if i == 0:
        print('Turn      mean_x      mean_y      mean_z     epsn_x     epsn_y     epsn_z   sigma_z  sigma_dp     Time')
    if i % 10 == 0:
        print('{:4d}   {:+.2e}   {:+.2e}   {:+.2e}   {:.2e}   {:.2e}   {:.2e}      {:.2f}      {:.2f}   {:2s}'.format(
            i, bunch.mean_x(), bunch.mean_y(), bunch.mean_z(), bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(),
            bunch.sigma_z(), bunch.sigma_dp(), str(time.time() - t0)))
print('\n*** Successfully completed!')


# Bunch monitor data from the h5 file

h5f = h5py.File('bunch_by_bunch_data.h5','r')

data_mean_x = None
data_mean_y = None
data_mean_z = None
    

for i, bunch_id in enumerate(filling_scheme):
    t_mean_x = h5f['Bunches'][str(bunch_id)]['mean_x'][:]
    t_epsn_x = h5f['Bunches'][str(bunch_id)]['epsn_x'][:]
    t_mean_y = h5f['Bunches'][str(bunch_id)]['mean_y'][:]
    t_mean_z = h5f['Bunches'][str(bunch_id)]['mean_z'][:]

    if data_mean_x is None:
        valid_map = (t_epsn_x > 0)

        turns = np.linspace(1, np.sum(valid_map), np.sum(valid_map))
        bunch_spacing = circumference / float(h_bunch)

        data_mean_x = np.zeros((np.sum(valid_map), len(filling_scheme)))
        data_mean_y = np.zeros((np.sum(valid_map), len(filling_scheme)))
        data_mean_z = np.zeros((np.sum(valid_map), len(filling_scheme)))

    np.copyto(data_mean_x[:,i], t_mean_x[valid_map])
    np.copyto(data_mean_y[:,i], t_mean_y[valid_map])
    np.copyto(data_mean_z[:,i], t_mean_z[valid_map] + -bunch_id*bunch_spacing)

try:
    os.remove('./bunch_by_bunch_data.h5')
except FileNotFoundError:
    pass

# Turn-by-turn bunch postion plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1_z = ax1.twiny()
ax2_z = ax2.twiny()

plot_n_turns = 50

ax1.set_prop_cycle(color=[plt.cm.viridis(i) for i in np.linspace(0, 1, plot_n_turns)])
ax2.set_prop_cycle(color=[plt.cm.viridis(i) for i in np.linspace(0, 1, plot_n_turns)])

for i in range(plot_n_turns):

    ax1.plot(filling_scheme, data_mean_x[-(i+1), :]*1e3, '.')
    ax2.plot(filling_scheme, data_mean_y[-(i+1), :]*1e3, '.')
    if i == 0:
        ax1_z.plot(data_mean_z[-(i+1), :], np.zeros(len(data_mean_x[-(i+1), :])),'.')
        ax1_z.cla()
        ax2_z.plot(data_mean_z[-(i+1), :], np.zeros(len(data_mean_x[-(i+1), :])),'.')
        ax2_z.cla()
        ax1_z.set_xlim(np.max(data_mean_z[-(i+1), :]), np.min(data_mean_z[-(i+1), :]))
        ax2_z.set_xlim(np.max(data_mean_z[-(i+1), :]), np.min(data_mean_z[-(i+1), :]))

ax1_z.set_xlabel('Distance [m]')
ax2_z.set_xlabel('Distance [m]')

ax1.set_xlabel('Bucket #')
ax2.set_xlabel('Bucket #')

ax1.set_ylabel('Bunch mean_x [mm]')
ax2.set_ylabel('Bunch mean_y [mm]')
plt.tight_layout()
plt.show()

