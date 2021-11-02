import time

import h5py
from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import c, e
from scipy.stats import linregress
from scipy.signal import hilbert

from LHC import LHC
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor


outputpath = './'  # outputpath relative to this file

n_macroparticles = 10000
n_turns = 5000

# # Parameters for a more realistic simulation
# n_macroparticles = 1e6
# n_turns = int(6e5)

intensity = 1.8e11
Qp_x = -10.0
Qp_y = -10.0
i_oct = 5
dampingrate = 0

# Injection
machine_configuration = 'Injection'
p0 = 450e9*e/c
wakefile = ('wakes/wakeforhdtl_PyZbase_Allthemachine_450GeV'
            '_B1_LHC_inj_450GeV_B1.dat')

# # Flat-top
# machine_configuration = '6.5_TeV_collision'
# p0 = 6.5e12*e/c
# wakefile = ('wakes/wakeforhdtl_PyZbase_Allthemachine_6p5TeV'
#             '_B1_LHC_ft_6.5TeV_B1.dat')

# Detuners
# factor 2p0 is PyHEADTAIL's convention for d/dJx instead of
# MAD-X's convention of d/d(2Jx)
app_x = 2 * p0 * 27380.10941 * i_oct / 100.
app_y = 2 * p0 * 28875.03442 * i_oct / 100.
app_xy = 2 * p0 * -21766.48714 * i_oct / 100.
Qpp_x = 4889.00298 * i_oct / 100.
Qpp_y = -2323.147896 * i_oct / 100.

# Machine
machine = LHC(n_segments=1,
              machine_configuration=machine_configuration,
              **{'app_x': app_x, 'app_y': app_y, 'app_xy': app_xy,
                 'Qp_x': [Qp_x, Qpp_x], 'Qp_y': [Qp_y, Qpp_y]})

# Beam
epsn_x = 1.8e-6  # normalised horizontal emittance
epsn_y = 1.8e-6  # normalised vertical emittance
sigma_z = 1.2e-9 * machine.beta*c/4.  # RMS bunch length in meters

bunch = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

print("\n--> Bunch length and emittance: {:g} m, {:g} eVs.".format(
    bunch.sigma_z(), bunch.epsn_z()))

# Wakes
slicer_for_wakefields = UniformBinSlicer(
    n_slices=500, z_cuts=(-3*sigma_z, 3*sigma_z))
wake_table1 = WakeTable(wakefile,
                        ['time', 'dipole_x', 'dipole_y',
                         'noquadrupole_x', 'noquadrupole_y',
                         'dipole_xy', 'dipole_yx',
                         ])
wake_field = WakeField(slicer_for_wakefields, wake_table1)


# Damper
damper = TransverseDamper(dampingrate, dampingrate)

machine.one_turn_map.append(wake_field)
machine.one_turn_map.append(damper)

# Monitors
try:
    bucket = machine.longitudinal_map.get_bucket(bunch)
except AttributeError:
    bucket = machine.rfbucket

simulation_parameters_dict = {
    'gamma': machine.gamma,
    'beta': machine.beta,
    'intensity': intensity,
    'Qx': machine.Q_x,
    'Qy': machine.Q_y,
    'Qs': bucket.Q_s,
    'beta_x': bunch.beta_Twiss_x(),
    'beta_y': bunch.beta_Twiss_y(),
    'beta_z': bucket.beta_z,
    'epsn_x': bunch.epsn_x(),
    'epsn_y': bunch.epsn_y(),
    'sigma_z': bunch.sigma_z(),
}
bunchmonitor = BunchMonitor(
    filename=outputpath+'/bunchmonitor', n_steps=n_turns,
    parameters_dict=simulation_parameters_dict,
    write_buffer_to_file_every=512, buffer_size=4096)

slicer_for_slicemonitor = UniformBinSlicer(
    n_slices=50, z_cuts=(-3*sigma_z, 3*sigma_z))

n_turns_slicemon = 500  # recording span of the slice statistics monitor

bunch_stats_to_store = [
    'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp']

slicemonitor = SliceMonitor(
    filename=outputpath+'/slicemonitor', n_steps=n_turns,
    slicer=slicer_for_slicemonitor, parameters_dict=simulation_parameters_dict,
    write_buffer_every=64, buffer_size=256,
    **{'bunch_stats_to_store': bunch_stats_to_store},
)

# Save for plotting
x = np.zeros(n_turns, dtype=float)

# Tracking loop
print('\n--> Begin tracking...\n')

for i in range(n_turns):

    t0 = time.time()

    # track the beam around the machine for one turn:
    machine.track(bunch)

    x[i] = bunch.mean_x()

    # monitor the bunch and slice statistics (once per turn):
    bunchmonitor.dump(bunch)
    slicemonitor.dump(bunch)

    # print status every 200 turns:
    if i % 200 == 0:
        t1 = time.time()
        print('Turn {:d}/{:d}'.format(i, n_turns))

print('\n*** Successfully completed!')

# Get data from monitor files
bunch_file = h5py.File(outputpath+'/bunchmonitor.h5')
slice_file = h5py.File(outputpath+'/slicemonitor.h5')

x_from_bunch_file = bunch_file['Bunch']['mean_x'][:]
x_from_slice_file = slice_file['Bunch']['mean_x'][:]

n_per_slice = slice_file['Slices']['n_macroparticles_per_slice'][:]
x_per_slice = slice_file['Slices']['mean_x'][:]
x_from_slices = np.average(x_per_slice, axis=0, weights=n_per_slice)

sx = np.sqrt(epsn_x * bunch_file.attrs['beta_x'] / bunch_file.attrs['gamma']
             / bunch_file.attrs['beta'])

z0, z1 = slicer_for_slicemonitor.z_cuts
dz = (z1 - z0) / slicer_for_slicemonitor.n_slices
z = z0 + (np.array(range(slicer_for_slicemonitor.n_slices)) + 0.5) * dz

# Plot results
turns = np.arange(n_turns)

plt.close('all')
plt.figure(0)

plt.plot(turns, x/sx, label='saved in script')
plt.plot(turns, x_from_bunch_file/sx, '--', label='from bunch monitor')
plt.plot(turns, x_from_slice_file/sx, '-.', label='from slice monitor')
plt.plot(turns, x_from_slices/sx, ':', label='from slices')

ampl = np.abs(hilbert(x))
iMin = 1000
iMax = n_turns - 1000
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
plt.plot(turns, np.exp(a + b * turns)/sx, "--k", label=f"{1/b:.2f} turns")
print(f"Growth rate {b*1e4:.2f} [10^-4/turn]")

plt.title(f"LHC {p0/e*c*1e-9:.0f} GeV")
plt.legend()
plt.xlabel("Turn")
plt.ylabel(r"Mean x [$\sigma_x$]")

plt.figure(1)
n_traces = 75
for tt in range(n_traces):
    plt.plot(z, slice_file['Slices']['mean_x'][:][:, -n_traces + tt]/sx,
             'k', alpha=0.4)

plt.title(f"LHC {p0/e*c*1e-9:.0f} GeV")
plt.xlabel("z [m]")
plt.ylabel(r"x [$\sigma_x$]")

plt.show()

h5py.File.close(bunch_file)
h5py.File.close(slice_file)
