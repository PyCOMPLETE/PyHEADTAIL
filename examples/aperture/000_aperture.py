import time

import h5py
from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import c as c_light, e as qe, m_p

from PyHEADTAIL.aperture.aperture import EllipticalApertureXY
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.monitors.monitors import ParticleMonitor


outputpath = './'  # outputpath relative to this file

n_turns = 60
macroparticlenumber = int(1e4)

# Create machine
machine_name = 'LHC'
p0_eVperc = 6.8e12
p0 = p0_eVperc * qe / c_light

beta_x = 92.7
beta_y = 93.2

Q_x = 64.31
Q_y = 59.32

alpha_momentum = 3.225e-4
h_RF = 35640
V_RF = 12.0e6
circumference = 26658.883199999

machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                      n_segments=1, beta_x=beta_x, beta_y=beta_y,
                      D_x=0.0, D_y=0.0, accQ_x=Q_x, accQ_y=Q_y,
                      alpha_mom_compaction=alpha_momentum,
                      longitudinal_mode='non-linear', h_RF=h_RF, V_RF=V_RF,
                      dphi_RF=0, p_increment=0.0,
                      p0=p0, charge=qe, mass=m_p)

# Create beam
intensity = 1.2e11
epsn_x = 2e-6   # normalised horizontal emittance
epsn_y = 2e-6   # normalised vertical emittance
sigma_z = 1e-9 * machine.beta * c_light / 4.   # RMS bunch length in meters

bunch = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles=macroparticlenumber,
    intensity=intensity,
    epsn_x=epsn_x,
    epsn_y=epsn_y,
    sigma_z=sigma_z,
)

sx = np.sqrt(epsn_x * beta_x / machine.gamma / machine.beta)
sy = np.sqrt(epsn_y * beta_y / machine.gamma / machine.beta)

# Aperture
n_sigma_aper = 1.5

aperture = EllipticalApertureXY(x_aper=n_sigma_aper*sx, y_aper=n_sigma_aper*sy)

machine.one_turn_map.append(aperture)

# Particle monitor
monitor = ParticleMonitor(filename=outputpath+'/particlemonitor')

# Create arrays for saving
x = np.zeros(n_turns, dtype=float)
y = np.zeros(n_turns, dtype=float)
n_mp = np.zeros(n_turns, dtype=float)

# Tracking loop
time_0 = time.time()
for i in range(n_turns):

    if i % 10 == 0:
        print('Turn {:d}/{:d}'.format(i, n_turns))

    for m in machine.one_turn_map:
        m.track(bunch)
        monitor.dump(bunch)
        
    x[i], y[i] = bunch.mean_x(), bunch.mean_y()
    n_mp[i] = bunch.macroparticlenumber

print('\n*** Successfully completed!')
print(f"Time for tracking: {time.time() - time_0} s")

# Get data from monitor file
particle_file = h5py.File(outputpath+'/particlemonitor.h5part')

turns_to_plot = list(range(0, 7, 2)) + [n_turns-1]
n_steps_per_turn = len(machine.one_turn_map)
x_parts = []
y_parts = []

for turn in turns_to_plot:
    step = n_steps_per_turn*(turn + 1) - 2  # Take step before aperture at each turn
    x_parts.append(particle_file[f'Step#{step:.0f}']['x'][:])
    y_parts.append(particle_file[f'Step#{step:.0f}']['y'][:])

# Plot results
turns = np.arange(n_turns)

print(f"Number of particles at last turn from bunch {n_mp[-2]:.0f}, " +
      f"from particle monitor {len(x_parts[-1]):.0f}")

print(f"Max x: {max(abs(x_parts[-1]))}, max y: {max(abs(y_parts[-1]))}")
print(f"Aperture x: {n_sigma_aper*sx}, aperture y: {n_sigma_aper*sy}")

plt.close('all')

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(turns, n_mp, '.-')
ax1.set_xlabel("Turn")
ax1.set_ylabel("Number of macroparticles")

for ii, turn in enumerate(turns_to_plot):
    ax2.plot(x_parts[ii]*1e3, y_parts[ii]*1e3, 'x',
             label=f"Turn {turn:.0f}")
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("y [mm]")
ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(-0.8, 0.8)
ax2.legend()

fig.suptitle(f"{machine_name} {p0_eVperc*1e-12:.1f} TeV/c")
fig.subplots_adjust(left=0.08, right=0.95, wspace=0.25)

plt.show()

h5py.File.close(particle_file)

