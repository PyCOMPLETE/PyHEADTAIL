import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.constants import c as c_light
from scipy.signal import find_peaks_cwt

from LHC import LHC


macroparticlenumber_track = 5000
macroparticlenumber_optics = 200000

n_turns = 10000

epsn_x  = 2.5e-6
epsn_y  = 3.5e-6
sigma_z = 0.6e-9 / 4.0 * c_light

intensity = 1e11

# Create machine using twiss parameters from optics pickle
with open('lhc_2015_80cm_optics.pkl', 'rb') as fid:
    optics = pickle.load(fid, encoding='latin1')
optics.pop('circumference')
optics.pop('part')
optics.pop('L_interaction')

machine = LHC(machine_configuration='6.5_TeV_collision_tunes',
              optics_mode = 'non-smooth', V_RF=10e6,
              **optics)
print(f'Synchrotron tune: {machine.Q_s}')

# Create bunch for optics test
print('Create bunch for optics...')
bunch_optics = machine.generate_6D_Gaussian_bunch_matched(
    macroparticlenumber_optics,
    intensity, epsn_x, epsn_y,
    sigma_z=sigma_z)

print('Done.')

# Kick bunch
bunch_optics.x += 10.
bunch_optics.y += 20.
bunch_optics.z += .020

# Temporarily remove longitudinal map
ix = machine.one_turn_map.index(machine.longitudinal_map)
machine.one_turn_map.remove(machine.longitudinal_map)

# Lists for saving
beam_alpha_x = []
beam_beta_x = []
beam_alpha_y = []
beam_beta_y = []

# Track through optics elements
for i_ele, m in enumerate(machine.one_turn_map):
    if i_ele % 10 == 0:
        print('Element {}/{}'.format(i_ele, len(machine.one_turn_map)))
    beam_alpha_x.append(bunch_optics.alpha_Twiss_x())
    beam_beta_x.append(bunch_optics.beta_Twiss_x())
    beam_alpha_y.append(bunch_optics.alpha_Twiss_y())
    beam_beta_y.append(bunch_optics.beta_Twiss_y())
    m.track(bunch_optics)

# Plot optics
plt.close('all')

fig, axes = plt.subplots(2, sharex=True, figsize=(10, 6))

axes[0].plot(np.array(beam_beta_x), 'bo')
axes[0].plot(machine.transverse_map.beta_x, 'b-', label='x')
axes[0].plot(np.array(beam_beta_y), 'ro')
axes[0].plot(machine.transverse_map.beta_y, 'r-', label='y')
axes[0].grid('on')
axes[0].set_ylabel('beta_x, beta_y')
axes[0].legend(prop={'size':12})

axes[1].plot(np.array(beam_alpha_x), 'bo')
axes[1].plot(machine.transverse_map.alpha_x, 'b-')
axes[1].plot(np.array(beam_alpha_y), 'ro')
axes[1].plot(machine.transverse_map.alpha_y, 'r-')
axes[1].grid('on')
axes[1].set_ylabel('alpha_x, alpha_y')
axes[1].set_xlabel('# point')

axes[0].plot(np.array(optics['beta_x']), 'xk')
axes[0].plot(np.array(optics['beta_y']), 'xk')
axes[1].plot(np.array(optics['alpha_x']), 'xk')
axes[1].plot(np.array(optics['alpha_y']), 'xk')

plt.subplots_adjust(left=0.1, right=0.9)


machine.one_turn_map.insert(ix, machine.longitudinal_map)

# Create bunch for tracking
print('Create bunch for tracking...')
bunch = machine.generate_6D_Gaussian_bunch_matched(
    macroparticlenumber_track, intensity, epsn_x, epsn_y, sigma_z=sigma_z)
print('Done.')

# Lists for saving
beam_x = []
beam_y = []
beam_z = []
sx, sy, sz = [], [], []
epsx, epsy, epsz = [], [], []

# Tracking loop
print(f'Track for {n_turns} turns')
for i_turn in range(n_turns):

    if i_turn % 100 == 0:
        print('Turn {}/{}'.format(i_turn, n_turns))

    machine.track(bunch)

    beam_x.append(bunch.mean_x())
    beam_y.append(bunch.mean_y())
    beam_z.append(bunch.mean_z())
    sx.append(bunch.sigma_x())
    sy.append(bunch.sigma_y())
    sz.append(bunch.sigma_z())
    epsx.append(bunch.epsn_x()*1e6)
    epsy.append(bunch.epsn_y()*1e6)
    epsz.append(bunch.epsn_z())

# Find tunes
freq_x = np.fft.rfftfreq(len(beam_x), d=1.)
ampl_x = np.abs(np.fft.rfft(beam_x))
ind_peaks_x = find_peaks_cwt(ampl_x, 5)
ind_max_peak_x = np.argmax(ampl_x[ind_peaks_x])
f_peak_x = freq_x[ind_peaks_x[ind_max_peak_x]]
print(f'Q_x found at {f_peak_x:.2f}')

freq_y = np.fft.rfftfreq(len(beam_y), d=1.)
ampl_y = np.abs(np.fft.rfft(beam_y))
ind_peaks_y = find_peaks_cwt(ampl_y, 5)
ind_max_peak_y = np.argmax(ampl_y[ind_peaks_y])
f_peak_y = freq_y[ind_peaks_y[ind_max_peak_y]]
print(f'Q_y found at {f_peak_y:.2f}')

freq_z = np.fft.rfftfreq(len(beam_z), d=1.)
ampl_z = np.abs(np.fft.rfft(beam_z))
ind_peaks_z = find_peaks_cwt(ampl_z, 5)
ind_max_peak_z = np.argmax(ampl_z[ind_peaks_z])
f_peak_z = freq_z[ind_peaks_z[ind_max_peak_z]]
print(f'Q_s found at {f_peak_z:.4f}')

# Plot mean positions and tunes
plt.figure(2, figsize=(16, 8), tight_layout=True)

plt.subplot(2, 3, 1)
plt.plot(beam_x)
plt.ylabel('x [m]')
plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

plt.subplot(2, 3, 2)
plt.plot(beam_y)
plt.ylabel('y [m]')
plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

plt.subplot(2, 3, 3)
plt.plot(beam_z)
plt.ylabel('z [m]')
plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

plt.subplot(2, 3, 4)
plt.plot(freq_x, ampl_x)
plt.ylabel('Amplitude')
plt.xlabel('Qx')

plt.subplot(2, 3, 5)
plt.plot(freq_y, ampl_y)
plt.ylabel('Amplitude')
plt.xlabel('Qy')

plt.subplot(2, 3, 6)
plt.plot(np.fft.rfftfreq(len(beam_z), d=1.), np.abs(np.fft.rfft(beam_z)))
plt.xlim(0, 0.1)
plt.ylabel('Amplitude')
plt.xlabel('Qz')

# Plot positions and emittances
fig, axes = plt.subplots(3, figsize=(16, 8), tight_layout=True)
twax = [plt.twinx(ax) for ax in axes]
axes[0].plot(sx, label=r'$\sigma_x$' )
twax[0].plot(epsx, '-g', label=r'$\varepsilon_x$')
axes[0].set_xlabel('Turns')
axes[0].set_ylabel(r'$\sigma_x$')
twax[0].set_ylabel(r'$\varepsilon_x$')

axes[1].plot(sy, label=r'$\sigma_y$' )
twax[1].plot(epsy, '-g', label=r'$\varepsilon_y$')
axes[1].set_xlabel('Turns')
axes[1].set_ylabel(r'$\sigma_y$')
twax[1].set_ylabel(r'$\varepsilon_y$')

axes[2].plot(sz, label=r'$\sigma_z$' )
twax[2].plot(epsz, '-g', label=r'$\varepsilon_z$')
axes[2].set_xlabel('Turns')
axes[2].set_ylabel(r'$\sigma_z$')
twax[2].set_ylabel(r'$\varepsilon_z$')

axes[0].grid()
axes[1].grid()
axes[2].grid()
for ax in list(axes)+list(twax): 
    ax.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
for ax in list(axes): 
    ax.legend(loc='upper right',prop={'size':12})
for ax in list(twax): 
    ax.legend(loc='lower right',prop={'size':12})

plt.show()
