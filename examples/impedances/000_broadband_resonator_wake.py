import numpy as np
import time

from matplotlib import pyplot as plt
from scipy.constants import c as c_light
from scipy.signal import hilbert
from scipy.stats import linregress

from PyHEADTAIL.impedances import wakes
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from LHC import LHC


n_turns = 3000
macroparticlenumber = int(1e5)


# Create machine
machine = LHC(n_segments=1, machine_configuration='6.5_TeV_collision')


# Create beam
intensity = 6e11
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


# Create BB resonator wake
# Resonator parameters
R_shunt = 25e6   # Shunt impedance [Ohm/m]
frequency = 2e9   # Resonance frequency [Hz]
Q = 1   # Quality factor

slices = 200
slicer_for_wakes = UniformBinSlicer(slices, n_sigma_z=6)

# Wake
wake = wakes.CircularResonator(R_shunt, frequency, Q)
wake_field = wakes.WakeField(slicer_for_wakes, wake)

machine.one_turn_map.append(wake_field)


# Create arrays for saving
sx = np.zeros(n_turns, dtype=float)
sy = np.zeros(n_turns, dtype=float)

# Tracking loop
time_0 = time.time()
for i in range(n_turns):

    if i % 100 == 0:
        print('Turn {:d}/{:d}'.format(i, n_turns))

    for m in machine.one_turn_map:
        m.track(bunch)

    sx[i], sy[i] = bunch.mean_x(), bunch.mean_y()

print('\n*** Successfully completed!')
print(f"Time for tracking: {time.time() - time_0} s")


# Plot results

turns = np.arange(n_turns)

plt.close('all')

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(turns, sx)
ax2.plot(turns, sy)

iMin = 1500
iMax = n_turns
if iMin >= iMax:
    iMin = 0

ampl_x = np.abs(hilbert(sx))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl_x[iMin:iMax]))
print(f"Growth rate x {b*1E4:.2F} [10^-4/turn]")
ax1.plot(turns, np.exp(a + b * turns), "--k", label=f"{1/b:.3E} turns")
ax1.legend(loc="upper left")
ax1.set_xlabel("Turn")
ax1.set_ylabel("x [m]")

ampl_y = np.abs(hilbert(sy))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl_y[iMin:iMax]))
print(f"Growth rate y {b*1E4:.2F} [10^-4/turn]")
ax2.plot(turns, np.exp(a + b * turns), "--k", label=f"{1/b:.3E} turns")
ax2.legend(loc="upper left")
ax2.set_xlabel("Turn")
ax2.set_ylabel("y [m]")

fig.suptitle("LHC 6.5 TeV")
fig.subplots_adjust(left=0.08, right=0.95, wspace=0.25)

plt.show()
