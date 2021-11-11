import time

from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.signal import hilbert
from scipy.stats import linregress

from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.machines.synchrotron import Synchrotron


n_turns = 800
macroparticlenumber = int(1e5)

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
intensity = 2e11
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

sx = np.sqrt(epsn_x * beta_x / machine.betagamma)
sy = np.sqrt(epsn_y * beta_y / machine.betagamma)


# Damper
damping_time = 200  # [turns]
damper = TransverseDamper(dampingrate_x=damping_time,
                          dampingrate_y=damping_time)

machine.one_turn_map.append(damper)

kick_in_sigmas = 0.75
bunch.x += kick_in_sigmas * bunch.sigma_x()
bunch.y += kick_in_sigmas * bunch.sigma_y()

# Create arrays for saving
x = np.zeros(n_turns, dtype=float)
y = np.zeros(n_turns, dtype=float)

# Tracking loop
time_0 = time.time()
for i in range(n_turns):

    if i % 100 == 0:
        print('Turn {:d}/{:d}'.format(i, n_turns))

    for m in machine.one_turn_map:
        m.track(bunch)

    x[i], y[i] = bunch.mean_x(), bunch.mean_y()

print('\n*** Successfully completed!')
print(f"Time for tracking: {time.time() - time_0} s")

# Plot results
turns = np.arange(n_turns)

plt.close('all')

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(turns, x/sx)
ax2.plot(turns, y/sy)

iMin = 5
iMax = n_turns - 5

ampl_x = np.abs(hilbert(x))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl_x[iMin:iMax]))
print(f"Damping time x {-1/b:.0F} [turns]")
ax1.plot(turns, np.exp(a + b * turns)/sx, "--k", label=f"{-1/b:.0F} turns")
ax1.legend()
ax1.set_xlabel("Turn")
ax1.set_ylabel(r"x [$\sigma_x$]")

ampl_y = np.abs(hilbert(y))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl_y[iMin:iMax]))
print(f"Damping time y {-1/b:.0F} [turns]")
ax2.plot(turns, np.exp(a + b * turns)/sy, "--k", label=f"{-1/b:.0F} turns")
ax2.legend()
ax2.set_xlabel("Turn")
ax2.set_ylabel(r"y [$\sigma_y$]")

fig.suptitle(f"{machine_name} {p0_eVperc*1e-12:.1f} TeV/c")
fig.subplots_adjust(left=0.08, right=0.95, wspace=0.25)

plt.show()
