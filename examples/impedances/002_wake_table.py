import time

from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.stats import linregress
from scipy.signal import hilbert

from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.particles.slicing import UniformBinSlicer


n_turns = 7000
n_macroparticles = int(1e4)

# Machine parameters
machine_name = 'LHC'
energy = 7e12  # [eV]
rest_energy = m_p * c_light**2 / qe  # [eV]
gamma = energy / rest_energy
betar = np.sqrt(1 - 1 / gamma ** 2)
p0 = m_p * betar * gamma * c_light

beta_x = 68.9
beta_y = 70.34

Q_x = 64.31
Q_y = 59.32

alpha_mom = 3.483575072011584e-04

eta = alpha_mom - 1.0 / gamma**2
V_RF = 12.0e6
h_RF = 35640
Q_s = np.sqrt(qe * V_RF * eta * h_RF / (2 * np.pi * betar * c_light * p0))

circumference = 26658.883199999

sigma_z = 1.2e-9 / 4.0 * c_light

bunch_intensity = 2e11
epsn = 1.8e-6

# Wake field
n_slices_wakes = 200
limit_z = 3 * sigma_z
slicer_for_wakefields = UniformBinSlicer(n_slices_wakes,
                                         z_cuts=(-limit_z, limit_z))

wakefile = ('wakes/wakeforhdtl_PyZbase_Allthemachine_7000GeV'
            '_B1_2021_TeleIndex1_wake.dat')

waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y',
                                 'quadrupole_x', 'quadrupole_y'])

wake_field = WakeField(slicer_for_wakefields, waketable)

# Damper
damping_time = 7000  # [turns]
damper = TransverseDamper(dampingrate_x=damping_time,
                          dampingrate_y=damping_time)

# Detuners
Qp_x = -5.0
Qp_y = 0.0
i_oct = 15.
detx_x = 1.4e5 * i_oct / 550.0
detx_y = -1.0e5 * i_oct / 550.0

# Create synchrotron
machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                      n_segments=1,
                      alpha_x=0.0, beta_x=beta_x, D_x=0.0,
                      alpha_y=0.0, beta_y=beta_y, D_y=0.0,
                      accQ_x=Q_x, accQ_y=Q_y, Qp_x=Qp_x, Qp_y=Qp_y,
                      app_x=detx_x * p0, app_y=detx_x * p0, app_xy=detx_y * p0,
                      alpha_mom_compaction=alpha_mom,
                      longitudinal_mode='linear', Q_s=Q_s,
                      dphi_RF=0.0, p_increment=0.0, p0=p0,
                      charge=qe, mass=m_p, RF_at='end_of_transverse')

machine.one_turn_map.append(wake_field)
machine.one_turn_map.append(damper)

particles = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles,
    intensity=bunch_intensity,
    epsn_x=epsn,
    epsn_y=epsn,
    sigma_z=sigma_z,
)

print("\n--> Bunch length and emittance: {:g} m, {:g} eVs.".format(
    particles.sigma_z(), particles.epsn_z()))

sx = np.sqrt(epsn * beta_x / gamma / betar)

# Array for saving
x = np.zeros(n_turns, dtype=float)

# Tracking loop
print('\nTracking...')

time_0 = time.time()
for turn in range(n_turns):

    if turn % 500 == 0:
        print('Turn {:d}/{:d}'.format(turn, n_turns))

    machine.track(particles)
    x[turn] = particles.mean_x()

print(f"Time for tracking: {time.time() - time_0} s")

turns = np.arange(n_turns)
iMin = 1000
iMax = n_turns - 1000

# Plot results
plt.figure(0)

plt.plot(turns, x/sx)

ampl = np.abs(hilbert(x))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
plt.plot(turns, np.exp(a + b * turns)/sx, "--k", label=f"{1/b:.2f} turns")
print(f"Growth rate {b*1e4:.2f} [10^-4/turn]")

plt.title(f"{machine_name} {energy*1e-12:.0f} TeV")
plt.legend()
plt.xlabel("Turn")
plt.ylabel(r"x [$\sigma_x$]")

plt.show()
