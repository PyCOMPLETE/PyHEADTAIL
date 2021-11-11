import time

from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.stats import linregress
from scipy.signal import hilbert

from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.trackers.detuners import ChromaticitySegment
from PyHEADTAIL.trackers.detuners import AmplitudeDetuningSegment
from PyHEADTAIL.machines.synchrotron import Synchrotron


nTurn = 5000
n_macroparticles = int(1e4)

# Machine parameters
machine_name = 'LHC'
energy = 6.5e12  # [eV]
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
voltage = 12.0e6
h = 35640
Q_s = np.sqrt(qe * voltage * eta * h / (2 * np.pi * betar * c_light * p0))

circumference = 26658.883199999
average_radius = circumference / (2 * np.pi)

sigma_z = 1.2e-9 / 4.0 * c_light
sigma_delta = Q_s * sigma_z / (average_radius * eta)
beta_s = sigma_z / sigma_delta
emit_s = 4 * np.pi * sigma_z * sigma_delta * p0 / qe  # eVs for PyHEADTAIL

bunch_intensity = 1.8e11
normemit = 1.8e-6

# Wake field
n_slices_wakes = 500
limit_z = 3 * sigma_z
slicer_for_wakefields = UniformBinSlicer(n_slices_wakes,
                                         z_cuts=(-limit_z, limit_z))

wakefile = ('wakes/wakeforhdtl_PyZbase_Allthemachine_7000GeV'
            '_B1_2021_TeleIndex1_wake.dat')

waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y',
                                 'quadrupole_x', 'quadrupole_y'])

wake_field = WakeField(slicer_for_wakefields, waketable)

# Damper
damping_time = 7000
damper = TransverseDamper(dampingrate_x=damping_time,
                          dampingrate_y=damping_time)

# Detuners
Qp_x = -5.0
Qp_y = 0.0
i_oct = 15.
detx_x = 1.4e5 * i_oct / 550.0
detx_y = -1.0e5 * i_oct / 550.0

# Create particles
print("\nCreate particles")
particles = generate_Gaussian6DTwiss(
    macroparticlenumber=n_macroparticles, intensity=bunch_intensity,
    charge=qe, mass=m_p, circumference=circumference, gamma=gamma,
    alpha_x=0.0, alpha_y=0.0, beta_x=beta_x, beta_y=beta_y, beta_z=beta_s,
    epsn_x=normemit, epsn_y=normemit, epsn_z=emit_s,
)

print("--> Bunch length and emittance: {:g} m, {:g} eVs.".format(
    particles.sigma_z(), particles.epsn_z()))

print(f"sigma x: from particles {particles.sigma_x():.3e}, "
      f"expected {np.sqrt(normemit * beta_x / gamma / betar):.3e}")
print(f"sigma y: from particles {particles.sigma_y():.3e}, "
      f"expected {np.sqrt(normemit * beta_y / gamma / betar):.3e}")
print(f"sigma z: from particles {particles.sigma_z():.3e}, "
      f"expected {sigma_z:.3e}")
print(f"delta: from particles {particles.sigma_dp():.3e}, "
      f"expected {sigma_delta:.3e}")

coords = {
    'x': particles.x, 'xp': particles.xp,
    'y': particles.y, 'yp': particles.yp,
    'z': particles.z, 'dp': particles.dp,
}

particles_sy = Particles(macroparticlenumber=particles.macroparticlenumber,
                         particlenumber_per_mp=particles.particlenumber_per_mp,
                         charge=particles.charge, mass=particles.mass,
                         circumference=particles.circumference,
                         gamma=particles.gamma,
                         coords_n_momenta_dict=coords)

# Create segments
chromatic_detuner = ChromaticitySegment(dQp_x=Qp_x, dQp_y=Qp_y)

transverse_detuner = AmplitudeDetuningSegment(
    dapp_x=detx_x * p0, dapp_y=detx_x * p0,
    dapp_xy=detx_y * p0, dapp_yx=detx_y * p0,
    alpha_x=0.0, beta_x=beta_x,
    alpha_y=0.0, beta_y=beta_y,
)
arc_transverse = TransverseSegmentMap(
    alpha_x_s0=0.0, beta_x_s0=beta_x, D_x_s0=0.0,
    alpha_x_s1=0.0, beta_x_s1=beta_x, D_x_s1=0.0,
    alpha_y_s0=0.0, beta_y_s0=beta_y, D_y_s0=0.0,
    alpha_y_s1=0.0, beta_y_s1=beta_y, D_y_s1=0.0,
    dQ_x=Q_x, dQ_y=Q_y,
    segment_detuners=[chromatic_detuner, transverse_detuner],
)
arc_longitudinal = LinearMap(
    alpha_array=[alpha_mom], circumference=circumference, Q_s=Q_s
)

# Create synchrotron
machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                      n_segments=1,
                      alpha_x=0.0, beta_x=beta_x, D_x=0.0,
                      alpha_y=0.0, beta_y=beta_y, D_y=0.0,
                      accQ_x=Q_x, accQ_y=Q_y, Qp_x=Qp_x, Qp_y=Qp_y,
                      app_x=detx_x * p0, app_y=detx_x * p0, app_xy=detx_y * p0,
                      alpha_mom_compaction=alpha_mom,
                      longitudinal_mode='linear', Q_s=Q_s,
                      dphi_RF=np.atleast_1d(0.0), p_increment=0.0,
                      p0=p0, charge=qe, mass=m_p, RF_at='end_of_transverse')

machine.one_turn_map.append(wake_field)
machine.one_turn_map.append(damper)

# Tracking loop
print('\nTracking...')
x = np.zeros(nTurn, dtype=float)
x_sy = np.zeros(nTurn, dtype=float)

for turn in range(nTurn):

    time0 = time.time()
    arc_transverse.track(particles)
    arc_longitudinal.track(particles)
    wake_field.track(particles)
    damper.track(particles)
    time3 = time.time()
    x[turn] = np.average(particles.x)

    time4 = time.time()
    machine.track(particles_sy)
    time5 = time.time()
    x_sy[turn] = np.average(particles_sy.x)

    if turn % 1000 == 0:
        print(f"Turn {turn} - Segments: {(time3-time0)*1e3:.2f} ms, "
              f"Synchrotron {(time5-time4)*1e3:.2f} ms")

# Plot results
turns = np.arange(nTurn)
iMin = 500
iMax = nTurn - 500

plt.figure(0)

plt.plot(turns, x, label='Segments')
plt.plot(turns, x_sy, '--', label='Synchrotron')

ampl = np.abs(hilbert(x))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
plt.plot(turns, np.exp(a + b * turns), "-.k",
         label=f"Rise time seg: {1/b:.0f} turns")
print(f"\nGrowth rate with segments {b*1e4:.2f} [10^-4/turn]")

ampl = np.abs(hilbert(x_sy))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
plt.plot(turns, np.exp(a + b * turns), "--", color='0.5',
         label=f"Rise time syn: {1/b:.0f} turns")
print(f"Growth rate with synchrotron {b*1e4:.2f} [10^-4/turn]")

plt.title(f"{machine_name} {energy*1e-12:.1f} TeV")
plt.legend()
plt.xlabel("Turn")
plt.ylabel("x [m]")

plt.show()
