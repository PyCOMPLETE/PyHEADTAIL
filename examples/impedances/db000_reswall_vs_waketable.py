import pathlib
import time

from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import e as qe, c as c_light, m_p
from scipy.signal import hilbert
from scipy.stats import linregress

from PyHEADTAIL.impedances import wakes
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.particles import Particles


n_turns = 1000
macroparticlenumber = int(1e5)

# Machine parameters
machine_name = 'PS'
E_kin = 1.4e9  # [eV]
E_rest = m_p * c_light**2 / qe  # [eV]
gamma = 1 + E_kin/E_rest
betagamma = np.sqrt(gamma**2 - 1)
p0 = betagamma * m_p * c_light

circumference = 2*np.pi*100

Q_x = 6.22
Q_y = 6.25

xi_x = -0.1
xi_y = -0.1

Qp_x = Q_x * xi_x
Qp_y = Q_y * xi_y

beta_x = 16.  # circumference/(2*np.pi*Q_x)
beta_y = 16.  # circumference/(2*np.pi*Q_y)

alpha_mom = 0.027

# Non-linear map
h_RF = 7
V_RF = 20e3
dphi_RF = np.pi
p_increment = 0.0

# Beam parameters
intensity = 1.6e+12
epsn_x = 2e-6   # Normalised horizontal emittance [m]
epsn_y = 2e-6   # Normalised vertical emittance [m]
sigma_z = 12.4  # [m]

# Create machine
machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                      n_segments=1, beta_x=beta_x, beta_y=beta_y,
                      D_x=0.0, D_y=0.0,
                      accQ_x=Q_x, accQ_y=Q_y, Qp_x=Qp_x, Qp_y=Qp_y,
                      alpha_mom_compaction=alpha_mom,
                      longitudinal_mode='non-linear', h_RF=h_RF, V_RF=V_RF,
                      dphi_RF=dphi_RF, p_increment=p_increment,
                      p0=p0, charge=qe, mass=m_p)

# Create particles
bunch_rw = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles=macroparticlenumber,
    intensity=intensity,
    epsn_x=epsn_x,
    epsn_y=epsn_y,
    sigma_z=sigma_z,
)

print('momentum spread =', bunch_rw.sigma_dp())
print('synchrotron tune = ', machine.longitudinal_map.Q_s)

sy = np.sqrt(epsn_y * beta_y / betagamma)

# Copy particles
coords = {'x': bunch_rw.x, 'xp': bunch_rw.xp,
          'y': bunch_rw.y, 'yp': bunch_rw.yp,
          'z': bunch_rw.z, 'dp': bunch_rw.dp}

bunch_wt = Particles(macroparticlenumber=bunch_rw.macroparticlenumber,
                     particlenumber_per_mp=bunch_rw.particlenumber_per_mp,
                     charge=bunch_rw.charge, mass=bunch_rw.mass,
                     circumference=bunch_rw.circumference,
                     gamma=bunch_rw.gamma,
                     coords_n_momenta_dict=coords)

# Create wakes
slices_for_wake = 500
slicer_for_wake = UniformBinSlicer(slices_for_wake, n_sigma_z=2)

# Resistive wall wake
conductivity = 1e6
pipe_sigma_y = 0.03565   # sigma_y * 8.
wake = wakes.ParallelHorizontalPlatesResistiveWall(
    conductivity=conductivity, pipe_radius=pipe_sigma_y,
    resistive_wall_length=10000, dt_min=1e-12)
wake_field_rw = wakes.WakeField(slicer_for_wake, wake)

# Wake table IW2D
wakefile_columns = ['time', 'dipole_x', 'dipole_y',
                    'quadrupole_x', 'quadrupole_y']
wake_folder = pathlib.Path(__file__).parent.joinpath('./').absolute()
wakefile = wake_folder.joinpath("wakes/wake_PyHT.txt")
wake_table = wakes.WakeTable(wakefile, wakefile_columns)
wake_field_wt = wakes.WakeField(slicer_for_wake, wake_table)

# Arrays for saving
y_rw = np.zeros(n_turns, dtype=float)
y_wt = np.zeros(n_turns, dtype=float)

# Tracking loop
time_0 = time.time()
for i in range(n_turns):

    if i % 100 == 0:
        print('Turn {:d}/{:d}'.format(i, n_turns))

    for m in machine.one_turn_map:
        m.track(bunch_rw)
        m.track(bunch_wt)

    wake_field_rw.track(bunch_rw)
    wake_field_wt.track(bunch_wt)

    y_rw[i] = bunch_rw.mean_y()
    y_wt[i] = bunch_wt.mean_y()

print('\n*** Successfully completed!')
print(f"Time for tracking: {time.time() - time_0} s")

# Plot results
turns = np.arange(n_turns)

plt.close('all')

plt.figure(0)

plt.plot(turns, y_wt/sy, label="WT")
plt.plot(turns, y_rw/sy, '--', label="RW")

iMin = 100
iMax = n_turns
ampl_wt = np.abs(hilbert(y_wt))
b_wt, a_wt, r_wt, p_wt, stderr_wt = linregress(turns[iMin:iMax],
                                               np.log(ampl_wt[iMin:iMax]))
print(f"Growth rate WT {b_wt*1e4:.2f} [10^-4/turn]")

plt.plot(turns, np.exp(a_wt + b_wt * turns)/sy, "--", color='0.3',
         label=f"WT rise time: {1/b_wt:.1f} turns")

ampl_rw = np.abs(hilbert(y_rw))
b_rw, a_rw, r_Rw, p_rw, stderr_rw = linregress(turns[iMin:iMax],
                                               np.log(ampl_rw[iMin:iMax]))
print(f"Growth rate RW {b_rw*1e4:.2f} [10^-4/turn]")

plt.plot(turns, np.exp(a_rw + b_rw * turns)/sy, "-.k",
         label=f"RW rise time: {1/b_rw:.1f} turns")

plt.legend(loc="upper left")
plt.xlabel("Turn")
plt.ylabel(r"y [$\sigma_y$]")
plt.title(f"{machine_name} {E_kin*1e-9:.1f} GeV")

plt.subplots_adjust(left=0.15)

plt.show()
