import numpy as np
import time

from matplotlib import pyplot as plt
from scipy.constants import e as qe, c as c_light, m_p
from scipy.signal import hilbert
from scipy.stats import linregress

from PyHEADTAIL.impedances import wakes
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.particles.slicing import UniformBinSlicer


n_turns = 1000
macroparticlenumber = int(1e5)

# Machine parameters
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

# Linear map
# Q_s = 1.24e-3

# Non-linear map
h_RF = 7
V_RF = 20e3
dphi_RF = np.pi
p_increment = 0.0

# Beam parameters
intensity = 1.6e+12
epsn_x = 2e-6   # Normalised horizontal emittance [m]
epsn_y = 2e6   # Normalised vertical emittance [m]
sigma_z = 12.4  # [m]

# Create machine
machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                      n_segments=1, beta_x=beta_x, beta_y=beta_y,
                      D_x=0.0, D_y=0.0,
                      accQ_x=Q_x, accQ_y=Q_y, Qp_x=Qp_x, Qp_y=Qp_y,
                      alpha_mom_compaction=alpha_mom,
                      # longitudinal_mode='linear', Q_s = Q_s,
                      longitudinal_mode='non-linear', h_RF=h_RF, V_RF=V_RF,
                      dphi_RF=dphi_RF, p_increment=p_increment,
                      p0=p0, charge=qe, mass=m_p)

# Create particles
bunch = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles=macroparticlenumber,
    intensity=intensity,
    epsn_x=epsn_x,
    epsn_y=epsn_y,
    sigma_z=sigma_z,
)

print('momentum spread = ', bunch.sigma_dp())
print('synchrotron tune = ', machine.longitudinal_map.Q_s)

# Create wakes
# ============
slices_for_wake = 500
slicer_for_wake = UniformBinSlicer(slices_for_wake, n_sigma_z=3)

# Resistive wall wake
conductivity = 1e6
pipe_sigma_y = 0.0356   # sigma_y * 8.
wake = wakes.ParallelHorizontalPlatesResistiveWall(
    conductivity=conductivity, pipe_radius=pipe_sigma_y,
    resistive_wall_length=10000, dt_min=1e-12)
wake_field = wakes.WakeField(slicer_for_wake, wake)

machine.one_turn_map.append(wake_field)


# Tracking loop
# =============

# Arrays for saving
sx = np.zeros(n_turns, dtype=float)
sy = np.zeros(n_turns, dtype=float)

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
# ============
turns = np.arange(n_turns)

plt.close('all')

plt.figure(0)
plt.plot(turns, sy)

iMin = 100
iMax = n_turns
if iMin >= iMax:
    iMin = 0
ampl = np.abs(hilbert(sy))
b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
print(f"Growth rate {b*1e4:.2f} [10^-4/turn]")

plt.plot(turns, np.exp(a + b * turns), "--k",
         label=f"{1/b:.1f} turns")

plt.legend(loc="upper left")
plt.xlabel("Turn")
plt.ylabel("y [m]")
plt.title(f"PS {E_kin*1e-9:.1f} GeV")

plt.subplots_adjust(left=0.15)

plt.show()
