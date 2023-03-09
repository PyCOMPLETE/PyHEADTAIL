import time

from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.stats import linregress
from scipy.signal import hilbert

from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField, CircularResonator
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


R_shunt = 10.2e6   # Shunt impedance [Ohm/m]
frequency = 1.3e9   # Resonance frequency [Hz]
Q = 1   # Quality factor


# Wake
wake_res = CircularResonator(R_shunt, frequency, Q)

wake_field = WakeField(slicer_for_wakefields, [waketable, wake_res])

