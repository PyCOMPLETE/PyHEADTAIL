import sys, os
BIN=os.path.expanduser('../../../')
sys.path.append(BIN)

from scipy.constants import e,c
from PyHEADTAIL.radiation.radiation import SynchrotronRadiationTransverse, SynchrotronRadiationLongitudinal
import numpy as np
import matplotlib.pyplot as plt

macroparticlenumber = 50000
n_turns = 512*8

import pickle

# MACHINE
# =======
from CLIC_DR import CLIC_DR
machine = CLIC_DR(machine_configuration='3TeV_linear', n_segments=1)

# BEAM
# ====
epsn_x  = 5*0.456e-6
epsn_y  = 5*0.0048e-6
sigma_z = 5*1.8e-3

intensity = 4.1e9
bunch   = machine.generate_6D_Gaussian_bunch(
    macroparticlenumber, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

bunch.x  += 1.e-6
bunch.xp += 1.e-6
bunch.y  += 1.e-6
bunch.yp += 1.e-6
bunch.z  += 0.02

# SYNCHROTRON RADIATION
# =====================
damping_time_z_turns = (1e-3*(c/machine.circumference))
E_loss_eV = 3.98e6
eq_sig_dp=1.074e-3
SynchrotronRadiationLongitudinal = SynchrotronRadiationLongitudinal(
	eq_sig_dp=eq_sig_dp, damping_time_z_turns=damping_time_z_turns, E_loss_eV=E_loss_eV)
dp_before = bunch.mean_dp()
SynchrotronRadiationLongitudinal.track(bunch)
dp_after = bunch.mean_dp()

print 'Energy loss\nEvaluated :%.6e [eV]\nExpected :%.6e [eV]\nERROR :%.2f'%((dp_before-dp_after)*machine.p0*c/np.abs(machine.charge),
		E_loss_eV,(E_loss_eV-((dp_before-dp_after)*machine.p0*c/np.abs(machine.charge)))*100/E_loss_eV)+'%'

