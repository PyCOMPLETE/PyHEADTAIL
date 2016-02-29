import sys, os
BIN=os.path.expanduser('../../../')
sys.path.append(BIN)

from scipy.constants import e,c
from PyHEADTAIL.synch_rad.synch_rad import Sync_rad_transverse, Sync_rad_longitudinal
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
synchdamp_z = (1e-3*(c/machine.circumference))
E_loss = 3.98e6
eq_sig_dp=1.074e-3
sync_rad_longitudinal = Sync_rad_longitudinal(
	eq_sig_dp=eq_sig_dp, synchdamp_z=synchdamp_z, E_loss=E_loss)

# TRACKING LOOP
# =============
machine.one_turn_map.append(sync_rad_longitudinal)

beam_x = []
beam_y = []
beam_z = []
mean_dp = []
sx, sy, sz, sdp = [], [], [], []
epsx, epsy, epsz = [], [], []
for i_turn in xrange(n_turns):
    print 'Turn %d/%d'%(i_turn, n_turns)
    sync_rad_longitudinal.track(bunch)

    beam_x.append(bunch.mean_x())
    beam_y.append(bunch.mean_y())
    beam_z.append(bunch.mean_z())
    mean_dp.append(bunch.mean_dp())
    sx.append(bunch.sigma_x())
    sy.append(bunch.sigma_y())
    sz.append(bunch.sigma_z())
    sdp.append(bunch.sigma_dp())
    epsx.append(bunch.epsn_x())
    epsy.append(bunch.epsn_y())
    epsz.append(bunch.epsn_z())

# PLOT
# ====

plt.figure(1, figsize=(16, 8))
plt.plot(mean_dp, label= 'Energy loss\nEvaluated :%.2e [eV]\nExpected :%.2e [eV]\nERROR :%.2f'%((mean_dp[0]-mean_dp[1])*machine.p0*c/np.abs(machine.charge),E_loss,(E_loss-((mean_dp[0]-mean_dp[1])*machine.p0*c/np.abs(machine.charge)))*100/E_loss)+'%')
plt.legend (loc=0, fontsize = 10)
plt.ylabel('mean_dp');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')

plt.show()
