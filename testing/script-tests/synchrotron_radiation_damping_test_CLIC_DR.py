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


# SYNCHROTRON RADIATION
# =====================
synchdamp_x = (2e-3*(c/machine.circumference))
synchdamp_y = (2e-3*(c/machine.circumference))
synchdamp_z = (1e-3*(c/machine.circumference))
E_loss = 3.98e6
eq_emit_x=0.456e-6
eq_emit_y=0.0048e-6
eq_sig_dp=1.074e-3
sync_rad_transverse = Sync_rad_transverse(eq_emit_x=eq_emit_x, eq_emit_y=eq_emit_y, 
	synchdamp_x=synchdamp_x, synchdamp_y=synchdamp_y, beta_x=machine.transverse_map.beta_x[-1], 
	beta_y=machine.transverse_map.beta_y[-1])
sync_rad_longitudinal = Sync_rad_longitudinal(
	eq_sig_dp=eq_sig_dp, synchdamp_z=synchdamp_z, E_loss=E_loss)

# TRACKING LOOP
# =============
machine.one_turn_map.append(sync_rad_transverse)
machine.one_turn_map.append(sync_rad_longitudinal)

beam_x = []
beam_y = []
beam_z = []
mean_dp = []
sx, sy, sz, sdp = [], [], [], []
epsx, epsy, epsz = [], [], []
for i_turn in xrange(n_turns):
    print 'Turn %d/%d'%(i_turn, n_turns)
    machine.track(bunch)

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

# PARAMETER EVALUATION
# ====================
tt = np.arange(n_turns)
xx_x = np.log(np.abs(beam_x))
p0_x, p1_x = np.polyfit(tt, xx_x, 1)
xx_y = np.log(np.abs(beam_y))
p0_y, p1_y = np.polyfit(tt, xx_y, 1)
mean_z_shift = (beam_z - beam_z[-1] + np.finfo(float).eps)
xx_z = np.log(np.abs(mean_z_shift))
p0_z, p1_z = np.polyfit(tt, xx_z, 1)

# PLOT
# ====

plt.figure(1, figsize=(16, 8), tight_layout=True)
plt.subplot(2,3,1)
plt.plot(beam_x)
plt.plot(2*np.exp(p1_x + tt * p0_x), label= 'Fitting curve \nDamping time:%.2e [s]'%((-1/p0_x)*(427.5/c)), lw=2, color = 'red')
plt.legend (loc=0, fontsize = 10)
plt.ylabel('x [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,2)
plt.plot(beam_y)
plt.plot(2*np.exp(p1_y + tt * p0_y), label= 'Fitting curve \nDamping time:%.2e [s]'%((-1/p0_y)*(427.5/c)), lw=2, color = 'red')
plt.legend (loc=0, fontsize = 10)
plt.ylabel('y [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,3)
plt.plot(beam_z)
plt.plot(2*np.exp(p1_z + tt * p0_z)+ beam_z[-1], label= 'Fitting curve \nDamping time:%.2e [s]'%((-1/p0_z)*(427.5/c)), lw=2, color = 'red')
plt.legend (loc=0, fontsize = 10)
plt.ylabel('z [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,4)
plt.plot(epsx)
plt.axhline(eq_emit_x, label= 'Expected equilibrium emittance:%.2e [m.rad]'%(eq_emit_x), lw=2, color = 'red')
plt.legend (loc=0, fontsize = 10)
plt.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
plt.ylabel('$\epsilon_x$ [m.rad]');plt.xlabel('Turn')
plt.subplot(2,3,5)
plt.plot(epsy)
plt.axhline(eq_emit_y, label= 'Expected equilibrium emittance:%.2e'%(eq_emit_y), lw=2, color = 'red')
plt.legend (loc=0, fontsize = 10)
plt.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
plt.ylabel('$\epsilon_y$ [m.rad]');plt.xlabel('Turn')
plt.subplot(2,3,6)
plt.plot(sdp)
plt.axhline(eq_sig_dp, label= 'Expected equilibrium momentum spread:%.2e'%(eq_sig_dp), lw=2, color = 'red')
plt.legend (loc=0, fontsize = 10)
plt.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
plt.ylabel('$\sigma_{dp}$');plt.xlabel('Turn')

plt.show()
