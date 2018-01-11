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


# SYNCHROTRON RADIATION
# =====================
damping_time_x_turns = (2e-3*(c/machine.circumference))
damping_time_y_turns = (2e-3*(c/machine.circumference))
damping_time_z_turns = (1e-3*(c/machine.circumference))
E_loss_eV = 3.98e6
eq_emit_x=0.456e-6
eq_emit_y=0.0048e-6
eq_sig_dp=1.074e-3
SynchrotronRadiationTransverse = SynchrotronRadiationTransverse(eq_emit_x=eq_emit_x, eq_emit_y=eq_emit_y, 
	damping_time_x_turns=damping_time_x_turns, damping_time_y_turns=damping_time_y_turns, beta_x=machine.transverse_map.beta_x[-1], 
	beta_y=machine.transverse_map.beta_y[-1])
SynchrotronRadiationLongitudinal = SynchrotronRadiationLongitudinal(
	eq_sig_dp=eq_sig_dp, damping_time_z_turns=damping_time_z_turns, E_loss_eV=E_loss_eV)

# TRACKING LOOP
# =============
machine.one_turn_map.append(SynchrotronRadiationTransverse)
machine.one_turn_map.append(SynchrotronRadiationLongitudinal)

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

n_turns_damping = int(2*damping_time_x_turns) #number of turns to evaluate the damping time
n_intervals_damping = 10             #number of intervals to evaluate the damping time
interval_vector_damping = np.arange(0, (n_turns_damping), (n_turns_damping/n_intervals_damping))
N_interval_damping = len(interval_vector_damping)
xx_x = beam_x
xx_y = beam_y
xx_x_env_damping = []
xx_y_env_damping = []
for j in range(N_interval_damping): #for loop to evaluate the envelope of centroid position in x and y
    xx_x_env_damping.append(np.max(xx_x[((n_turns_damping/n_intervals_damping)*j):((n_turns_damping/n_intervals_damping)*(j+1))])) # take the maximum in each interval
    xx_y_env_damping.append(np.max(xx_y[((n_turns_damping/n_intervals_damping)*j):((n_turns_damping/n_intervals_damping)*(j+1))]))
xx_x_env_log_damping = np.log(np.abs(xx_x_env_damping) + sys.float_info.epsilon) #exponential fit using linear fit (exp->log)
p0_x, p1_x = np.polyfit(interval_vector_damping, xx_x_env_log_damping, 1)
xx_y_env_log_damping = np.log(np.abs(xx_y_env_damping) + sys.float_info.epsilon)
p0_y, p1_y = np.polyfit(interval_vector_damping, xx_y_env_log_damping, 1)

tt = np.arange(n_turns)
mean_z_shift = (beam_z - beam_z[-1] + np.finfo(float).eps)
xx_z = np.log(np.abs(mean_z_shift))
p0_z, p1_z = np.polyfit(tt, xx_z, 1)

# PLOT
# ====

plt.figure(1, figsize=(16, 8), tight_layout=True)
plt.subplot(2,3,1)
plt.plot(beam_x)
plt.plot(np.max(beam_x)*np.exp(-tt/damping_time_x_turns), label= 'Damping time\nExpected :%.2e [s]'%(damping_time_x_turns*(machine.circumference/c)), lw=2, color = 'black')
plt.plot(np.exp(p1_x + tt * p0_x), label= 'Evaluated :%.2e [s]\nERROR :%.2f'%((-1/p0_x)*(machine.circumference/c),np.abs((damping_time_x_turns)-(-1/p0_x))*100/(damping_time_x_turns))+'%', lw=2, color = 'red', linestyle = '--')
plt.legend (loc=0, fontsize = 10)
plt.ylabel('x [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,2)
plt.plot(beam_y)
plt.plot(np.max(beam_y)*np.exp(-tt/damping_time_y_turns), label= 'Damping time\nExpected :%.2e [s]'%(damping_time_y_turns*(machine.circumference/c)), lw=2, color = 'black')
plt.plot(np.exp(p1_y + tt * p0_y), label= 'Evaluated :%.2e [s]\nERROR :%.2f'%((-1/p0_y)*(machine.circumference/c),np.abs((damping_time_y_turns)-(-1/p0_y))*100/(damping_time_y_turns))+'%', lw=2, color = 'red', linestyle = '--')
plt.legend (loc=0, fontsize = 10)
plt.ylabel('y [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,3)
plt.plot(beam_z)
plt.plot((np.max(beam_z)-beam_z[-1])*np.exp(-tt/damping_time_z_turns)+beam_z[-1], label= 'Damping time\nExpected :%.2e [s]'%(damping_time_z_turns*(machine.circumference/c)), lw=2, color = 'black')
plt.plot(2*np.exp(p1_z + tt * p0_z)+ beam_z[-1], label= 'Evaluated :%.2e [s]\nERROR :%.2f'%((-1/p0_z)*(machine.circumference/c),np.abs((damping_time_z_turns)-(-1/p0_z))*100/(damping_time_z_turns))+'%', lw=2, color = 'red', linestyle = '--')
plt.legend (loc=0, fontsize = 10)
plt.ylabel('z [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,4)
plt.plot(epsx)
plt.axhline(eq_emit_x, label= 'Equilibrium emittance\nExpected :%.2e [s]'%(eq_emit_x), lw=2, color = 'black')
plt.axhline(epsx[-1], label= 'Evaluated :%.2e [s]\nERROR :%.2f'%(epsx[-1],np.abs(eq_emit_x-epsx[-1])*100/eq_emit_x)+'%', lw=2, color = 'red', linestyle = '--')
plt.legend (loc=0, fontsize = 10)
plt.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
plt.ylabel('$\epsilon_x$ [m.rad]');plt.xlabel('Turn')
plt.subplot(2,3,5)
plt.plot(epsy)
plt.axhline(eq_emit_y, label= 'Equilibrium emittance\nExpected :%.2e [s]'%(eq_emit_y), lw=2, color = 'black')
plt.axhline(epsy[-1], label= 'Evaluated :%.2e [s]\nERROR :%.2f'%(epsy[-1],np.abs(eq_emit_y-epsy[-1])*100/eq_emit_y)+'%', lw=2, color = 'red', linestyle = '--')
plt.legend (loc=0, fontsize = 10)
plt.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
plt.ylabel('$\epsilon_y$ [m.rad]');plt.xlabel('Turn')
plt.subplot(2,3,6)
plt.plot(sdp)
plt.axhline(eq_sig_dp, label= 'Equilibrium momentum spread\nExpected :%.2e'%(eq_sig_dp), lw=2, color = 'black')
plt.axhline(sdp[-1], label= 'Evaluated :%.2e\nERROR :%.2f'%(sdp[-1],np.abs(eq_sig_dp-sdp[-1])*100/eq_sig_dp)+'%', lw=2, color = 'red', linestyle = '--')
plt.legend (loc=0, fontsize = 10)
plt.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
plt.ylabel('$\sigma_{dp}$');plt.xlabel('Turn')

plt.show()
