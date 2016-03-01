from __future__ import division

import sys, os
BIN=os.path.expanduser('../../../')
sys.path.append(BIN)

from scipy.constants import e,c
from PyHEADTAIL.radiation.radiation import SynchrotronRadiationTransverse, SynchrotronRadiationLongitudinal
import numpy as np
import matplotlib.pyplot as plt

macroparticlenumber = 50000
n_turns = 512*8
tt = np.arange(n_turns)

import pickle
import time

from CLIC_DR import CLIC_DR
machine = CLIC_DR(machine_configuration='3TeV', n_segments=1)
n_turns_vec = np.arange(0, n_turns, 1)*(machine.circumference/c)*1e3

# SYNCHROTRON RADIATION
# =====================
damping_time_z_turns = (1e-3*(c/machine.circumference))
E_loss_eV = 3.98e6
eq_sig_dp=1.074e-3
SynchrotronRadiationLongitudinal = SynchrotronRadiationLongitudinal(
	eq_sig_dp=eq_sig_dp, damping_time_z_turns=damping_time_z_turns, E_loss_eV=E_loss_eV)

# EVALUATE STABLE PHASE AND Z
# ===========================
phi_s = np.arcsin(E_loss_eV/machine.longitudinal_map.voltages[0])
z_s = (machine.circumference)*phi_s/(2*np.pi*machine.longitudinal_map.harmonics[0])
print z_s

# BEAM
# ====
epsn_x  = 5*0.456e-6
epsn_y  = 5*0.0048e-6
sigma_z = 1.8e-3

intensity = 4.1e9
bunch   = machine.generate_6D_Gaussian_bunch(
    macroparticlenumber, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

bunch.x  += 1.e-6
bunch.xp += 1.e-6
bunch.y  += 1.e-6
bunch.yp += 1.e-6
# bunch.z  *= 0.
bunch.dp *= 0.
bunch.z  += z_s

# TRACKING LOOP
# =============
machine.one_turn_map = [SynchrotronRadiationLongitudinal]+machine.one_turn_map

# Tracking
# ========
plt.close()
plt.ion()

print '--> Begin tracking...'

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(14,10))
        
sigma_x  = np.zeros(n_turns) + np.NAN
sigma_xp = np.zeros(n_turns) + np.NAN
mean_x   = np.zeros(n_turns) + np.NAN
epsn_x   = np.zeros(n_turns) + np.NAN
sigma_y  = np.zeros(n_turns) + np.NAN
sigma_yp = np.zeros(n_turns) + np.NAN
mean_y   = np.zeros(n_turns) + np.NAN
epsn_y   = np.zeros(n_turns) + np.NAN
sigma_z  = np.zeros(n_turns) + np.NAN
sigma_dp = np.zeros(n_turns) + np.NAN
mean_z   = np.zeros(n_turns) + np.NAN
epsn_z   = np.zeros(n_turns) + np.NAN
mean_dp   = np.zeros(n_turns) + np.NAN

for i in range(n_turns):

    machine.track(bunch)
    print 'Turn %d/%d'%(i, n_turns)
    sigma_x[i]  = bunch.sigma_x()
    mean_x[i]   = bunch.mean_x()
    epsn_x[i]   = bunch.epsn_x()
    sigma_y[i]  = bunch.sigma_y()
    mean_y[i]   = bunch.mean_y()
    epsn_y[i]   = bunch.epsn_y()
    sigma_z[i]  = bunch.sigma_z()
    sigma_dp[i] = bunch.sigma_dp()
    mean_z[i]   = bunch.mean_z()
    epsn_z[i]   = bunch.epsn_z()    
    mean_dp[i]   = bunch.mean_dp()    
        
    if not (i+1)%10:
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

        size_font = 15
        
        ax = ax1
        plt.rc("font", size=size_font)
        ax.set_xlim(0.02, 0.06)
        ax.set_xlabel('z', fontsize=size_font)
        ax.set_ylim(-1e-2, 1e-2)
        ax.set_ylabel('$\delta$', fontsize=size_font, rotation = 0)
        ax.scatter(bunch.z, bunch.dp, s=8, color = 'b')
        ax.scatter(bunch.z[50], bunch.dp[50], s=8, color = 'r')
        ax.scatter(bunch.z[51], bunch.dp[51], s=8, color = 'w')
        ax.ticklabel_format(style='sci', scilimits=(0,0),axis='both', useOffset = False)
        
        ax = ax2
        plt.rc("font", size=size_font)
        ax.set_xlim(0, n_turns_vec[-1])
        ax.set_xlabel('t [ms]', fontsize=size_font)
        ax.plot(n_turns_vec, mean_z)
	ax.axhline(z_s, label= 'Equilibrium z\nExpected :%.2e [m]'%(z_s), lw=2, color = 'red')
        ax.set_ylabel('$<z>$ [m]', fontsize=size_font)
        ax.ticklabel_format(style='sci', scilimits=(0,0),axis='y', useOffset = False)
	ax.legend (loc=0, fontsize = 10)
        
        ax = ax3
        plt.rc("font", size=size_font)
        ax.set_xlim(0, n_turns_vec[-1])
        ax.set_xlabel('t [ms]', fontsize=size_font)
        ax.plot(n_turns_vec, mean_dp)
        ax.set_ylabel('mean_dp', fontsize=size_font)
	ax.axhline(E_loss_eV/(machine.p0*c/np.abs(machine.charge))/2., 
		label= 'Equilibrium dp\nExpected :%.2e'%(E_loss_eV/(machine.p0*c/np.abs(machine.charge))/2.), lw=2, color = 'red')
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useOffset = False)  
	ax.legend (loc=0, fontsize = 10)        

        ax = ax4
        plt.rc("font", size=size_font)
        ax.set_xlim(0, n_turns_vec[-1])
        ax.set_xlabel('t [ms]', fontsize=size_font)
        ax.axhline(eq_sig_dp, label= 'Equilibrium momentum spread\nExpected :%.2e'%(eq_sig_dp), lw=2, color = 'red')
        ax.plot(n_turns_vec, sigma_dp)
        ax.legend (loc=0, fontsize = 10)
        ax.set_ylabel('$\sigma_{dp}$', fontsize=size_font)
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useOffset = False)
        
	plt.suptitle('Longitudinal phase space is samples "opposite" to the kicks (half drift occurred).')	

        #plt.savefig('mismatched_nonlinear_' + '%02d'%i + '.png', dpi=150)
        
	time.sleep(.1)
        plt.draw()


        '' 
        #print 'bunch.z and bunch.dp = ',bunch.z, bunch.dp
        #print 'mean_z = ',mean_z
        #print 'sigma_dp = ',sigma_dp
        #print 'sigma_z = ',sigma_z
        
print '--> Done.'



