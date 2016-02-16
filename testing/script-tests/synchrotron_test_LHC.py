import sys, os
BIN=os.path.expanduser('../../../')
sys.path.append(BIN)


from scipy.constants import e,c

macroparticlenumber_track = 50000
macroparticlenumber_optics = 2000000
n_turns = 512*4

epsn_x  = 2.5e-6
epsn_y  = 3.5e-6
sigma_z = 0.05

intensity = 1e11

mode = 'smooth'
#mode = 'non-smooth'

import pickle
from LHC import LHC

if mode == 'smooth':
    machine = LHC(machine_configuration='Injection', n_segments=29, D_x=10)
elif mode == 'non-smooth':
    with open('lhc_2015_80cm_optics.pkl') as fid:
        optics = pickle.load(fid)
    optics.pop('circumference')

    machine = LHC(machine_configuration='Injection', optics_mode = 'non-smooth', V_RF=10e6,  **optics)

print 'Create bunch for optics...'
bunch   = machine.generate_6D_Gaussian_bunch_matched(
    macroparticlenumber_optics, intensity, epsn_x, epsn_y, sigma_z=sigma_z)
print 'Done.'

bunch.x += 10.
bunch.y += 20.
bunch.z += .020

ix = machine.one_turn_map.index(machine.longitudinal_map)
machine.one_turn_map.remove(machine.longitudinal_map)

beam_alpha_x = []
beam_beta_x = []
beam_alpha_y = []
beam_beta_y = []
for i_ele, m in enumerate(machine.one_turn_map):
    print 'Element %d/%d'%(i_ele, len(machine.one_turn_map))
    beam_alpha_x.append(bunch.alpha_Twiss_x())
    beam_beta_x.append(bunch.beta_Twiss_x())
    beam_alpha_y.append(bunch.alpha_Twiss_y())
    beam_beta_y.append(bunch.beta_Twiss_y())
    m.track(bunch)

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


fig, axes = plt.subplots(2, sharex=True)

axes[0].plot(np.array(beam_beta_x), 'bo')
axes[0].plot(machine.transverse_map.beta_x, 'b-', label='x')
axes[0].plot(np.array(beam_beta_y), 'ro')
axes[0].plot(machine.transverse_map.beta_y, 'r-', label='y')
axes[0].grid('on')
axes[0].set_ylabel('beta_x, beta_y')
axes[0].legend(bbox_to_anchor=(1, 1),loc='upper left',prop={'size':12})
plt.subplots_adjust(right=.86)

axes[1].plot(np.array(beam_alpha_x), 'bo')
axes[1].plot(machine.transverse_map.alpha_x, 'b-')
axes[1].plot(np.array(beam_alpha_y), 'ro')
axes[1].plot(machine.transverse_map.alpha_y, 'r-')
axes[1].grid('on')
axes[1].set_ylabel('alpha_x, alpha_y')
axes[1].set_xlabel('# point')

if mode == 'non-smooth':
    axes[0].plot(np.array(optics['beta_x']), 'xk')
    axes[0].plot(np.array(optics['beta_y']), 'xk')
    axes[1].plot(np.array(optics['alpha_x']), 'xk')
    axes[1].plot(np.array(optics['alpha_y']), 'xk')

plt.show()


machine.one_turn_map.insert(ix, machine.longitudinal_map)
bunch   = machine.generate_6D_Gaussian_bunch_matched(
    macroparticlenumber_track, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

beam_x = []
beam_y = []
beam_z = []
sx, sy, sz = [], [], []
epsx, epsy, epsz = [], [], []
for i_turn in xrange(n_turns):
    print 'Turn %d/%d'%(i_turn, n_turns)
    machine.track(bunch)

    beam_x.append(bunch.mean_x())
    beam_y.append(bunch.mean_y())
    beam_z.append(bunch.mean_z())
    sx.append(bunch.sigma_x())
    sy.append(bunch.sigma_y())
    sz.append(bunch.sigma_z())
    epsx.append(bunch.epsn_x()*1e6)
    epsy.append(bunch.epsn_y()*1e6)
    epsz.append(bunch.epsn_z())

plt.figure(2, figsize=(16, 8), tight_layout=True)
plt.subplot(2,3,1)
plt.plot(beam_x)
plt.ylabel('x [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,2)
plt.plot(beam_y)
plt.ylabel('y [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,3)
plt.plot(beam_z)
plt.ylabel('z [m]');plt.xlabel('Turn')
plt.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
plt.subplot(2,3,4)
plt.plot(np.fft.rfftfreq(len(beam_x), d=1.), np.abs(np.fft.rfft(beam_x)))
plt.ylabel('Amplitude');plt.xlabel('Qx')
plt.subplot(2,3,5)
plt.plot(np.fft.rfftfreq(len(beam_y), d=1.), np.abs(np.fft.rfft(beam_y)))
plt.ylabel('Amplitude');plt.xlabel('Qy')
plt.subplot(2,3,6)
plt.plot(np.fft.rfftfreq(len(beam_z), d=1.), np.abs(np.fft.rfft(beam_z)))
plt.xlim(0, 0.1)
plt.ylabel('Amplitude');plt.xlabel('Qz')

fig, axes = plt.subplots(3, figsize=(16, 8), tight_layout=True)
twax = [plt.twinx(ax) for ax in axes]
axes[0].plot(sx, label=r'$\sigma_x$' )
twax[0].plot(epsx, '-g', label=r'$\varepsilon_x$')
axes[0].set_xlabel('Turns')
axes[0].set_ylabel(r'$\sigma_x$')
twax[0].set_ylabel(r'$\varepsilon_x$')
axes[1].plot(sy, label=r'$\sigma_y$' )
twax[1].plot(epsy, '-g', label=r'$\varepsilon_y$')
axes[1].set_xlabel('Turns')
axes[1].set_ylabel(r'$\sigma_y$')
twax[1].set_ylabel(r'$\varepsilon_y$')
axes[2].plot(sz, label=r'$\sigma_z$' )
twax[2].plot(epsz, '-g', label=r'$\varepsilon_z$')
axes[2].set_xlabel('Turns')
axes[2].set_ylabel(r'$\sigma_z$')
twax[2].set_ylabel(r'$\varepsilon_z$')
axes[0].grid()
axes[1].grid()
axes[2].grid()
for ax in list(axes)+list(twax): 
    ax.ticklabel_format(useOffset=False, style='sci', scilimits=(0,0),axis='y')
for ax in list(axes): 
    ax.legend(loc='upper right',prop={'size':12})
for ax in list(twax): 
    ax.legend(loc='lower right',prop={'size':12})
    
#~ plt.figure(100)
#~ plt.plot(optics['s'][:],optics['beta_x'][:], '-o')

LHC_with_octupole_injection = LHC(machine_configuration='Injection', n_segments=5, octupole_knob = -1.5)
print '450GeV:'
print 'i_octupole_focusing =',LHC_with_octupole_injection.i_octupole_focusing
print 'i_octupole_defocusing =',LHC_with_octupole_injection.i_octupole_defocusing
print 'in the machine we get 19.557'
print '  '
LHC_with_octupole_flattop = LHC(machine_configuration='Injection', n_segments=5, p0=6.5e12*e/c, octupole_knob = -2.9)

print '6.5TeV:'
print 'i_octupole_focusing =',LHC_with_octupole_flattop.i_octupole_focusing
print 'i_octupole_defocusing =',LHC_with_octupole_flattop.i_octupole_defocusing
print 'in the machine we get 546.146'

plt.show()
