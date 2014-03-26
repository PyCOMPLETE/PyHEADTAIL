from __future__ import division
from IPython.lib.deepreload import reload as dreload
import cProfile, itertools, ipdb, time, timeit
import numpy as np

from beams.bunch import *
from beams import slices
from monitors.monitors import *
from aperture.aperture import *
from impedances.wake_fields  import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *
from plots import *
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants


# simulation setup
charge = e
mass = m_p
n_particles = 1.95e11 
beta_x = 54.6408 # [m]
beta_y = 54.5054 # [m]
bunch_length = 0.2 # [m]
momentum_spread = 0.001
epsn_x = 2.0 # [um]
epsn_y = 2.0 # [um]
gamma_t = 1/np.sqrt(0.00192)
C = 6911. # [m]
energy = 26e9 # total [eV]
n_turns = 500
nsigmaz = 3
Qx = 26.13
Qy = 26.18
Qp_x = 0
Qp_y = 0
n_macroparticles = 100000
n_slices = 500
R_frequency = 1.0e9 # [Hz]
Q = 1.
R_shunt = 0.23e6 # [Ohm/m]
initial_kick_x = 0.1 * np.sqrt(beta_x * epsn_x*1.e-6 / (energy / 0.938e9))
initial_kick_y = 0.1 * np.sqrt(beta_y * epsn_y*1.e-6 / (energy / 0.938e9))
RF_voltage = 0.7e6 # [V]
harmonic_number = 4620


# Monitors
bunchmonitor = BunchMonitor('bunch', n_turns)
#~ particlemonitor = ParticleMonitor('particles', n_turns)


# Betatron
n_segments = 1
s = np.arange(1, n_segments + 1) * C / n_segments
linear_map = TransverseTracker.from_copy(s,
                               np.zeros(n_segments),
                               np.ones(n_segments) * beta_x,
                               np.zeros(n_segments),
                               np.zeros(n_segments),
                               np.ones(n_segments) * beta_y,
                               np.zeros(n_segments),
                               Qx, Qp_x, 0, Qy, Qp_y, 0)


# Synchrotron motion
cavity = RFCavity(C, C, gamma_t, harmonic_number, RF_voltage, 0, integrator='rk4')


# Bunch
#~ bunch = bunch_matched_and_sliced(n_macroparticles, n_particles, charge, energy, mass,
                                 #~ epsn_x, epsn_y, linear_map[0], bunch_length, bucket=cavity, matching='simple',
                                 #~ n_slices=n_slices, nsigmaz=nsigmaz, slicemode='cspace') 
bunch =  bunch_unmatched_inbucket_sliced(n_macroparticles, n_particles, charge, energy, mass,
                             epsn_x, epsn_y, linear_map[0], bunch_length, momentum_spread, bucket=cavity,
                             n_slices=n_slices, nsigmaz=nsigmaz, slicemode='cspace')                      
# initial transverse kicks
bunch.x += initial_kick_x
bunch.y += initial_kick_y

#~ # save initial distribution
#~ ParticleMonitor('initial_distribution').dump(bunch)

# distribution from file
#~ bunch = bunch_from_file('initial_distribution', 0, n_particles, charge, energy, mass, n_slices, nsigmaz, slicemode='cspace')


# Resonator wakefields
#~ wakes = BB_Resonator_Circular(R_shunt=R_shunt, frequency=R_frequency, Q=Q)
#~ wakes = BB_Resonator_ParallelPlates(R_shunt=R_shunt, frequency=R_frequency, Q=Q)
wakes = BB_Resonator_longitudinal(R_shunt=R_shunt, frequency=R_frequency, Q=Q)

# Resistive wall
#~ wakes = Resistive_wall_Circular(pipe_radius=0.05, length_resistive_wall=C, conductivity=5.4e17)
    
# Wakefield from file
#~ wakes = Wake_table.from_ASCII('BenchMark_WakeTable.wake', ['time', 'dipolar_x', 'dipolar_y', 'quadrupolar_x', 'quadrupolar_y'])


# aperture
aperture = Longitudinal_aperture([C / harmonic_number / 2])


# accelerator map
map_ = [wakes] + [aperture] + [cavity]

# define color scale for plotting 
normalization = np.max(bunch.dz) / np.max(bunch.dp)
r = bunch.dz ** 2 + (normalization * bunch.dp) ** 2


#~ max_x = 0
#~ max_xp = 0
plt.ion()
for i in range(n_turns):
    #~ print 'Turn: ', i
    t0 = time.clock() 
    for m in map_:
        #~ t1 = time.clock() 
        m.track(bunch) 
        #~ print m, ', elapsed time: ' + str(time.clock() - t1) + ' s'
    bunchmonitor.dump(bunch)
    print '{0:4d} \t {1:+3e} \t {2:+3e} \t {3:+3e} \t {4:3e} \t {5:3e} \t {6:3f} \t {7:3f} \t {8:3f} \t {9:4e} \t {10:3s}'.format(i, bunch.slices.mean_x[-2], bunch.slices.mean_y[-2], bunch.slices.mean_dz[-2], bunch.slices.epsn_x[-2], bunch.slices.epsn_y[-2], bunch.slices.epsn_z[-2], bunch.slices.sigma_dz[-2], bunch.slices.sigma_dp[-2], bunch.slices.n_macroparticles[-2] / bunch.n_macroparticles * bunch.n_particles, str(time.clock() - t0))
    
    #~ particlemonitor.dump(bunch)
    #~ aperture.limit_y -= 2e-5
    #~ plt.clf()
    #~ plt.plot(bunch.slices.n_macroparticles)
    #~ plt.plot(bunch.in_slice)
    #~ plt.plot(bunch.y[bunch.identity>0], bunch.yp[bunch.identity>0],'.')
    
    #plt.figure(1)
    #~ plt.plot(i,bunch.slices.n_macroparticles[-1], '.')
    #plt.plot(i,bunch.slices.n_macroparticles[-2] / bunch.n_macroparticles, '.')
    #~ plt.ylim(0, None)
    #plt.xlim(0, n_turns)    
    #~ plt.ylim(None, n_macroparticles)
    #~ plt.figure(2)
    #~ plt.clf()
    #~ plt.plot(bunch.slices.mean_y[1:-3]*bunch.slices.n_macroparticles[1:-3])
    #~ plt.plot(i, aperture.limit_y, '.')
    #~ plt.xlim(0, n_turns)
    #~ plt.ylim(0, None)
    #~ plt.ylim(-10, 10)
    #~ plt.figure(3)
    #~ plt.clf()
    #~ plot_bunch('bunch')
    #~ plt.figure(4)
    #~ plt.clf()
    #~ max_x = max(np.max(np.abs(bunch.x)), max_x)
    #~ max_xp = max(np.max(np.abs(bunch.xp)), max_xp)
    #~ plt.plot(bunch.x[:-1-bunch.slices.n_macroparticles[-1]:10], bunch.xp[:-1-bunch.slices.n_macroparticles[-1]:10],'.')
    #~ plt.plot(bunch.x[bunch.slices.n_macroparticles[-2]::10], bunch.xp[bunch.slices.n_macroparticles[-2]::10],'r.')
    #~ plt.xlim(-max_x,max_x)
    #~ plt.ylim(-max_xp,max_xp)
    
    #~ ax = plt.gca()
    #~ ax.set_ylim(-10., 10.)
    #~ plt.plot(bunch.slices.mean_y)
    #~ plt.gca().set_ylim(-1, 1)
    
    #~ plt.figure(5)
    #~ plt.clf()
    #~ plt.plot(bunch.slices.mean_dz[1:-3]/c, bunch.slices.n_macroparticles[1:-3])
    
    #plt.figure(6)
    #plt.plot(i, bunch.slices.sigma_dz[-2],'.')
    #~ if np.mod(i,20) == 0:
        #~ plt.figure(7)
        #~ plot_phasespace(bunch, r)
    #~ plt.clf()
    #~ ax = plt.gca()
    #~ ax.scatter(bunch.dz, bunch.dp, 10, bunch.identity, marker='.', lw=0)
    #~ ax.set_xlim(-1., 1.)
    #~ ax.set_ylim(-0.5e-2, 0.5e-2)
    #~ plt.draw()
    #~ plot_bunch('bunch')
    #~ plot_emittance('bunch-ns1')
    #~ plt.draw()
    #~ plt.show()
    

    
    #~ plt.scatter(bunch.x, bunch.xp)
    plt.draw()
    #~ print 'Turn: ', i, ' took: ' + str(time.clock() - t0) + ' s \n'


bunchmonitor.h5file.close()
