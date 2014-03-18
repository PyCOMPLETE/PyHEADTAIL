from __future__ import division
from IPython.lib.deepreload import reload as dreload
import cProfile, itertools, ipdb, time, timeit
import numpy as np

from configuration import *
from beams.bunch import *
from beams import slices
from monitors.monitors import *
#~ from solvers.grid import *
from solvers.poissonfft import *
from impedances.wake_fields  import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *
from plots import *
from scipy.constants import c, e, m_p



# simulation setup
charge = 1
mass = m_p
intensity = 1.15e11 
beta_x = 54.6408 # [m]
beta_y = 54.5054 # [m]
bunch_length = 0.3 # [m]
momentum_spread = 0.001053
epsn_x = 2.0 # [um]
epsn_y = 2.0 # [um]
gamma_t = 1/np.sqrt(0.0031)
C = 6911. # [m]
energy = 26 # total [GeV]
n_turns = 500
nsigmaz = 2
Qx = 20.13
Qy = 20.18
Qp_x = 0
Qp_y = 0
n_particles = 100000
n_slices = 500
R_frequency = 1.0e9 # [Hz]
Q = 1.
R_shunt = 20e6 # [Ohm/m]
initial_kick_x = 0.1*np.sqrt(beta_x * epsn_x*1.e-6 / (energy / 0.938))
initial_kick_y = 0.1*np.sqrt(beta_y * epsn_y*1.e-6 / (energy / 0.938))
RF_voltage = 4e6 # [V]
harmonic_number = 4620


# Monitors
bunchmonitor = BunchMonitor('bunch', n_turns)
particlemonitor = ParticleMonitor('particles', n_turns)


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


# Synchrotron
cavity = RFCavity(C, C, gamma_t, harmonic_number, RF_voltage, 0)

# Bunch
#~ bunch = bunch_matched_and_sliced(n_particles, charge, energy, intensity, mass,
                                 #~ epsn_x, epsn_y, linear_map[0], bunch_length, bucket=cavity, matching='simple',
                                 #~ n_slices=n_slices, nsigmaz=nsigmaz, slicemode='cspace') 
bunch =  bunch_unmatched_inbucket_sliced(n_particles, charge, energy, intensity, mass,
                             epsn_x, epsn_y, linear_map[0], bunch_length, momentum_spread, bucket=cavity,
                             n_slices=n_slices, nsigmaz=nsigmaz, slicemode='cspace')                       

# initial transverse kicks
bunch.x += initial_kick_x
bunch.y += initial_kick_y


# Resonator wakefields
wakes = BB_Resonator_Circular(R_shunt=R_shunt, frequency=R_frequency, Q=Q)
#~ wakes = BB_Resonator_ParallelPlates(R_shunt=R_shunt, frequency=R_frequency, Q=Q)


# accelerator map
map_ = [linear_map, [wakes], [cavity]]
map_ = list(itertools.chain.from_iterable(map_))


# define color scale for plotting 
normalization = np.max(bunch.dz) / np.max(bunch.dp)
r = bunch.dz ** 2 + (normalization * bunch.dp) ** 2


plt.ion()
for i in range(n_turns):
    #~ print 'Turn: ', i
    #~ t0 = time.clock() 
    for m in map_:
        #~ t1 = time.clock() 
        m.track(bunch) 
        #~ print m, ', elapsed time: ' + str(time.clock() - t1) + ' s'
    bunchmonitor.dump(bunch)
    #~ particlemonitor.dump(bunch)
    
    #~ plt.clf()
    #~ plt.plot(bunch.slices.mean_x[1:-2]*bunch.slices.charge[1:-2])
    #~ plt.plot(bunch.slices.mean_y)
    #~ plt.gca().set_ylim(-1, 1)
    
    #~ plot_phasespace(bunch, r)
    #~ plot_bunch('bunch-ns1')
    #~ plot_emittance('bunch-ns1')
    #~ plt.draw()
    #~ plt.show()
    
    #~ plt.scatter(bunch.x, bunch.xp)
    #~ plt.draw()
    #~ print 'Turn: ', i, ' took: ' + str(time.clock() - t0) + ' s \n'
    print '{0:4d} \t {1:+3e} \t {2:+3e} \t {3:+3e} \t {4:3e} \t {5:3e} \t {6:3f} \t {7:3f} \t {8:3f}'.format(i, bunch.slices.mean_x[-1], bunch.slices.mean_y[-1], bunch.slices.mean_dz[-1], bunch.slices.epsn_x[-1], bunch.slices.epsn_y[-1], bunch.slices.epsn_z[-1], bunch.slices.sigma_dz[-1], bunch.slices.sigma_dp[-1])


