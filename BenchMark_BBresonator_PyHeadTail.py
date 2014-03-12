from __future__ import division
from IPython.lib.deepreload import reload as dreload
import cProfile, itertools, ipdb, time, timeit
import numpy as np

from beams.bunch import *
from monitors.monitors import *
from solvers.poissonfft import *
from impedances.wake_resonator import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *

from plots import *

# constants and simulation parameters
from scipy import constants


# Constants
c = constants.c
e = constants.e
mp = constants.m_p


# simulation setup
charge = 1
mass = mp
intensity = 1e11
beta_x = 54.6408
beta_y = 54.5054
bunch_length = 0.3
epsn_z = 0.5
epsn_x = 2.0
epsn_y = 2.0
gamma_t = 1/np.sqrt(0.0031)
C = 6911.
energy = 26
n_turns = 10
nsigmaz = 2
Qx = 20.13
Qy = 20.18
Qp_x = 0
Qp_y = 0
n_particles = 1000000
n_slices = 500
R_frequency = 1.0e9
Q = 1
R_shunt = 20e6
initial_kick_x = 0.1*np.sqrt(beta_x * epsn_x*1.e-6 / (energy / 0.938))
initial_kick_y = 0.1*np.sqrt(beta_y * epsn_y*1.e-6 / (energy / 0.938))
RF_voltage = 4e6
harmonic_number = 4620


plt.ion()

# Monitors
bunchmonitor = BunchMonitor('bunch-ns1', n_turns)
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
bunch = Bunch.from_parameters(n_particles, charge, energy, intensity, mass,
                              epsn_x, beta_x, epsn_y, beta_y, epsn_z, bunch_length, cavity=cavity, matching='simple')
                                                                                        
# initial transverse kicks
bunch.x += initial_kick_x
bunch.y += initial_kick_y

# slicing
bunch.slice(n_slices, nsigmaz=nsigmaz, mode='cspace')
bunch.compute_statistics()

# Resonator wakefields
wakes = BB_Resonator_Circular(R_shunt=R_shunt, frequency=R_frequency, Q=Q)
#~ wakes = BB_Resonator_ParallelPlates(R_shunt=R_shunt, frequency=R_frequency, Q=Q)


# accelerator map
map_ = [[wakes], linear_map, [cavity]]
#~ map_ = [[wakes]]
map_ = list(itertools.chain.from_iterable(map_))


# define color scale for plotting 
normalization = np.max(bunch.dz) / np.max(bunch.dp)
r = bunch.dz ** 2 + (normalization * bunch.dp) ** 2

for i in range(n_turns):
    print 'Turn: ', i
    t0 = time.clock() 
    for m in map_:
        t1 = time.clock() 
        m.track(bunch) 
        print m, ', elapsed time: ' + str(time.clock() - t1) + ' s'
    t1 = time.clock()           
    bunch.slice(n_slices, nsigmaz=nsigmaz, mode='cspace')
    print 'bunch slicing, elapsed time: ' + str(time.clock() - t1) + ' s'
    t1 = time.clock()
    bunchmonitor.dump(bunch)
    print 'saving bunch, elapsed time: ' + str(time.clock() - t1) + ' s'
    #~ t1 = time.clock()
    #~ particlemonitor.dump(bunch)
    #~ print 'saving particles, elapsed time: ' + str(time.clock() - t1) + ' s'
    
    #~ plt.clf()
    #~ plt.plot(bunch.slices.mean_x)
    #~ plt.plot(bunch.slices.mean_y)
    #~ plt.gca().set_ylim(-0.1, 0.1)

    
     
    #~ plot_phasespace(bunch, r)
    #~ plot_bunch('bunch-ns1')
    #~ plot_emittance('bunch-ns1')
    #~ plt.draw()
    #~ plt.show()
    
    #~ plt.scatter(bunch.x, bunch.xp)
    #~ plt.draw
    print 'Turn: ', i, ' took: ' + str(time.clock() - t0) + ' s \n'
    



