from __future__ import division
from IPython.lib.deepreload import reload as dreload
import cProfile, itertools, ipdb, time, timeit


from configuration import *
from beams.bunch import *
from monitors.monitors import *
from solvers.poissonfft import *
from impedances.wake_resonator import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *


from plots import *


plt.ion()
n_turns = 1000

# Monitors
bunchmonitor = BunchMonitor('bunch-ns1', n_turns)
particlemonitor = ParticleMonitor('particles', n_turns)

# Betatron
n_segments = 1
C = 6911.
s = np.arange(1, n_segments + 1) * C / n_segments
linear_map = TransverseTracker.from_copy(s,
                               np.zeros(n_segments),
                               np.ones(n_segments) * 100,
                               np.zeros(n_segments),
                               np.zeros(n_segments),
                               np.ones(n_segments) * 100,
                               np.zeros(n_segments),
                               26.13, 0, 0, 26.18, 0, 0)

# Synchrotron
#~ cavity = CSCavity(C, 18, 0.017)
cavity = RFCavity(C, C, 18, 4620, 2e6, 0)

# Bunch
bunch = Bunch.from_parameters(n_particles, charge, energy, intensity, mass,
                              epsn_x, beta_x, epsn_y, beta_y, epsn_z, length=0.220, cavity=cavity, matching='simple')
# initial kick in horizontal
bunch.x += 0.1

# slicing
bunch.slice(n_slices, nsigmaz=None, mode='cspace')
bunch.compute_statistics()


# Resonator wakefields
wakes = BB_Resonator_Circular(R_shunt=10e7, frequency=1e6, Q=1)


#~ poisson = PoissonFFT(100)

#~ plt.scatter(bunch.x, bunch.xp)
#~ plt.draw

map_ = [linear_map, [wakes]]
#~ map_ = [linear_map, [wakes], [cavity]]
#~ map_ = [linear_map, [cavity]]
map_ = list(itertools.chain.from_iterable(map_))

t1 = time.clock()
normalization = np.max(bunch.dz) / np.max(bunch.dp)
r = bunch.dz ** 2 + (normalization * bunch.dp) ** 2

    
for i in range(n_turns):
    print 'Turn: ', i
    # t1 = time.clock()
    for m in map_:
#            t1 = time.clock()
        m.track(bunch)
#            t0 = time.clock() - t1
#            print m, ', elapsed time: ' + str(t0) + ' s'
    bunch.slice(n_slices, nsigmaz=None, mode='cspace')
    bunchmonitor.dump(bunch)
    particlemonitor.dump(bunch)
    
    plt.clf()
    plt.plot(bunch.slices.mean_x)
    plt.gca().set_ylim(-10, 10)
    plt.draw()
    
    #~ print bunch.slices.mean_x
    
    #~ plot_phasespace(bunch, r)

    
    #~ plt.scatter(bunch.x, bunch.xp)
    #~ plt.draw


# if __name__ == '__main__':
    # cProfile.run('main()', 'stats01.prof')
    # kernprof.py -l main.py
    # main()


