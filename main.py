from __future__ import division
from IPython.lib.deepreload import reload as dreload
import cProfile, itertools, ipdb, time, timeit


from configuration import *
from beams.bunch import *
from monitors.monitors import *
from solvers.poissonfft import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *


from plots import *


plt.ion()
tmp_mean_x = []
tmp_mean_y = []
tmp_epsn_x = []
tmp_epsn_y = []
tmp_mean_dz = []
tmp_epsn_z = []

n_turns = 20

# Monitors
bunchmonitor = BunchMonitor('bunch', n_turns)
particlemonitor = ParticleMonitor('particles', n_turns)

# Bunch
bunch = Bunch.from_parameters(n_particles, charge, energy, intensity, mass,
            epsn_x, beta_x, epsn_y, beta_y, epsn_z, length)
bunch.slice(n_slices, nsigmaz=None, mode='cspace')

# Betatron
n_segments = 10
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
V = 2e6
f = 200e6
gamma_tr = 18
Qs = 0.017
cavity = CSCavity(C, gamma_tr, Qs)
# cavity = RFCavity(C, gamma_tr, f, V, 0)

poisson = PoissonFFT(100)

#     plt.scatter(bunch.x, bunch.xp)
#     plt.show()

map_ = [linear_map, [cavity]]
map_ = list(itertools.chain.from_iterable(map_))

t1 = time.clock()
for i in range(n_turns):
    # t1 = time.clock()
    for m in map_:
#            t1 = time.clock()
        m.track(bunch)
#            t0 = time.clock() - t1
#            print m, ', elapsed time: ' + str(t0) + ' s'
    bunchmonitor.dump(bunch)
    # plot_phasespace(bunch)
    tmp_mean_x.append(bunch.slices.mean_x[-1])
    tmp_epsn_x.append(bunch.slices.epsn_x[-1])
    tmp_mean_y.append(bunch.slices.mean_y[-1])
    tmp_epsn_y.append(bunch.slices.epsn_y[-1])
    tmp_mean_dz.append(bunch.slices.mean_dz[-1])
    tmp_epsn_z.append(bunch.slices.epsn_z[-1])

figure(figsize=(16,8))
ax1 = subplot(131)
ax2 = subplot(132)
ax3 = subplot(133)
ax1.plot(tmp_mean_x)
ax1.plot(tmp_mean_y)
ax2.plot(tmp_epsn_x)
ax2.plot(tmp_epsn_y, ls='--')
ax2.set_ylim(0, 5)
ax3.plot(tmp_mean_dz)
ax3.plot(tmp_epsn_z)
show()


# if __name__ == '__main__':
    # cProfile.run('main()', 'stats01.prof')
    # kernprof.py -l main.py
    # main()
