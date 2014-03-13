from __future__ import division
from IPython.lib.deepreload import reload as dreload
import cProfile, itertools, ipdb, time, timeit


from configuration import *
from cobra_functions import stats, random
from beams.bunch import *
from beams import slices
from beams.matching import match_transverse, match_longitudinal
from monitors.monitors import *
from solvers.grid import *
from solvers.poissonfft import *
from impedances.wake_resonator import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *


from plots import *


# plt.ion()
n_turns = 100

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
cavity = CSCavity(C, 18, 0.017)
# cavity = RFCavity(C, C, 18, 4620, 2e6, 0)

# bunch = Bunch.from_empty(2e3, charge, energy, intensity, mass)
# x, xp, y, yp, dz, dp = random.gsl_quasirandom(bunch)
# Bunch
bunch = bunch_matched_and_sliced(n_particles, charge, energy, intensity, mass,
                                 2.5, 2.5, linear_map[0], 0.21, bucket=0.5, matching=None,
                                 n_slices=64, nsigmaz=None, slicemode='cspace')
bunch.update_slices()

# PIC grid
poisson = PoissonFFT(plt.std(bunch.x) * 10, plt.std(bunch.y) * 10, 128, 128)
# poisson.gather(1, bunch)
[plt.axvline(v, c='r') for v in poisson.mx]
[plt.axhline(h, c='r') for h in poisson.my]
plt.scatter(bunch.x, bunch.y, marker='.')
plt.gca().set_xlim(poisson.mx[0], poisson.mx[-1])
plt.gca().set_ylim(poisson.my[0], poisson.my[-1])
plt.plot(poisson.rho)
plt.show()
exit(-1)

# pdf, bins, patches = plt.hist(bunch.dz, n_slices)
# plt.stem(bunch.slices.dz_centers[:-1], bunch.slices.charge[:-1], linefmt='g', markerfmt='go')
# [plt.axvline(i, c='y') for i in bunch.slices.dz_bins]
# plt.show()
# exit(-1)

# Resonator wakefields
# wakes = WakeResonator(R_shunt=2e6, frequency=1e9, Q=1)

poisson = PoissonFFT(100)

#     plt.scatter(bunch.x, bunch.xp)
#     plt.show()

map_ = [linear_map, [cavity]]
map_ = list(itertools.chain.from_iterable(map_))

t1 = time.clock()
normalization = np.max(bunch.dz) / np.max(bunch.dp)
r = bunch.dz ** 2 + (normalization * bunch.dp) ** 2
for i in range(n_turns):
    # t1 = time.clock()
    for m in map_:
#            t1 = time.clock()
        m.track(bunch)
#            t0 = time.clock() - t1
#            print m, ', elapsed time: ' + str(t0) + ' s'
    bunchmonitor.dump(bunch)
    particlemonitor.dump(bunch)
    plot_phasespace(bunch, r)


# if __name__ == '__main__':
    # cProfile.run('main()', 'stats01.prof')
    # kernprof.py -l main.py
    # main()
