from __future__ import division
import cProfile, itertools, sys, time, timeit

from scipy.constants import c, e, m_p

from cobra_functions import stats, random
from beams.bunch import *
from beams import slices
from beams.matching import match_transverse, match_longitudinal
from monitors.monitors import *
from solvers.grid import *
from solvers.poissonfft import *
from impedances.wake_fields import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *


from plots import *


# plt.ion()
n_turns = 100

# Monitors
bunchmonitor = BunchMonitor('bunch', n_turns)
# particlemonitor = ParticleMonitor('particles', n_turns)

# Betatron
n_segments = 1
C = 6911.
s = np.arange(1, n_segments + 1) * C / n_segments
linear_map = TransverseTracker.from_copy(s,
                                         np.zeros(n_segments), np.ones(n_segments) * 100, np.zeros(n_segments),
                                         np.zeros(n_segments), np.ones(n_segments) * 100, np.zeros(n_segments),
                                         26.13, 0, 0, 26.18, 0, 0)

# Synchrotron
cavity = CSCavity(C, 18, 0.017)
# cavity = RFCavity(C, C, 18, 4620, 2e6, 0)

# bunch = Bunch.from_empty(2e3, charge, energy, intensity, mass)
# x, xp, y, yp, dz, dp = random.gsl_quasirandom(bunch)
# Bunch
bunch = bunch_matched_and_sliced(10000, n_particles=1.15e11, charge=1*e, energy=26e9, mass=m_p,
                                 epsn_x=2.5, epsn_y=2.5, ltm=linear_map[0], bunch_length=0.21, bucket=0.5, matching=None,
                                 n_slices=64, nsigmaz=None, slicemode='cspace')
bunch.update_slices()

# PIC grid
poisson = PoissonFFT(plt.std(bunch.x) * 16, plt.std(bunch.y) * 8, 64, 64)
poisson.track(bunch)
t0 = time.clock()
poisson.compute_potential()
print 'Time took', time.clock() - t0, 's'
# [plt.axvline(v, c='orange') for v in poisson.mx[0,:]]
# [plt.axhline(h, c='orange') for h in poisson.my[:,0]]
# plt.gca().set_xlim(plt.amin(poisson.mx), plt.amax(poisson.mx[-1]))
# plt.gca().set_ylim(plt.amin(poisson.my), plt.amax(poisson.my[-1]))
# plt.scatter(bunch.x, bunch.y, marker='.')
# plt.scatter(poisson.mx, poisson.my, s=poisson.rho*2, c=poisson.rho)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)#, sharex=True, sharey=True)
ax1.contour(poisson.fgreen.T, 100)
ax2.plot(poisson.phi[poisson.ny / 2, :poisson.nx], '-g')
ax3.contour(poisson.rho[:poisson.ny, :poisson.nx], 100)
ax3.contour(poisson.phi[:poisson.ny, :poisson.nx], 100, lw=2)
plt.show()
sys.exit(-1)

t0 = time.clock()
poisson.compute_potential_fgreenm2m()
poisson.B= poisson.phi[poisson.ny / 2, :poisson.nx]
print 'Time took', time.clock() - t0, 's'
ax2.plot(poisson.phi[poisson.ny / 2, :poisson.nx])
ax4.plot(poisson.phi[poisson.ny / 2, :poisson.nx])
ax3.contour(poisson.phi, 100, cmap=plt.cm.get_cmap('hsv'))

# plt.gca().set_aspect('equal')
plt.show()
bunchmonitor.h5file.close()
sys.exit(-1)

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
