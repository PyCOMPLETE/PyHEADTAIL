from __future__ import division
import cProfile, itertools, sys, time, timeit

from scipy.constants import c, e, m_p

import beams.beam as beam
from cobra_functions import stats, random
from beams.bunch import *
from beams.matching import match_transverse, match_longitudinal
from monitors.monitors import *
from spacecharge.spacecharge import *
from impedances.wake_fields import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *


from plots import *


# plt.ion()
n_turns = 20

# Monitors
bunchmonitor = BunchMonitor('bunch', n_turns)
# particlemonitor = ParticleMonitor('particles', n_turns)

# Betatron
n_segments = 1
C = 6911.
s = np.arange(1, n_segments + 1) * C / n_segments
linear_map = TransverseTracker.from_copy(s,
                                         np.zeros(n_segments), np.ones(n_segments) * 54.6, np.zeros(n_segments),
                                         np.zeros(n_segments), np.ones(n_segments) * 54.6, np.zeros(n_segments),
                                         20.13, 0, 0, 20.18, 0, 0)

# Synchrotron
cavity = CSCavity(C, 18, 0.017)
# cavity = RFCavity(C, C, 18, 4620, 2e6, 0)

# bunch = Bunch.from_empty(2e3, charge, energy, intensity, mass)
# x, xp, y, yp, dz, dp = random.gsl_quasirandom(bunch)
# Bunch
bunch = bunch_matched_and_sliced(100, n_particles=1.15e11, charge=1*e, energy=26e9, mass=m_p,
                                 epsn_x=2.5, epsn_y=2.5, ltm=linear_map[0], bunch_length=0.23, bucket=0.5, matching=None,
                                 n_slices=64, nsigmaz=3, slicemode='cspace')
# bunch.update_slices()

slices1 = beam.Slices(10, 2, mode='cspace')
slices1.update_slices(bunch)
# print slices1.n_cut_tail, slices1.n_cut_head, slices1.n_macroparticles, sum(slices1.n_macroparticles) + slices1.n_cut_tail + slices1.n_cut_head
# print slices1.z_index

plt.hist(bunch.dz, 10)
[plt.axvline(z, c='m') for z in slices1.z_bins]
# [plt.axvline(z, c='g') for z in slices1.z_centers]
plt.plot(bunch.dz, plt.ones(bunch.n_macroparticles) * 2, 'o')
[plt.plot(bunch.dz[i], 2, marker='x', c='y', ms=20) for i in slices1.z_index]
plt.stem(slices1.z_centers, slices1.n_macroparticles)
plt.show()

exit(-1)

# Cloud
ecloud = Cloud.from_parameters(100000, 10e11, plt.std(bunch.x) * 20, plt.std(bunch.y) * 20, C)

# print plt.std(bunch.x) * 20, plt.std(bunch.y) * 20, C
# exit(-1)

# Space charge
cloud = SpaceCharge(ecloud, 'cloud', plt.std(bunch.x) * 20, plt.std(bunch.y) * 20, 128, 128)
# Test the PIC here!

# pdf, bins, patches = plt.hist(bunch.dz, n_slices)
# plt.stem(bunch.slices.dz_centers[:-1], bunch.slices.charge[:-1], linefmt='g', markerfmt='go')
# [plt.axvline(i, c='y') for i in bunch.slices.dz_bins]
# plt.show()
# exit(-1)

# Resonator wakefields
# wakes = WakeResonator(R_shunt=2e6, frequency=1e9, Q=1)
map_ = list(itertools.chain.from_iterable([[l] + [cloud] for l in linear_map] + [[cavity]]))

t1 = time.clock()
normalization = np.max(bunch.dz) / np.max(bunch.dp)
r = bunch.dz ** 2 + (normalization * bunch.dp) ** 2
for i in range(n_turns):
    t0 = time.clock()

    for m in map_:
        m.track(bunch)

    print 'Turn',i,': elapsed time: ' + '{:3f}'.format(time.clock() - t0) + ' s'
    bunchmonitor.dump(bunch)
    # particlemonitor.dump(bunch)
    # plot_phasespace(bunch, r)


# if __name__ == '__main__':
    # cProfile.run('main()', 'stats01.prof')
    # kernprof.py -l main.py
    # main()
