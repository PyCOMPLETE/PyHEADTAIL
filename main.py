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


plt.ion()


# @profile
def track():
    phi1 = plt.zeros((bunch.poisson_other.ny, bunch.poisson_other.nx))
    phi2 = plt.zeros((bunch.poisson_other.ny, bunch.poisson_other.nx))

    index_after_bin_edges = np.cumsum(bunch.slices.n_macroparticles)[:-3]
    index_after_bin_edges[0] = 0

    for i in xrange(bunch.slices.n_slices):
        ix = np.s_[index_after_bin_edges[i]:index_after_bin_edges[i + 1]]

        # Cloud track
        cloud.poisson_self.gather_from(cloud.x, cloud.y, cloud.poisson_self.rho)
        cloud.poisson_self.compute_potential()
        cloud.poisson_self.compute_fields()
        cloud.poisson_self.scatter_to(bunch)

        bunch.poisson_other.gather_from(bunch.x[ix], bunch.y[ix], bunch.poisson_other.rho)
        bunch.poisson_other.compute_potential()
        bunch.poisson_other.compute_fields()
        # bunch.poisson_other.compute_potential_fgreenm2m(bunch.poisson_other.x, bunch.poisson_other.y,
        #                                                 phi1, bunch.poisson_other.rho)
        # bunch.poisson_other.compute_potential_fgreenp2m(bunch.x, bunch.y,
        #                                                 bunch.poisson_other.x, bunch.poisson_other.y,
        #                                                 phi2, bunch.poisson_other.rho)
        bunch.poisson_other.scatter_to(cloud)

        cloud.push(bunch, ix)


        [ax.cla() for ax in (ax1, ax2, ax3, ax4)]
        # [ax.set_aspect('equal') for ax in (ax1, ax2, ax3, ax4)]
        ax1.contour(p.fgreen.T, 100)
        ax2.plot(p.phi[p.ny / 2,:], '-g')
        ax2.plot(phi1[p.ny / 2,:], '-r')
        ax2.plot(phi2[p.ny / 2,:], '-', c='orange')
        ax3.contourf(p.x, p.y, -10 * plt.log10(p.rho), 100)
        # ax3.scatter(cloud.x[::20], cloud.y[::20], c='b', marker='.')
        # ax3.quiver(cloud.x[::50], cloud.y[::50], cloud.kx[::50], cloud.ky[::50], color='g')
        # ax3.contour(p.x, p.y, p.phi, 100, lw=2)
        ax3.scatter(bunch.x[ix], bunch.y[ix], c='y', marker='.', alpha=0.8)
        ax4.imshow(plt.sqrt(bunch.poisson_other.ex ** 2 + bunch.poisson_other.ey ** 2), origin='lower', aspect='auto', extent=(p.x[0,0], p.x[0,-1], p.y[0,0], p.y[-1,0]))

        plt.draw()

    return phi1, phi2


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
bunch = bunch_matched_and_sliced(100000, n_particles=1.15e11, charge=1*e, energy=26e9, mass=m_p,
                                 epsn_x=2.5, epsn_y=2.5, ltm=linear_map[0], bunch_length=0.21, bucket=0.5, matching=None,
                                 n_slices=64, nsigmaz=None, slicemode='cspace')
bunch.update_slices()

# Cloud
cloud = Cloud.from_parameters(100000, 5e11, plt.std(bunch.x) * 20, plt.std(bunch.y) * 20, C)
cloud.add_poisson(plt.std(bunch.x) * 20, plt.std(bunch.y) * 20, 128, 128, other=bunch)

# PIC grid
# poisson = PoissonFFT(plt.std(bunch.x) * 16, plt.std(bunch.y) * 8, 64, 128)
# poisson.inject(master=cloud, slave=bunch)
# Test the PIC here!
t0 = time.clock()
print 'Time took', time.clock() - t0, 's'

# Plot results
# [plt.axvline(v, c='orange') for v in poisson.mx[0,:]]
# [plt.axhline(h, c='orange') for h in poisson.my[:,0]]
# plt.gca().set_xlim(plt.amin(poisson.mx), plt.amax(poisson.mx[-1]))
# plt.gca().set_ylim(plt.amin(poisson.my), plt.amax(poisson.my[-1]))
# plt.scatter(bunch.x, bunch.y, marker='.')
# plt.scatter(poisson.mx, poisson.my, s=poisson.rho*2, c=poisson.rho)

p = cloud.poisson_self
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,12))#, sharex=True, sharey=True)
phi1, phi2 = track()
# ax1.contour(p.fgreen.T, 100)
# ax2.plot(p.phi[p.ny / 2,:], '-g')
# ax2.plot(phi1[p.ny / 2,:], '-r')
# ax2.plot(phi2[p.ny / 2,:], '-', c='orange')
# ax3.contourf(p.x, p.y, p.rho, 100)
# ax3.contour(p.x, p.y, p.phi, 100, lw=2)
# ax3.scatter(bunch.x, bunch.y, marker='.', c='y', alpha=0.8)
# ax4.imshow(p.ex, origin='lower', aspect='auto', extent=(p.x[0,0], p.x[0,-1], p.y[0,0], p.y[-1,0]))

# p = cloud.poisson_self
# fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)#, sharex=True, sharey=True)
# ax1.contour(p.fgreen.T, 100)
# ax2.plot(p.phi[p.ny / 2,:], '-g')
# ax3.contourf(p.x, p.y, p.rho, 100)
# ax3.contour(p.x, p.y, p.phi, 100, lw=2)
# ax3.scatter(cloud.x, cloud.y, marker='.', c='y', alpha=0.8)
# ax4.imshow(p.ex, origin='lower', aspect='auto', extent=(p.x[0,0], p.x[0,-1], p.y[0,0], p.y[-1,0]))
plt.show()
sys.exit(-1)


# pdf, bins, patches = plt.hist(bunch.dz, n_slices)
# plt.stem(bunch.slices.dz_centers[:-1], bunch.slices.charge[:-1], linefmt='g', markerfmt='go')
# [plt.axvline(i, c='y') for i in bunch.slices.dz_bins]
# plt.show()
# exit(-1)

# Resonator wakefields
# wakes = WakeResonator(R_shunt=2e6, frequency=1e9, Q=1)


map_ = linear_map +  [cavity]

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
