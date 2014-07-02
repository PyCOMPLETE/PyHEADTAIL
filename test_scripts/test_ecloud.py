from __future__ import division



import itertools, time

from ecloud.ecloud import *
from particles.particles import *
from particles.slicer import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *

from scipy.constants import e, m_e

import pylab as plt

plt.ion()
# Parameters
# ==========
n_macroparticles = 100000

C = 6911.
R = C / (2 * np.pi)
gamma_tr = 18.
gamma = 27.7
eta = 1 / gamma_tr ** 2 - 1 / gamma ** 2
Qx = 20.13
Qy = 20.18
Qs = 0.017
beta_x = 54.6
beta_y = 54.6
beta_z = np.abs(eta) * R / Qs
epsn_x = 2.5
epsn_y = 2.5
epsn_z = 0.5

n_turns = 5

# Beam
bunch = Particles.as_gaussian(10000, e, gamma, 1.15e11, m_p, 0, beta_x, epsn_x, 0, beta_y, epsn_y, beta_z, epsn_z)

# Betatron
n_segments = 1
s = np.arange(n_segments + 1) * C / n_segments
ltm = TransverseTracker.from_copy(s,
    np.zeros(n_segments), np.ones(n_segments) * beta_x, np.zeros(n_segments),
    np.zeros(n_segments), np.ones(n_segments) * beta_y, np.zeros(n_segments),
    Qx, 0, 0, 0, Qy, 0, 0, 0)

# Synchrotron
alpha = 1 / gamma_tr ** 2
cavity = LinearMap(C, alpha, Qs)

# E-cloud
e_density = 2e11 # electrons per m^3
x_max = 20 * plt.std(bunch.x)
x_min = -x_max
y_max = 20 * plt.std(bunch.y)
y_min = -y_max

grid_extension_x = x_max
grid_extension_y = y_max
grid_nx = 128
grid_ny = 128

N_electrons = e_density * (x_max - x_min) * (y_max - y_min) * C / n_segments

particles = Particles.as_uniformXY(n_macroparticles, -e, 1., N_electrons, m_e, x_min, x_max, y_min, y_max)
#~ plt.plot(particles.x, particles.y, '.')
#~ plt.show()
slicer = Slicer(64, nsigmaz=3)
ecloud = Ecloud(particles, grid_extension_x, grid_extension_y, grid_nx, grid_ny, slicer)


map_ = list(itertools.chain.from_iterable([[l] + [ecloud] for l in ltm] + [[cavity]]))

r = 10 * plt.log10(bunch.z ** 2 + (beta_z * bunch.dp) ** 2)
for i in range(n_turns):
    t0 = time.clock()
    for m in map_:
        m.track(bunch)

    #~ bunchmonitor.dump(bunch)
    print '--> Elapsed time:', time.clock() - t0
        # plt.cla()
        # plt.scatter(bunch.z, bunch.dp, c=r, marker='o', lw=0)
        # plt.gca().set_xlim(-1, 1)
        # plt.gca().set_ylim(-1e-2, 1e-2)
        # plt.draw()



