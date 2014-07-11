from __future__ import division
import cProfile, itertools, sys, time, timeit

from scipy.constants import c, e, m_p

from particles.particles import *
from particles.slicer import *
from cobra_functions import stats, random
from monitors.monitors import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *

import pylab as plt


# plt.ion()


# Parameters
# ==========
C = 6911
R = C / (2 * np.pi)
gamma_tr = 18
gamma = 27.7
eta = 1 / gamma_tr ** 2 - 1 / gamma ** 2
Qx = 20.13
Qy = 20.18
Qs = 0.017
beta_x = 54.6
beta_y = 54.6
beta_z = eta * R / Qs
epsn_x = 2.5
epsn_y = 2.5
epsn_z = 0.5

n_turns = 100

# Object definitions
# ==================
# Bunch - TODO: fix initialisation of longitudinal phase space
bunch = Particles.as_gaussian(10000, e, gamma, 1.15e11, m_p, 0, beta_x, epsn_x, 0, beta_y, epsn_y, beta_z, epsn_z)
# ecloud = Cloud.from_parameters(100000, 10e11, plt.std(bunch.x) * 20, plt.std(bunch.y) * 20, C)

# Slices - the last particle is always in cut_head
slices1 = Slicer(16, mode='const_space'); slices1.update_slices(bunch)
print slices1.n_cut_tail, slices1.n_cut_head, slices1.n_macroparticles, sum(slices1.n_macroparticles) + slices1.n_cut_tail + slices1.n_cut_head
slices2 = Slicer(16, mode='const_charge'); slices2.update_slices(bunch)
print slices2.n_cut_tail, slices2.n_cut_head, slices2.n_macroparticles, sum(slices2.n_macroparticles) + slices2.n_cut_tail + slices2.n_cut_head


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
ax1.plot(bunch.x, bunch.y, 'g.')
ax2.plot(bunch.z, bunch.dp, 'g.')

ax3.hist(bunch.z, 16, color='r', alpha=0.6)
[ax3.axvline(z, c='m') for z in slices1.z_bins]
ax3.plot(bunch.z, plt.ones(bunch.n_macroparticles) * 2, 'o')
[ax3.plot(bunch.z[i], 2, marker='x', c='y', ms=20) for i in slices1.z_index]
ax3.stem(slices1.z_centers, slices1.n_macroparticles)

[ax4.axvline(z, c='m') for z in slices2.z_bins]
ax4.plot(bunch.z, plt.ones(bunch.n_macroparticles) * 2, 'o')
[ax4.plot(bunch.z[i], 2, marker='x', c='y', ms=20) for i in slices2.z_index]
ax4.stem(slices2.z_centers, slices2.n_macroparticles)


plt.suptitle('%e'%bunch.intensity)
plt.show()
exit(-1)
